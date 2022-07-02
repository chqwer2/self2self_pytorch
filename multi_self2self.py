###########################################################
# This script tests a modified lenet5.
###########################################################

import numpy as np # linear algebra
import logging
import os
import glob
import time
from matplotlib import cm
from matplotlib import pyplot as plt
import cv2
import util
from util import timeSince
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset,DataLoader
from torchvision import utils,models
import torch.nn.functional as F
import torchvision.transforms as T
from sklearn.model_selection import train_test_split
from PIL import Image
from tqdm import trange
from time import sleep
from scipy.io import loadmat
import torchvision.datasets as dset
from torch.utils.data import sampler
from torch.nn.parallel import DistributedDataParallel as DDP
import torch.multiprocessing as mp
import torch.distributed as dist
from partialconv2d import PartialConv2d
from model import self2self
from config import CFG
import albumentations as A
from albumentations.pytorch import ToTensorV2


# os.environ["CUDA_VISIBLE_DEVICES"]="4"

cfg = CFG()
if not os.path.exists(cfg.expdir):
        os.mkdir(cfg.expdir)  
if not os.path.exists(cfg.imgdir):
    os.mkdir(cfg.imgdir)    
if not os.path.exists(cfg.mdldir):
    os.mkdir(cfg.mdldir)  

##Enable GPU
if cfg.USE_GPU and torch.cuda.is_available():
        device = torch.device('cuda')
else:
    device = torch.device('cpu')
  

  

def image_fliper(image, p1, p2):
    # print("fliper size:", image.shape)  # fliper size: torch.Size([3, 512, 512, 3]
    if p1 > cfg.flip_rate:
        image = torch.flip(image, dims=(2,))
    if p2 > cfg.flip_rate:
        image = torch.flip(image, dims=(1,))
        
    return image
    
    
transforms = None
transforms = {
    "train": A.OneOf([A.Compose([
            A.GaussianBlur(p=1.0),
            A.RandomBrightnessContrast(brightness_limit=[-0.2, 0], brightness_by_max=True, always_apply=False, p=1.0),  #[-0.2,0]
            # A.Sharpen(alpha=(0, 0.1), lightness=(0.1, 0.3), p=1.0),
            A.MedianBlur(p=1.0),
        
            # A.ISONoise (color_shift=(0.01, 0.05), intensity=(0.1, 0.5), always_apply=False, p=0.5),        
            # A.HueSaturationValue(hue_shift_limit=5, sat_shift_limit=5, val_shift_limit=0, always_apply=False, p=0.5) ], p=1.0),
            # GaussNoise (var_limit=(10.0, 50.0), mean=0, per_channel=True, always_apply=False, p=0.5) 
            # albumentations.augmentations.transforms.MotionBlur
            # albumentations.augmentations.transforms.
            # albumentations.augmentations.transforms.PixelDropout
            # albumentations.augmentations.transforms.RandomSunFlare
            # albumentations.augmentations.transforms.UnsharpMask
  
        ], p=1.0),])
}



def train_self2self(imgdir, img_channel, p, sigma=-1, is_realnoisy = False):
    
    
    # =========== Prepare
    cfg.imgdir = os.path.join(cfg.imgdir, imgdir.split('/')[-1] + cfg.imgtag)
    cfg.mdldir = os.path.join(cfg.mdldir, imgdir.split('/')[-1] + cfg.imgtag)
    
    if not os.path.exists(cfg.imgdir):
        os.mkdir(cfg.imgdir)
  
    if not os.path.exists(cfg.mdldir):
        os.mkdir(cfg.mdldir)
        
    logging.basicConfig(filename=os.path.join(cfg.imgdir, 'train.log'),
                    filemode='a',
                    format='%(asctime)s,%(msecs)d %(name)s %(levelname)s %(message)s',
                    datefmt='%H:%M:%S',
                    level=logging.DEBUG)
        
    model = self2self(img_channel, p)
    model = model.cuda()
 
    optimizer = optim.Adam(model.parameters(), lr = cfg.learning_rate)

    # ========= Start ========
    # Add Noise
    img = np.array(Image.open(imgdir))[:320, :480,:]    # 321, 481
    img = util.add_gaussian_noise(img, cfg.imgdir, sigma)   

    img_expand = np.expand_dims(img, 0)
    img_expand = np.repeat(img_expand, cfg.batch, axis=0)
    
    
    print("img.shape:", img_expand.shape, np.max(img_expand))      # 512, 512, 3, 255
    w, h, c = img.shape
    start = end = time.time()

    losses = []
    
    
    for itr in range(cfg.iteration):
        model.train()
        slice_avg = torch.zeros([cfg.batch, w, h, c]).to(device)
  
        p_mtx = np.random.uniform(size=[img.shape[0],img.shape[1],img.shape[2]])
        mask = (p_mtx>cfg.mask_rate).astype(np.double)*0.7
        mask = np.expand_dims(mask, 0)
        mask = torch.tensor(mask).to(device, dtype=torch.float32).repeat(cfg.batch, 1, 1, 1).cuda()        
        
        img_input = np.array(img_expand, dtype=np.uint8) 
        y = img_expand 
   
        # if transforms:
        #     for i in range(cfg.batch-1):
        #         img_input[i+1] = transforms['train'](image=img_input[i+1])['image']  
                
        
        img_input = torch.Tensor(img_input).to(device, dtype=torch.float32) / 255.0
        target = torch.Tensor(y).to(device, dtype=torch.float32)  / 255.0
        
        p1 = np.random.uniform(size=1) 
        p2 = np.random.uniform(size=1)
        

        img_input_tensor = image_fliper(img_input, p1, p2)
        target = image_fliper(target, p1, p2)
                              
        #
        Gaussian_Noise = torch.tensor(np.random.normal(scale=50 / 255., size=img_input_tensor.shape).astype(np.float32)).to(device, dtype=torch.float32)
        # print("Gaussian_Noise:", torch.max(Gaussian_Noise))
        
        n = mask * Gaussian_Noise
        
        clip = 0
        img_input_tensor = img_input_tensor + n
        
        if clip:
            img_input_tensor = torch.clip(img_input_tensor + n, min=0, max=1)
        img_input_tensor = img_input_tensor.cuda()
        
        # print("img_input_tensor:", torch.max(img_input_tensor))  #ß ([1, 512, 512, 3]

        output = model(img_input_tensor.permute(0, 3, 1, 2), mask.permute(0, 3, 1, 2)).permute(0, 2, 3, 1)
        

        if itr == 0:
            slice_avg = image_fliper(output, p1, p2)
        else:
            slice_avg = slice_avg*0.99 + image_fliper(output, p1, p2)*0.01
            
        # print("Loss:,", torch.max(output), torch.max(target), )
        loss = torch.sum((output-target)*(output-target)*(1-mask))/torch.sum(1-mask)    # unmasked area
        losses.append(loss.cpu().detach())
  
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        
   
        end = time.time()
        
        if ((itr+1)%20 == 0):
            print("Elapsed %s  iteration %d, loss = %.4f" % (timeSince(start, float(itr+1)/cfg.iteration), itr+1, loss.item()*100))
            logging.info("Elapsed %s  iteration %d, loss = %.4f" % (timeSince(start, float(itr+1)/cfg.iteration), itr+1, loss.item()*100))
            
        # if (itr+1)%1000 == 0:           
        # if ((itr+1)%2500 == 0 and (itr+1)<100000 == 0) or (((itr+1)<5000 and (itr+1)%200 == 0)) or (((itr+1)<100 and (itr+1)%25 == 0)):
        if ((itr+1)%250 == 0):
            model.eval()
            sum_preds = np.zeros((img.shape[0],img.shape[1],img.shape[2]))
            for j in range(cfg.NPred):
                p_mtx = np.random.uniform(size=img.shape)
                mask = (p_mtx>p).astype(np.double) * 0.7
                
                img_input = np.expand_dims(img , 0)  # * mask* mask* maskStill Mask ... 
                img_input_tensor = torch.Tensor(img_input).to(device, dtype=torch.float32) / 255.0 #image_loader(img_input, device, 0.1, 0.1, 0)
                
                mask = np.expand_dims(mask, 0)
                mask = torch.tensor(mask).to(device, dtype=torch.float32)
                # print("img_input_tensor:", img_input_tensor.permute(0, 3, 1, 2).shape,mask.permute(0, 3, 1, 2).shape)   # 1, 512, 512, 3]
                
                output_test = model(img_input_tensor.permute(0, 3, 1, 2), mask.permute(0, 3, 1, 2)).permute(0, 2, 3, 1)
                sum_preds[:,:,:] += output_test.detach().cpu().numpy()[0]
    
            avg_preds = np.squeeze(np.uint8(np.clip((sum_preds-np.min(sum_preds)) / (np.max(sum_preds)-np.min(sum_preds)), 0, 1) * 255))
  
            # print("saved:", avg_preds)
            write_img = Image.fromarray(avg_preds)
            write_img.save(os.path.join(cfg.imgdir, "My-"+str(itr+1)+".png"))
 



if __name__ == "__main__":
    
    img = '/root/autodl-tmp/data/CBSD68/noisy50/0018.png'
    #"./testsets/PolyU/45.JPG" # (321, 481)
    # autodl-tmp/data/CBSD68/noisy50/0046.png  # 18
    
    print(f"Start denoise {img}")
    # img = './testsets/BSD68/11.png'
    # img = './testsets/Set9/5.png'  # (512, 512, 3)
    sigma = -1
    # train_self2self(img, 3, cfg.dropout_rate, sigma=sigma, is_realnoisy=True)
    train_self2self(img, 3, cfg.dropout_rate, sigma=sigma, is_realnoisy=True)

   
   

