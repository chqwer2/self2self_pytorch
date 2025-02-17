import numpy as np
import cv2
import scipy.io as sio
import random
import time
import math

def add_gaussian_noise(img, model_path, sigma):
    index = model_path.rfind("/")
    if sigma > 0:
        noise = np.random.normal(scale=sigma / 255., size=img.shape).astype(np.float32)
        sio.savemat(model_path[0:index] + '/noise.mat', {'noise': noise})
        noisy_img = (img + noise).astype(np.float32)
    else:
        noisy_img = img.astype(np.float32)
    cv2.imwrite(model_path[0:index] + '/noisy.png',
                np.squeeze(np.int32(np.clip(noisy_img, 0, 1))))
    return noisy_img


def load_np_image(path, is_scale=True):
    img = cv2.imread(path, -1)
    if img.ndim == 2:
        img = np.expand_dims(img, axis=2)
    img = np.expand_dims(img, axis=0)
    if is_scale:
        img = np.array(img).astype(np.float32) / 255.
    return img


def mask_pixel(img, model_path, rate):
    index = model_path.rfind("/")
    masked_img = img.copy()
    mask = np.ones_like(masked_img)
    perm_idx = [i for i in range(np.shape(img)[1] * np.shape(img)[2])]
    random.shuffle(perm_idx)
    for i in range(np.int32(np.shape(img)[1] * np.shape(img)[2] * rate)):
        x, y = np.divmod(perm_idx[i], np.shape(img)[2])
        masked_img[:, x, y, :] = 0
        mask[:, x, y, :] = 0
    cv2.imwrite(model_path[0:index] + '/masked_img.png', np.squeeze(np.uint8(np.clip(masked_img, 0, 1) * 255.)))
    cv2.imwrite(model_path[0:index] + '/mask.png', np.squeeze(np.uint8(np.clip(mask, 0, 1) * 255.)))
    return masked_img, mask


def asMinutes(s):
    m = math.floor(s / 60)
    s -= m * 60
    return '%dm %ds' % (m, s)


def timeSince(since, percent):
    now = time.time()
    s = now - since
    es = s / (percent)
    rs = es - s
    return '%s (remain %s)' % (asMinutes(s), asMinutes(rs))
