import numpy as np
import cv2
import os


annotator_ckpts_path = os.path.join(os.path.dirname(__file__), 'ckpts')


def HWC3(x):
    assert x.dtype == np.uint8
    if x.ndim == 2:
        x = x[:, :, None]
    assert x.ndim == 3
    H, W, C = x.shape
    assert C == 1 or C == 3 or C == 4
    if C == 3:
        return x
    if C == 1:
        return np.concatenate([x, x, x], axis=2)
    if C == 4:
        color = x[:, :, 0:3].astype(np.float32)
        alpha = x[:, :, 3:4].astype(np.float32) / 255.0
        y = color * alpha + 255.0 * (1.0 - alpha)
        y = y.clip(0, 255).astype(np.uint8)
        return y


def resize_image(input_image, resolution):
    H, W, C = input_image.shape
    H = float(H)
    W = float(W)

    if resolution <= 0:
        raise ValueError("Resolution must be a positive value.")
    
    k = float(resolution) / min(H, W)
    H *= k
    W *= k
    H = int(np.round(H / 64.0)) * 64
    W = int(np.round(W / 64.0)) * 64
    if H <= 0 or W <= 0:
        raise ValueError(f"Computed dimensions are invalid: Width={W}, Height={H}")
    img = cv2.resize(input_image, (W, H), interpolation=cv2.INTER_LANCZOS4 if k > 1 else cv2.INTER_AREA)
    return img

def my_resize_image(input_image, resolution):
    H, W, C = input_image.shape
    H = float(H)
    W = float(W)
    
    k= min(H,W)/64

    H = int(np.round(H / 64.0)) * 64 if H>64 else 64
    W = int(np.round(W / 64.0)) * 64 if W>64 else 64

    if H < 0 or W < 0:
        raise ValueError(f"Computed dimensions are invalid: Width={W}, Height={H}")
    
    img = cv2.resize(input_image, (W, H), interpolation=cv2.INTER_LANCZOS4 if k>=1 else cv2.INTER_AREA)
    return img

def modified_resize_image(input_image, resolution):
    H, W, C = input_image.shape
    H = float(H)
    W = float(W)

    if resolution <= 0:
        raise ValueError("Resolution must be a positive value.")
    
    k = float(resolution) / min(H, W)
    if k>1:
        H = int(H)
        W = int(W)
    else:
        H *= k
        W *= k
        H = int(np.round(H / 64.0)) * 64
        W = int(np.round(W / 64.0)) * 64
    if H <= 0 or W <= 0:
        raise ValueError(f"Computed dimensions are invalid: Width={W}, Height={H}")
    img = cv2.resize(input_image, (W, H), interpolation=cv2.INTER_LANCZOS4 if k > 1 else cv2.INTER_AREA)
    return img
