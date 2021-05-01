import numpy as np
import itertools
import cv2


def check_background(img, crop_size) -> bool:
    s_th = 40  # saturation blancking threshold
    p_th = 1000 * (crop_size // 256) ** 2  # threshold for the minimum number of pixels
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    _, ss, _ = cv2.split(hsv)
    background = True if (ss > s_th).sum() <= p_th or img.sum() <= p_th else False
    return background


def make_grid_wo_overlap(shape, window=256):
    x, y = shape
    x1 = np.arange(0, x, window)
    x1[-1] = x - window
    y1 = np.arange(0, y, window)
    y1[-1] = y - window
    nx, ny = len(x1), len(y1)
    slices = np.zeros((nx, ny, 2), dtype=np.int64) 
    for i in range(nx):
        for j in range(ny):
            slices[i, j] = x1[i], y1[j]   
    return slices.reshape(nx * ny, 2)


def make_grid(shape, window, min_overlap):
    x,y = shape
    step = window - min_overlap
    
    x1 = [_x for _x in range(0,x-window,step)]
    if x1[-1] != x - window - 1:
        x1 += [x - window - 1]
        
    y1 = [_y for _y in range(0,y-window,step)]
    if y1[-1] != y - window - 1:
        y1 += [y - window - 1]
        
    return [(_x, _y) for _x,_y in itertools.product(x1, y1)]