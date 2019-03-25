# IXI MRI data has varying slices of T1 and T2
# This function reslices it to sidexsidexside images
import cv2
import numpy as np

def resize_3d(img, side=256):

    x, y, z = img.shape
    #print(x, y, z)
    # 6 cases
    '''
    x > side
    x < side
    y > side
    y < side
    z > side
    z < side
    '''
    if x < side:
        diff = side - x
        pad = np.zeros((diff//2, y, z))
        if diff % 2 == 0:
            img = np.concatenate((pad, img, pad), axis=0)
        else:
            img = np.concatenate((pad, img, pad, np.zeros((1, y, z))), axis=0)
    elif x > side:
        diff = x - side
        img = img[diff//2:diff//2 + side,:,:]
    x=side
    #print('x ',img.shape)

    if y < side:
        diff = side - y
        pad = np.zeros((x, diff//2, z))
        if diff % 2 == 0:
            img = np.concatenate((pad, img, pad), axis=1)
        else:
            img = np.concatenate((pad, img, pad, np.zeros((x, 1, z))), axis=1)
    elif y > side:
        diff = y - side
        img = img[:,diff//2:diff//2 + side,:]
    y=side
    #print('y ',img.shape)

    if z < side:
        diff = side - z
        pad = np.zeros((x, y, diff//2))
        if diff % 2 == 0:
            img = np.concatenate((pad, img, pad), axis=2)
        else:
            img = np.concatenate((pad, img, pad, np.zeros((x, y, 1))), axis=2)
    elif z > side:
        diff = z - side
        img = img[:,:,diff//2:diff//2 + side]
    z=side
    #print('z ',img.shape)

    return img