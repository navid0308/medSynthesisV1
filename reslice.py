# IXI MRI data has varying slices of T1 and T2
# This function reslices it to 128 slices of 256x256 images
import cv2
import numpy as np

def resize_3d(img):
    resliced_img = np.zeros((128,256,256))
    for i in range(len(img[:,:,0])):
        image = img[:,:,i]
        image = cv2.resize(image, (256,128), interpolation=cv2.INTER_CUBIC)
        resliced_img[:,:,i] = image
    return resliced_img