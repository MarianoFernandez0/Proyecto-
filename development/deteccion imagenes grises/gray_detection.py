import cv2
from scipy.signal.signaltools import wiener
from scipy import ndimage
import numpy as np

def morf_operations(img_in,kernel):
    dilation = cv2.dilate(img_in,kernel,iterations = 1)
    closing = cv2.morphologyEx(dilation, cv2.MORPH_CLOSE, kernel)
    erosion = cv2.erode(closing,kernel,iterations = 1)
    return erosion

def sobel_filtering(im):
    dx = ndimage.sobel(im, 0)  # horizontal derivative
    dy = ndimage.sobel(im, 1)  # vertical derivative
    mag = np.hypot(dx, dy)  # magnitude
    mag *= 255.0 / np.max(mag)  # normalize (Q&D)
    return mag

def gray_detection(img_in):
    greyscale = cv2.cvtColor(img_in, cv2.COLOR_BGR2GRAY) # Convertir a grises
    wfiltered_img = wiener(greyscale, (3,3))  # Filtro de wiener
    sobel_img = sobel_filtering(wfiltered_img) # Sobel
    sobel_img = np.array(sobel_img, dtype=np.uint8)
    tresh,otsu = cv2.threshold(sobel_img,0,255,cv2.THRESH_BINARY+cv2.THRESH_OTSU) # Otsu
    kernel = np.ones((3,3),np.uint8)
    apply_morf_op = morf_operations(otsu,kernel) # Apply morfological operation
    return apply_morf_op