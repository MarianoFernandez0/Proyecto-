#!/home/leo/anaconda3/bin/python
import cv2
import numpy as np


def segmentation(image):
    '''
    Given a input image, its segmented by applying
    OTSU, then it is steched and finally, dilated.

    Input:
        image: image in rgb or bw
    Output:
        mask: image segmented
    '''
    image_bw = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    _, image_tresh = cv2.threshold(image_bw, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    image_streched = cv2.morphologyEx(image_tresh, cv2.MORPH_OPEN, np.ones((3, 3), np.uint8))
    mask = cv2.dilate(image_streched, np.ones((3, 3), np.uint8), iterations=1)

    return mask
