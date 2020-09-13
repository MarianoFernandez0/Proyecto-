def morf_operations(img_in, kernel):
    dilation = cv2.dilate(img_in, kernel, iterations=1)
    closing = cv2.morphologyEx(dilation, cv2.MORPH_CLOSE, kernel)
    erosion = cv2.erode(closing, kernel, iterations=1)
    return erosion


def sobel_filtering(im):
    dx = ndimage.sobel(im, 0)  # horizontal derivative
    dy = ndimage.sobel(im, 1)  # vertical derivative
    mag = np.hypot(dx, dy)  # magnitude
    mag *= 255.0 / np.max(mag)  # normalize (Q&D)
    return mag