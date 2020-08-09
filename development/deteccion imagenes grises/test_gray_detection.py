import imageio
from gray_detection import gray_detection
import matplotlib.pyplot as plt

video = imageio.mimread('1472 semen-00.avi')
img_in = video[0]
bin_img = gray_detection(img_in)
plt.imsave('deteccion_img_grises.png',bin_img,cmap='gray')