import imageio
import cv2
from deteccion import detect_particles

# Cargar el video
video = imageio.mimread('1472 semen-00.avi')
# Pasar a grises
grises = cv2.cvtColor(video[0], cv2.COLOR_BGR2GRAY)

particles = detect_particles(grises)


