import imageio
from gray_detection import gray_detection
from deteccion_mariano import detect_particles

# Cargar el video
video = imageio.mimread('1472 semen-00.avi')

# Hacer la deteccion en grises
img_in = video[0]
img_out = gray_detection(img_in)
particles = detect_particles(img_out)
