import tifffile
from SEGMENTATION.segmentation import segmentation


tif = tifffile.TiffFile('deteccion/Images_in/11.tif')
sequence = tif.asarray()
print(sequence.shape)