import tifffile
from evaluation import evaluation

tif = tifffile.TiffFile('detection/Images_in/11.tif')

df = evaluation(tif)

print(df)