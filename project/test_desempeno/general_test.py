import tifffile

tif = tifffile.TiffFile('Images_in/11.tif')
print(tif.pages)