import tifffile

tif = tifffile.TiffFile('Images_in/1 026.tif')
sequence = tif.asarray()

print(sequence.shape)
