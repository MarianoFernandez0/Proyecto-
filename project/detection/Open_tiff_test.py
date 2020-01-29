import tifffile

#help(tifffile)
tif = tifffile.TiffFile('Images_in/11.tif')
print(type(tif))
data = tif.asarray()
print(data.shape)
print(tif.imagej_metadata)
print(tif.pages[0].tags['XResolution'])
print(tif.pages[0].tags['YResolution'])
print(tif.pages[0].tags['ResolutionUnit'])
print(tif.pages[0].tags['ImageDescription'])
print(tif.pages[0].tags['SamplesPerPixel'])

xr = tif.pages[0].tags['XResolution']
print(dir(xr))
print(type(xr.value))
print(xr.value[1]/xr.value[0])
yr = tif.pages[0].tags['YResolution']
print(yr.value[1]/yr.value[0])

pixel_size = xr.value[1]/xr.value[0]



print('Here')

from skimage.external.tifffile import TiffFile 
with TiffFile('Images_in/1 026.tif', fastij=True, is_ome=True) as tif:
	data = tif.asarray()
	#print(data.shape)
	print(tif.info())
    #print((tif.series[0].pages[0].info()))
    #print(str(tif.series))

