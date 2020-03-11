from skimage.io import imread, imsave
import matplotlib.pyplot as plt
from matplotlib.patches import Circle
from segmentation.segmentation import segmentation
from detection.deteccion import detect_particles, size_filter
import tifffile


sample_files = ['9.tif']  # , '10.tif', '11.tif', '1 026.tif']
for sample in sample_files:
    tif = tifffile.TiffFile('Images_in/' + sample)
    x_res = tif.pages[0].tags['XResolution']
    y_res = tif.pages[0].tags['YResolution']
    pixel_size = [x_res.value[1] / x_res.value[0], y_res.value[1] / y_res.value[0]]

    sequence = tif.asarray()
    samples = [sequence[0, :, :], sequence[sequence.shape[0] // 2, :, :], sequence[sequence.shape[0] - 1, :, :]]

    for i in range(3):
        img = samples[i]
        imsave('Images_out/sample ' + str(i) + '_(' + sample + ').png', img)

        seg_img = segmentation(img)
        particles = detect_particles(seg_img)

        print('Total de partículas detectadas antes de filtrar ' + str(i) + ':', len(particles))

        fig, ax = plt.subplots(1)
        ax.imshow(seg_img, cmap='gray')
        for p in range(particles.shape[0]):
            patch = Circle((particles.at[p, 'y'], particles.at[p, 'x']), radius=1, color='red')
            ax.add_patch(patch)
        plt.axis('off')
        plt.savefig('Images_out/coords ' + str(i) + '_(' + sample + ').png', bbox_inches='tight')

        particles = size_filter(particles, pixel_size)

        print('Total de partículas detectadas luego de filtrar ' + str(i) + ':', len(particles))

        fig, ax = plt.subplots(1)
        ax.imshow(seg_img, cmap='gray')
        for p in particles.index:
            patch = Circle((particles.at[p, 'y'], particles.at[p, 'x']), radius=1, color='red')
            ax.add_patch(patch)
        plt.axis('off')
        plt.savefig('Images_out/coords_filtered ' + str(i) + '_(' + sample + ').png', bbox_inches='tight')
