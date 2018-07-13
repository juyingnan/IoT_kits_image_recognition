from skimage import io
import skimage.segmentation
import skimage.util
from skimage import transform
import math
import matplotlib.pyplot as plt


def save_img(img, file_path):
    # io.imsave(file_path, img)
    fig, ax = plt.subplots(ncols=1, nrows=1, figsize=(8, 8))
    ax.imshow(img)
    plt.savefig(file_path)


def segment_image(img, scale, sigma, min_size, file_path, is_using_thumb=True):
    thumb_img = img
    if is_using_thumb:
        pixels = 300 * 400
        original_w = img.shape[0]
        original_h = img.shape[1]
        original_pixels = original_w * original_h
        scale_index = math.sqrt(original_pixels / pixels)
        new_w = int(original_w / scale_index)
        new_h = int(original_h / scale_index)
        thumb_img = transform.resize(img, (new_w, new_h))
    im_mask = skimage.segmentation.felzenszwalb(
        skimage.util.img_as_float(thumb_img), scale=scale, sigma=sigma, min_size=min_size)
    # show_image(im_mask)
    save_img(im_mask, file_path)


def read_image(file_path):
    img = io.imread(file_path)
    return img


def show_image(img):
    fig, ax = plt.subplots(ncols=1, nrows=1, figsize=(8, 8))
    ax.imshow(img)
    plt.show()


img_path = r'C:\Users\bunny\Desktop\graph_test\100\IMG_1281.JPG'
image = read_image(img_path)
path_prefix = img_path.split('.')[0]
# path_postfix = img_path.split('.')[1]
for sc in range(0, 10001, 2000):
    for si in range(0, 101, 20):
        for mi in range(0, 101, 20):
            path_mid = '_' + str(sc) + '_' + str(si) + '_' + str(mi)
            segment_image(image, scale=sc, sigma=(si / 100.0), min_size=mi,
                          file_path=path_prefix + path_mid)  # + path_postfix)
