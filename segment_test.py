from skimage import io
import skimage.segmentation
import skimage.util
from skimage import transform
import math
import matplotlib.pyplot as plt
# from matplotlib import cm
import matplotlib.patches as mpatches
import selectivesearch


def get_component_position(img, scale, sigma, min_size, is_using_thumb=True):
    thumb_img = img
    scale_index = 1
    if is_using_thumb:
        pixels = 300 * 400
        original_w = img.shape[0]
        original_h = img.shape[1]
        original_pixels = original_w * original_h
        scale_index = math.sqrt(original_pixels / pixels)
        new_w = int(original_w / scale_index)
        new_h = int(original_h / scale_index)
        thumb_img = transform.resize(img, (new_w, new_h))
    # perform selective search
    img_lbl, regions = selectivesearch.selective_search(
        thumb_img, scale=scale, sigma=sigma, min_size=min_size)

    candidates = set()
    for r in regions:
        # excluding same rectangle (with different segments)
        if r['rect'] in candidates:
            continue
        # excluding regions smaller than 2000 pixels
        if r['size'] < 200:
            continue
        # distorted rects
        x, y, w, h = r['rect']
        if h == 0 or w == 0:
            continue
        # if w / h > 4 or h / w > 4:
        #     continue
        candidates.add(r['rect'])
    if is_using_thumb:
        candidates = {tuple(int(loc * scale_index) for loc in candidate) for candidate in candidates}
    # print(candidates)
    return candidates


def save_img(img_mask, img_blocks, file_path, sub_title='', ground_truth=[]):
    # io.imsave(file_path, img)
    # fig, ax = plt.subplots(ncols=1, nrows=1, figsize=(8, 8))
    fig, axes = plt.subplots(1, 2, figsize=(16, 8))  # , sharex=True, sharey=True)
    ax = axes.ravel()
    ax[0].imshow(img_mask)  # , cmap=cm.gray)
    ax[0].set_title('img_mask' + sub_title)
    ax[1].imshow(image)  # , cmap=cm.gray)
    ax[1].set_title('img_block')
    for x_, y_, w_, h_ in img_blocks:
        rect = mpatches.Rectangle((x_, y_), w_, h_, fill=False, edgecolor='red', linewidth=1)
        ax[1].add_patch(rect)

    for x_, y_, w_, h_ in ground_truth:
        rect = mpatches.Rectangle((x_, y_), w_, h_, fill=False, edgecolor='green', linewidth=1)
        ax[1].add_patch(rect)

    for a in ax:
        a.set_axis_off()
    plt.tight_layout()
    plt.savefig(file_path)


def segment_image(img, scale, sigma, min_size, is_using_thumb=True):
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
    return im_mask


def read_image(file_path):
    img = io.imread(file_path)
    return img


def show_image(img):
    fig, ax = plt.subplots(ncols=1, nrows=1, figsize=(8, 8))
    ax.imshow(img)
    plt.show()


def process_image(img, scale, sigma, min_size, file_path, is_using_thumb=True, ground_truth=[]):
    img_mask = segment_image(img, scale=scale, sigma=sigma, min_size=min_size, is_using_thumb=is_using_thumb)
    raw_blocks = get_component_position(img, scale, sigma, min_size, is_using_thumb=is_using_thumb)
    print(len(raw_blocks))
    save_img(img_mask, raw_blocks, file_path, ' | scale={0}, sigma={1}, min_size={2}'.format(scale, sigma, min_size),
             ground_truth=ground_truth)
    print('file saved to {0}'.format(file_path))
    return raw_blocks


def calculate_abo(gc, bl):
    sum_overlap = 0
    for a1x, a1y, w_, h_ in gc:
        a2x = a1x + w_
        a2y = a1y + h_
        SA = (a2x - a1x) * (a2y - a1y)
        max_overlap = 0

        for b1x, b1y, w_, h_ in bl:
            b2x = b1x + w_
            b2y = b1y + h_
            SB = (b2x - b1x) * (b2y - b1y)
            SI = max(0, min(a2x, b2x) - max(a1x, b1x)) * max(0, min(a2y, b2y) - max(a1y, b1y))
            SU = SA + SB - SI
            overlap_ratio = SI / SU
            if overlap_ratio > max_overlap:
                max_overlap = overlap_ratio

        sum_overlap += max_overlap
    return sum_overlap / len(gc)


# ABO test
# ground truth for test image
ground_truth_test_image = [
    [187, 249, 472, 807],
    [948, 455, 220, 372],
    [1450, 317, 346, 623],
    [2032, 173, 500, 761],
    [188, 1314, 403, 449],
    [884, 1168, 244, 824],
    [1589, 1228, 245, 402],
    [2232, 1291, 260, 255],
    [213, 2369, 226, 551],
    [695, 2511, 224, 295],
    [1161, 2041, 687, 1342],
    [2166, 2066, 227, 336],
    [2088, 2591, 366, 469],
]

img_path = r'C:\Users\bunny\Desktop\graph_test\100\test.JPG'
image = read_image(img_path)
# from skimage import color
# image = color.rgb2hsv(image)
import time
start = time.time()
path_prefix = img_path.split('.')[0]
for sc in range(4000, 4200, 200):
    # for si in range(20, 100, 20):
    # for mi in range(20, 100, 20):
    si = 40
    mi = 60
    path_mid = '_' + str(sc) + '_' + str(si) + '_' + str(mi)
    blocks = process_image(image, scale=sc, sigma=(si / 100.0), min_size=mi, file_path=path_prefix + path_mid,
                           ground_truth=ground_truth_test_image)
    abo = calculate_abo(ground_truth_test_image, blocks)
    print(abo)
# process_image(image, scale=2000, sigma=0.2, min_size=20, file_path=path_prefix + '_2000_20_20', is_using_thumb=True)
end = time.time()
print(end-start)