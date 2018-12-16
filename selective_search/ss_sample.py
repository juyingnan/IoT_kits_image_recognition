# -*- coding: utf-8 -*-
from __future__ import (
    division,
    print_function,
)

from skimage import io
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import selectivesearch


def main():

    # loading astronaut image
    # img = skimage.data.astronaut()
    im = r'C:\Users\bunny\Desktop\IMG_1282.JPG'
    img = io.imread(im)

    # perform selective search
    img_lbl, regions = selectivesearch.selective_search(
        img, scale=5000, sigma=1.0, min_size=10)

    candidates = set()
    for r in regions:
        # excluding same rectangle (with different segments)
        if r['rect'] in candidates:
            continue
        # excluding regions smaller than 2000 pixels
        if r['size'] < 2000:
            continue
        # distorted rects
        x, y, w, h = r['rect']
        if w / h > 3 or h / w > 3:
            continue
        candidates.add(r['rect'])

    # draw rectangles on the original image
    fig, ax = plt.subplots(ncols=1, nrows=1, figsize=(6, 6))
    ax.imshow(img)
    for x, y, w, h in candidates:
        print(x, y, w, h)
        rect = mpatches.Rectangle(
            (x, y), w, h, fill=False, edgecolor='red', linewidth=1)
        ax.add_patch(rect)

    plt.show()

if __name__ == "__main__":
    main()