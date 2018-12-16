import os
import shutil
import time
import math
from skimage import transform, io
from classification import selectivesearch


def get_component_position(img, is_using_thumb=True):
    thumb_img = img
    scale_index = 1
    pixels = img.shape[0] * img.shape[1]
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
        thumb_img, scale=4000, sigma=0.4, min_size=60)

    candidates = set()
    for r in regions:
        # excluding same rectangle (with different segments)
        if r['rect'] in candidates:
            continue
        # excluding regions smaller than 2000 pixels
        if r['size'] < 125:
            continue
        if r['size'] > pixels * 0.1:
            continue
        # distorted rects
        x, y, w, h = r['rect']
        if h == 0 or w == 0:
            continue
        if w / h > 4 or h / w > 4:
            continue
        candidates.add(r['rect'])
    if is_using_thumb:
        candidates = {tuple(int(loc * scale_index) for loc in candidate) for candidate in candidates}
    # print(candidates)
    return candidates


def filter_block2(blocks):
    threshold = 0.6
    result = list(blocks)
    has_overlap = True
    while has_overlap is True:
        has_overlap = False
        for eachBlock in result:
            if has_overlap is True:
                break
            for another_block in result:
                if eachBlock == another_block:
                    continue
                # count overlap rate of two blocks
                if calculate_overlap_rate(eachBlock, another_block) > threshold:
                    result.append(get_overlap_block(eachBlock, another_block))
                    result.remove(another_block)
                    result.remove(eachBlock)
                    has_overlap = True
                    break
    return result


def calculate_overlap_rate(block_a, block_b):
    a1x, a1y, w_, h_ = block_a
    a2x = a1x + w_
    a2y = a1y + h_
    sa = (a2x - a1x) * (a2y - a1y)

    b1x, b1y, w_, h_ = block_b
    b2x = b1x + w_
    b2y = b1y + h_
    sb = (b2x - b1x) * (b2y - b1y)

    si = max(0, min(a2x, b2x) - max(a1x, b1x)) * max(0, min(a2y, b2y) - max(a1y, b1y))
    su = sa + sb - si
    overlap_ratio = si / su
    return overlap_ratio


def get_overlap_block(block_a, block_b):
    a1x, a1y, w_, h_ = block_a
    a2x = a1x + w_
    a2y = a1y + h_

    b1x, b1y, w_, h_ = block_b
    b2x = b1x + w_
    b2y = b1y + h_

    c1x = min(a1x, b1x)
    c1y = min(a1y, b1y)
    c2x = max(a2x, b2x)
    c2y = max(a2y, b2y)

    return [c1x, c1y, c2x - c1x, c2y - c1y]


def make_dir(path):
    if not os.path.exists(path):
        os.makedirs(path)
    else:
        shutil.rmtree(path)
        os.makedirs(path)


def read_img_random(path, total_count):
    imgs = []
    labels = []
    count = 0
    file_path_list = [os.path.join(path, file_name) for file_name in os.listdir(path)
                      if os.path.isfile(os.path.join(path, file_name))]
    file_names = [file_name.replace('.', '_') for file_name in os.listdir(path)
                  if os.path.isfile(os.path.join(path, file_name))]
    while count < total_count and count < len(file_path_list):
        im = file_path_list[count]
        img = io.imread(im)
        if img.shape[2] == 4:
            img = img[:, :, :3]
        imgs.append(img)
        labels.append(file_names[count])
        count += 1
        print("\rreading {0}/{1}".format(count, min(total_count, len(file_path_list))), end='')
    print('\r', end='')
    return imgs, labels


root_path = r"D:\Projects\IoT_recognition\20181028\raw/"
result_path = root_path + 'result' + '/'
frame_per_second = 60
start_time = time.time()
make_dir(result_path)
images, names = read_img_random(root_path, 1000)
for i in range(len(images)):
    image = images[i]
    name = names[i]

    raw_blocks = get_component_position(image)
    filtered_blocks = filter_block2(raw_blocks)
    for block in filtered_blocks:
        _x = block[0]
        _y = block[1]
        _w = block[2]
        _h = block[3]
        sub_image = image[_y:_y + _h, _x:_x + _w]
        save_path = result_path + '/' + name + '_' + str(filtered_blocks.index(block)) + '.png'
        io.imsave(save_path, sub_image)

    pre_end_time = time.time()
    pre_processing_time = pre_end_time - start_time
    print("{} processed in {} seconds".format(name, pre_processing_time))
    start_time = time.time()
