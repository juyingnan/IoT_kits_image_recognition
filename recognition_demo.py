# from tensorflow import keras
# from tensorflow.python.keras import regularizers
# from tensorflow.python.keras.layers import Dense, Conv2D, MaxPooling2D, Flatten
# from tensorflow.python.keras.models import Sequential
from keras.models import load_model
import tensorflow as tf
from tensorflow.python.keras.backend import set_session
import numpy as np
from skimage import io
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import selectivesearch
from skimage import transform
import sys
import time
import math


def read_image(image_path):
    img = io.imread(image_path)
    return img


def get_component_position(img, is_using_thumb=True):
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
        thumb_img, scale=5000, sigma=0.9, min_size=10)

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
        if w / h > 4 or h / w > 4:
            continue
        candidates.add(r['rect'])
    if is_using_thumb:
        candidates = {tuple(int(loc * scale_index) for loc in candidate) for candidate in candidates}
    # print(candidates)
    return candidates


def filter_block(blocks):
    result = []
    for eachBlock in blocks:
        is_independent = True
        for another_block in blocks:
            if eachBlock == another_block:
                continue
            ax, ay = eachBlock[0], eachBlock[1]
            bx, by = another_block[0], another_block[1]
            ax_, ay_ = ax + eachBlock[2], ay + eachBlock[3]
            bx_, by_ = bx + another_block[2], by + another_block[3]
            if ax >= bx and ay >= by and ax_ <= bx_ and ay_ <= by_:
                is_independent = False
                break
        if is_independent:
            result.append(eachBlock)
    return result


def draw_image(img, block_candidates):
    # draw rectangles on the original image
    fig, ax = plt.subplots(ncols=1, nrows=1, figsize=(8, 8))
    ax.imshow(img)
    for x_, y_, w_, h_ in block_candidates:
        rect = mpatches.Rectangle(
            (x_, y_), w_, h_, fill=False, edgecolor='red', linewidth=1)
        ax.add_patch(rect)
    plt.show()


def draw_name_on_image(img, block_candidates):
    # draw rectangles on the original image
    fig, ax = plt.subplots(ncols=1, nrows=1, figsize=(20, 20))
    plt.imshow(img)
    for x_, y_, w_, h_ in block_candidates:
        rect = mpatches.Rectangle(
            (x_, y_), w_, h_, fill=False, edgecolor='red', linewidth=1)
        ax.add_patch(rect)
    for wachBlock in block_candidates:
        x = wachBlock[0]
        y = wachBlock[1]
        index = block_candidates.index(wachBlock)
        plt.text(x, y, cat_list[cat[index]], color='green', fontsize=16)
    # plt.show()
    fig.savefig(result_path)


img_path = r'C:/Users/bunny/Desktop/mega2560_raw/IMG_1287.JPG'
if len(sys.argv) < 2:
    print("parameter: file path")
    print("now use default path")
else:
    img_path = sys.argv[1]
result_path = img_path.split('.')[0] + '_result.png'

default_log_file = r'C:/Users/bunny/Desktop/log.txt'
fp = open(default_log_file, 'a')
fp.write(img_path + '\n')

image = read_image(img_path)
start_time = time.time()
raw_blocks = get_component_position(image)
# print(len(raw_blocks))
fp.write(str(len(raw_blocks)) + '\n')

h_aver = sum([group[3] for group in raw_blocks]) / len(raw_blocks)
# print(h_aver)
w_aver = sum([group[2] for group in raw_blocks]) / len(raw_blocks)
# print(w_aver)

size_index = 2.5
size_filtered_blocks = [group for group in raw_blocks if
                        (group[2] <= w_aver * size_index and group[3] <= h_aver * size_index)]

filtered_blocks = filter_block(size_filtered_blocks)
# print(len(filtered_blocks))
fp.write(str(len(filtered_blocks)) + '\n')
# draw_image(image, filtered_blocks)

w = 200
h = 200
c = 3

sub_images = []
for block in filtered_blocks:
    _x = block[0]
    _y = block[1]
    _w = block[2]
    _h = block[3]
    sub_image = image[_y:_y + _h, _x:_x + _w]
    if sub_image.shape[2] == 4:
        sub_image = sub_image[:, :, :3]
    sub_image = transform.resize(sub_image, (w, h))
    sub_images.append(sub_image)

pre_end_time = time.time()
pre_processing_time = pre_end_time - start_time
# print(pre_processing_time)
fp.write(str(pre_processing_time) + '\n')

config = tf.ConfigProto()
config.gpu_options.allow_growth = True
set_session(tf.Session(config=config))

input_shape = (w, h, c)
learning_rate = 0.0001
regularization_rate = 0.0001
category_count = 13 + 1
train_path = r'C:\Users\bunny\Desktop\Iot\mega_2560_cat\TRAIN/'
model = load_model(train_path + '/model.h5')

cat = model.predict_classes(np.asarray(sub_images, np.float32))
cat_list = [
    "blank",
    "gy-521_module",
    "ir_receiver_module",
    "max7219_module",
    "mega2560_controller_board",
    "pir_motion_sensor_HC-SR501",
    "power_supply_module",
    "relay_5v",
    "rotary_encoder_module",
    "sound_sensor_module",
    "stepper_motor_driver_board_uln2003",
    "temperature_and_humidity_module_DHT11",
    "ultrasonic_sensor",
    "water_level_detection_sensor_module",
]
draw_name_on_image(image, filtered_blocks)
final_end_time = time.time()
machine_learning_time = final_end_time - pre_end_time
# print(machine_learning_time)
fp.write(str(machine_learning_time) + '\n')
fp.close()
