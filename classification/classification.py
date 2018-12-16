import os
import shutil
import numpy as np
from skimage import io, transform
from keras.models import load_model
import tensorflow as tf
from tensorflow.python.keras.backend import set_session


def classify_images(img_root_path, cat_names, cat_list, img_name_list):
    for i in range(len(cat_names)):
        folder_path = img_root_path + cat_names[i] + "/"
        make_dir(folder_path)
    for i in range(len(cat_list)):
        cat = cat_names[cat_list[i]]
        img_name = img_name_list[i]
        folder_path = img_root_path + cat + "/"
        img_path = img_root_path + img_name
        shutil.copy(img_path, folder_path)


def read_img_random(path, total_count):
    file_path_list = [os.path.join(path, file_name) for file_name in os.listdir(path)
                      if os.path.isfile(os.path.join(path, file_name))]
    imgs = []
    labels = []
    count = 0
    while count < total_count and count < len(file_path_list):
        im = file_path_list[count]
        file_name = im.split('/')[-1]
        count += 1
        img = io.imread(im)

        if len(img.shape) > 2 and img.shape[2] == 4:
            img = img[:, :, :3]
        img = transform.resize(img, (w, h))
        imgs.append(img)
        labels.append(file_name)
        if count % 1 == 0:
            print("\rreading {0}/{1}".format(count, min(total_count, len(file_path_list))), end='')
    print('\r', end='')
    return np.asarray(imgs, np.float32), np.asarray(labels, np.str_)


def make_dir(path):
    if not os.path.exists(path):
        os.makedirs(path)
    else:
        shutil.rmtree(path)
        os.makedirs(path)


def get_max_and_confidence(pred_results):
    cat_result = []
    for _result in pred_results:
        result_as_list = [v for v in _result]
        max_confidence = max(result_as_list)
        index = result_as_list.index(max_confidence)
        cat_result.append((index, max_confidence))
    return cat_result


w = 150
h = 150
c = 3
train_path = r'D:\Projects\IoT_recognition\20181028\samples_500\train/'
sample_images_path = r'D:\Projects\IoT_recognition\20181028\samples_5000/'
sub_images, file_names = read_img_random(sample_images_path, 5000)

config = tf.ConfigProto()
config.gpu_options.allow_growth = True
set_session(tf.Session(config=config))

input_shape = (w, h, c)
learning_rate = 0.0001
regularization_rate = 0.0001
category_count = 13 + 1

model = load_model(train_path + '/model.h5')
result = model.predict(np.asarray(sub_images, np.float32))
cat = get_max_and_confidence(result)
cat_name_list = [
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

index = 0
result_cat_list = []
for i in range(len(result)):
    result_cat_list.append(cat[i][0])

classify_images(sample_images_path, cat_name_list, result_cat_list, file_names)
