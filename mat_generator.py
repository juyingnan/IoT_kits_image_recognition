from skimage import io, transform, color
import numpy as np
import os
import random
from scipy import io as sio


def read_img_random(path, total_count, resize=None, as_gray=False):
    cate = [path + folder for folder in os.listdir(path) if os.path.isdir(path + folder)]
    imgs = []
    labels = []
    ratios = []
    for idx, folder in enumerate(cate):
        print('reading the images:%s' % folder)
        count = 0
        file_path_list = [os.path.join(folder, file_name) for file_name in os.listdir(folder)
                          if os.path.isfile(os.path.join(folder, file_name))]
        while count < total_count and count < len(file_path_list):
            im = file_path_list[count]
            count += 1
            img = io.imread(im, as_gray=as_gray)
            ratios.append([get_width_height_ratio(img)])
            if resize is not None:
                img = transform.resize(img, resize)
            imgs.append(img)
            labels.append(idx)
            if count % 100 == 0:
                print("\rreading {0}/{1}".format(count, min(total_count, len(file_path_list))), end='')
        print('\r', end='')
    return np.asarray(imgs, np.float32), np.asarray(labels, int), ratios


def calculate_average_hue_saturation(img, h=True, s=True, v=True):
    img_hsv = color.rgb2hsv(img)
    img_h = img_hsv[:, :, 0]
    img_s = img_hsv[:, :, 1]
    img_v = img_hsv[:, :, 2]
    average_h = img_h.mean()
    average_s = img_s.mean()
    average_v = img_v.mean()
    result = []
    for pair in [(average_h, h), (average_s, s), (average_v, v)]:
        if pair[1]:
            result.append(pair[0])
    return result


def get_raw_pixel_features(data):
    if len(data.shape) == 3:
        result = data.reshape(
            (data.shape[0], data.shape[1] * data.shape[2]))
    elif len(data.shape) == 4:
        result = data.reshape(
            (data.shape[0], data.shape[1] * data.shape[2] * data.shape[3]))
    else:
        result = []
    return result


def get_width_height_ratio(img):
    ratio = img.shape[0] / img.shape[1]
    if ratio > 1:
        ratio = 1 / ratio
    return ratio


def get_global_color_features(data):
    result = []
    for i in range(len(data)):
        result.append([])
        img = data[i]
        result[-1].extend(calculate_average_hue_saturation(img, h=True, s=True, v=True))
        # result[-1].extend(calculate_hue_distribution(img))
    return result


def normalize_features(data, v_max=1.0, v_min=0.0):
    data_array = np.asarray(data, np.float32)
    mins = np.min(data_array, axis=0)
    maxs = np.max(data_array, axis=0)
    rng = maxs - mins
    result = v_max - ((v_max - v_min) * (maxs - data_array) / rng)
    return result


def get_d2_data(feature_result_list):
    d2_data = []
    for i in range(len(feature_result_list[0])):
        d2_data.append([])
    for feature_result in feature_result_list:
        for i in range(len(feature_result)):
            d2_data[i].extend(feature_result[i])
    d2_data = normalize_features(d2_data, v_max=1.0, v_min=0.0)
    return d2_data


if __name__ == '__main__':
    w = 50
    h = 50
    c = 3
    train_image_count = 10000
    category_count = 4
    train_path = r'D:\Projects\IoT_recognition\20181028\vis/'
    np.seterr(all='ignore')
    train_data, train_label, ratios = read_img_random(train_path, train_image_count, resize=(w, h), as_gray=True)
    d2_train_data = get_raw_pixel_features(train_data)
    # train_data, train_label, ratios = read_img_random(train_path, train_image_count, resize=(w, h))
    # result_list = [ratios,
    #                get_global_color_features(train_data)]
    # d2_train_data = get_d2_data(result_list)

    sio.savemat(train_path + 'raw_50.mat', mdict={'feature_matrix': d2_train_data, 'label': train_label})
