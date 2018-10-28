import os
import shutil
import random


def make_dir(path):
    if not os.path.exists(path):
        os.makedirs(path)
    else:
        shutil.rmtree(path)
        os.makedirs(path)


root_dir = r'D:\Projects\IoT_recognition\20181028\all_raw/'
output_dir = r'D:\Projects\IoT_recognition\20181028\samples_5000/'
make_dir(output_dir)
sample_count = 5000

file_name_list = [file_name for file_name in os.listdir(root_dir)
                  if os.path.isfile(os.path.join(root_dir, file_name))]
random.shuffle(file_name_list)

for file_name in file_name_list[:sample_count]:
    file_to_copy = os.path.join(root_dir, file_name)
    if os.path.isfile(file_to_copy) == True:
        shutil.copy(file_to_copy, output_dir)
