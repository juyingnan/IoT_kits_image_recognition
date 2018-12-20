import shutil
import os


def make_dir(path):
    if not os.path.exists(path):
        os.makedirs(path)
    else:
        shutil.rmtree(path)
        os.makedirs(path)


root_path = r"D:\Projects\IoT_recognition\20181205\keras\TRAIN/"
folder_name_list = ['7_segment_display',
                    '9v_battery',
                    '16_pin_chip',
                    'blank',
                    'buzzer',
                    'capacitor',
                    'ds3231_rtc_module',
                    'gy-521_module',
                    'ir_receiver_module',
                    'joystick_module',
                    'lcd_module',
                    'max7219_module',
                    'mega2560_controller_board',
                    'membrance_switch_module',
                    'motor',
                    'pir_motion_sensor_HC-SR501',
                    'power_supply_module',
                    'prototype_expansion_board',
                    'relay_5v',
                    'remote',
                    'rfid_module',
                    'rotary_encoder_module',
                    'servo_motor',
                    'sound_sensor_module',
                    'stepper_motor',
                    'stepper_motor_driver_board_uln2003',
                    'temperature_and_humidity_module_DHT11',
                    'tilt_ball_switch',
                    'transistor',
                    'ultrasonic_sensor',
                    'water_level_detection_sensor_module',
                    ]
for folder_name in folder_name_list:
    folder_path = root_path + folder_name + '/'
    make_dir(folder_path)
