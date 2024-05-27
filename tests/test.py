import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import cv2
import tensorflow as tf
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix, classification_report, accuracy_score
from tensorflow.keras import layers, models
from tensorflow.keras.preprocessing.image import ImageDataGenerator, img_to_array, load_img
from tensorflow import keras


# Функция для проверки размера изображения
def test_images_size(img_list):
    flag_1 = True
    print("Начался тест 1 на размер изображений")
    for i in range(len(img_list)):
        x = img_to_array(img_list[i])
        if x.shape != (96, 96, 3):
            print(f"Тест не пройден, ожидался размер (96, 96, 3), а получен {x.shape}")
            flag_1 = False
    if flag_1: 
        print(f"Тест 1 пройден успешно", end="\n")
    else: print("Тест 1 не пройден")


# Функция для проверки метрик f1 и accuracy на тестовых данных
def test_metrics(metrics_dict, emotions):
    flag_2 = True
    print("Начался тест 2 на метрики f_1 и accuracy")
    for i in emotions:
        if metrics_dict[i]['f1-score'] <= 0.5:
            print(f"Тест не пройден, ожидалось значение метрики f1 >= 0.5, а получено {metrics_dict[i]['f1-score']}")
            flag_2 = False
    if metrics_dict['accuracy'] <= 0.5:
        print(f"Тест не пройден, ожидалось значение метрики accuracy >= 0.5, а получено {metrics_dict[i]['accuracy']}")
        flag_2 = False
    if flag_2: 
        print(f"Тест 2 пройден успешно", end="\n")
    else: print("Тетс 2 не пройден")


# Функция для проверки структуры модели
def test_model_structure(model):
    flag_3 = True
    print("Начался тест 3 на структуру модели")
    if len(model.layers) != 8:
        print(f"Ожидалось 8 слоев, а получено {len(model.layers)}")
        flag_3 = False
    if model.layers[0].__class__.__name__ != "Conv2D":
        print(f"Ожидалось 1-й слой Conv2D, а получено {model.layers[0].__class__.__name__}")
        flag_3 = False
    if model.layers[1].__class__.__name__ != "Conv2D":
        print(f"Ожидалось 2-й слой Conv2D, а получено {model.layers[1].__class__.__name__}")
        flag_3 = False
    if model.layers[2].__class__.__name__ != "MaxPooling2D":
        print(f"Ожидалось 3-й слой MaxPooling2D, а получено {model.layers[2].__class__.__name__}")
        flag_3 = False
    if model.layers[3].__class__.__name__ != "Conv2D":
        print(f"Ожидалось 4-й слой Conv2D, а получено {model.layers[3].__class__.__name__}")
        flag_3 = False
    if model.layers[4].__class__.__name__ != "MaxPooling2D":
        print(f"Ожидалось 5-й слой MaxPooling2D, а получено {model.layers[4].__class__.__name__}")
        flag_3 = False
    if model.layers[5].__class__.__name__ != "Flatten":
        print(f"Ожидалось 6-й слой Flatten, а получено {model.layers[5].__class__.__name__}")
        flag_3 = False
    if model.layers[6].__class__.__name__ != "Dense":
        print(f"Ожидалось 7-й слой Dense, а получено {model.layers[6].__class__.__name__}")
        flag_3 = False
    if model.layers[7].__class__.__name__ != "Dense":
        print(f"Ожидалось 8-й слой Dense, а получено {model.layers[7].__class__.__name__}")
        flag_3 = False
    if model.count_params() != 8023688:
        print(f"Ожидалось количество параметров 8023688, а получено {model.count_params()}")
        flag_3 = False
    if flag_3:
        print(f"Тест 3 пройден успешно", end="\n")
    else: print("Тест 3 не пройден")


# Функция для проверки на количество изображений в train/test
def test_size_of_train_test(X_train, X_test):
    flag_4 = True
    print("Начался тест 4 на количество объектов в train/test")
    if X_train.shape[0] < 16414:
        print(f"Тест не пройден, ожидался размер train не менее 16414, а получен {X_train.shape[0]}")
        flag_4 = False
    if X_test.shape[0] < 4104:
        print(f"Тест не пройден, ожидался размер test не менее 4104, а получен {X_test.shape[0]}")
        flag_4 = False
    if flag_4: 
        print(f"Тест 4 пройден успешно", end="\n")
    else: print("Тест 4 не пройден")
