import cv2
import numpy as np


def find_road_number(image: np.ndarray) -> int:
    """
    Найти номер дороги, на которой нет препятсвия в конце пути.

    :param image: исходное изображение
    :return: номер дороги, на котором нет препятсвия на дороге
    """
    image_hsv = cv2.cvtColor(image, cv2.COLOR_RGB2HSV)
    # height, width = image.shape[0:2]

    pix = 255

    # найдем полосы (ака кол-во дорог)
    yellow_low = (55 / 2, 0.2 * pix, 0.9 * pix)
    yellow_high = (60 / 2, 0.6 * pix, 1 * pix)

    lines_area = cv2.inRange(image_hsv, yellow_low, yellow_high)

    road_number = 0
    center_arr = np.ndarray(0, dtype=int)
    height, width = lines_area.shape

    begin = -1
    road_width = -1
    for i in range(width - 1):
        if lines_area[height // 2, i] == 255 and lines_area[height // 2, i + 1] == 0:
            road_number += 1
            begin = i
            road_width = 0
            # print(f"{i} start")
        elif lines_area[height // 2, i] == 0 and lines_area[height // 2, i + 1] == 255:
            center = (begin + (begin + road_width)) // 2
            center_arr = np.append(center_arr, center)
            # print(f"{i} end")
        elif lines_area[height // 2, i] == 0:
            road_width += 1
            # print(f"{i} road")

    # print(road_number)
    # print(center_arr)

    # ищем квадраты
    red_low = (5 / 2, 0.9 * pix, 0.9 * pix)
    red_high = (10 / 2, 1 * pix, 1 * pix)

    road_blocked = [False] * road_number
    # road_color = np.ndarray(0)

    squares_area = cv2.inRange(image_hsv, red_low, red_high)
    for i in range(road_number):
        for j in range(height - 1):
            # squares_area[j][center_arr[i]] = 150
            # road_color = np.append(road_color, squares_area[j][center_arr[i]])
            if squares_area[j][center_arr[i]] == 255:
                road_blocked[i] = True
                break
        # print(road_color)
    # print(road_blocked)

    # ищем машину
    blue_low = (210 / 2, 0.7 * pix, 0.9 * pix)
    blue_high = (230 / 2, 1 * pix, 1 * pix)

    car_lane = -1

    # np.set_printoptions(threshold=np.inf)
    car_area = cv2.inRange(image_hsv, blue_low, blue_high)
    for i in range(road_number):
        for j in range(height - 1):
            if car_area[j][center_arr[i]] == 255:
                car_lane = i
                break
    # print(car_lane)

    if not road_blocked[car_lane]:
        return -1
    else:
        return road_blocked.index(False)
