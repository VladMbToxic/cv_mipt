import cv2
import numpy as np
import matplotlib.pyplot as plt


def road_counter(image):
    pix = 256
    low_gray = (0.9 * 213, 0.9 * 213, 0.9 * 213)
    high_gray = (1.1 * pix, 1.1 * pix, 1.1 * pix)

    gray_area = cv2.inRange(image, low_gray, high_gray)

    road_number = 0
    height, width = gray_area.shape

    for i in range(1, width - 1):
        if gray_area[406, i] == 255 and gray_area[406, i + 1] == 0:
            road_number += 1
    return road_number


def road_center(image):
    pix = 256
    low_gray = (0.9 * 213, 0.9 * 213, 0.9 * 213)
    high_gray = (1.1 * pix, 1.1 * pix, 1.1 * pix)

    gray_area = cv2.inRange(image, low_gray, high_gray)
    road_number = road_counter(image)

    height, width = gray_area.shape

    center_arr = list()

    for i in range(width - 1):
        if gray_area[406, i] == 0 and gray_area[406, i + 1] == 255:
            begin = i
            road_width = 0
            while (i < width and gray_area[406, i] != 0):
                road_width += 1
                i += 1
            center = (begin + (begin + road_width)) / 2
            center_arr.append(center)
    return center_arr


def open_road(image):
    pix = 256
    low_gray = (0.9 * 213, 0.9 * 213, 0.9 * 213)
    high_gray = (1.1 * pix, 1.1 * pix, 1.1 * pix)

    gray_area = cv2.inRange(image, low_gray, high_gray)
    road_number = road_counter(image)

    height, width = gray_area.shape

    center_arr = road_center(image)
    open_road_arr = [True] * min(5, len(center_arr))

    for i in range(len(open_road_arr)):
        for j in range(2, height - 1):
            gray_area[int(center_arr[i]), j] = 150
            # if (gray_area[j, int(center_arr[i])] == 0):
            #     open_road_arr[i] = False
            #     break

    fig, m_axs = plt.subplots(1, 2, figsize=(12, 9))
    ax1, ax2 = m_axs

    ax1.set_title('Исходная картинка', fontsize=15)
    ax1.imshow(image)
    ax1.axis('off')

    ax2.set_title('Только дороги', fontsize=15)
    ax2.imshow(gray_area)
    ax2.axis('off')

    plt.show()

    return open_road_arr


image = cv2.imread('task_2/image_01.jpg')
image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

pix = 256
low_gray = (0.9 * 213, 0.9 * 213, 0.9 * 213)
high_gray = (1.1 * pix, 1.1 * pix, 1.1 * pix)

gray_area = cv2.inRange(image, low_gray, high_gray)

print(open_road(image))

# fig, m_axs = plt.subplots(1, 2, figsize=(12, 9))
# ax1, ax2 = m_axs
#
# ax1.set_title('Исходная картинка', fontsize=15)
# ax1.imshow(image)
# ax1.axis('off')
#
# ax2.set_title('Только дороги', fontsize=15)
# ax2.imshow(gray_area, cmap='gray')
# ax2.axis('off')
#
# plt.show()