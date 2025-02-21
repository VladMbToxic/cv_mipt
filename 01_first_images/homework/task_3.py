import cv2
import numpy as np


def rotate(image, point: tuple, angle: float) -> np.ndarray:
    """
    Повернуть изображение по часовой стрелке на угол от 0 до 360 градусов и преобразовать размер изображения.

    :param image: исходное изображение
    :param point: значение точки (x, y), вокруг которой повернуть изображение
    :param angle: угол поворота
    :return: повернутное изображение
    """
    height, width = image.shape[0:2]

    rot_mat = cv2.getRotationMatrix2D(point, angle, scale=1)

    cos = abs(rot_mat[0, 0])
    sin = abs(rot_mat[0, 1])

    new_width = int((height * sin) + (width * cos))
    new_height = int((height * cos) + (width * sin))

    rot_mat[0, 2] = 0
    rot_mat[1, 2] = width * sin

    res_image = cv2.warpAffine(image, rot_mat, (new_width, new_height))

    return res_image

