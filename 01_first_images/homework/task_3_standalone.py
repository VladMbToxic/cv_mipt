import cv2
import numpy as np
import matplotlib.pyplot as plt


def rotate(image, point, angle):
    height, width = image.shape[0:2]

    rot_mat = cv2.getRotationMatrix2D(point, angle, scale=1)
    print(rot_mat)

    # res_image = cv2.warpAffine(image, rot_mat, (width, height))

    cos = abs(rot_mat[0, 0])
    sin = abs(rot_mat[0, 1])

    new_width = int((height * sin) + (width * cos))
    new_height = int((height * cos) + (width * sin))
    #
    rot_mat[0, 2] = 0
    rot_mat[1, 2] = width * sin

    res_image = cv2.warpAffine(image, rot_mat, (new_width, new_height))

    return res_image

image = cv2.imread('task_3/lk.jpg')
image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
image_rotated = cv2.imread('task_3/lk_rotate.jpg')
height, width = image.shape[0:2]
h, w = image_rotated.shape[0:2]
print(h, w)

res_image = rotate(image, (width//2, height//2), 15)

fig, m_axs = plt.subplots(1, 2, figsize = (12, 9))
ax1, ax2 = m_axs

ax1.set_title('Исходная картинка', fontsize=15)
ax1.imshow(image_rotated)
ax2.set_title('Только то что надо', fontsize=15)
ax2.imshow(res_image)

plt.show()