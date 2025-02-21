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


def doc_scanning(image):
    height, width = image.shape[:2]
    image_mask = cv2.cvtColor(image, cv2.COLOR_RGB2HSV)

    pix = 255
    image_mask = cv2.inRange(image_mask, (0, 0.65*pix, 0.2*pix), (10/2, 0.85*pix, 0.8*pix)) + cv2.inRange(image_mask, (350/2,  0.65*pix, 0.2*pix), (360/2, 0.9*pix, 0.9*pix))

    corners = [None, None, None, None]
    new_corners = np.float32([(0, height), (0, 0), (width, 0), (width, height)])

    for y in range(height):
        for x in range(width):
            if corners[1] is not None and corners[3] is not None:
                break
            if image_mask[y][x] != 0:
                corners[1] = (x, y)
            if image_mask[height - y - 1][x] != 0:
                corners[3] = (x, height - y - 1)


    for x in range(width):
        for y in range(height):
            if corners[0] is not None and corners[2] is not None:
                break
            if image_mask[y][x] != 0:
                corners[0] = (x, y)
            if image_mask[y][width - x - 1] != 0:
                corners[2] = (width - x - 1, y)

    affin_mat = cv2.getPerspectiveTransform(np.float32(corners), new_corners)

    res_image = cv2.warpPerspective(image, affin_mat, (width, height))

    return res_image


image = cv2.imread('task_3/notebook.jpg')
image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

res_image = doc_scanning(image)

fig, m_axs = plt.subplots(1, 2, figsize = (12, 9))
ax1, ax2 = m_axs

ax1.set_title('Исходная картинка', fontsize=15)
ax1.imshow(image)
ax2.set_title('Только то что надо', fontsize=15)
ax2.imshow(res_image)

plt.show()