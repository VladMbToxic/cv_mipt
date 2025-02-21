import cv2
import numpy as np
import matplotlib.pyplot as plt
from time import time

image = cv2.imread('task_1/20 by 20 orthogonal maze.png')
image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
height, width = image.shape[0:2]

simple_image = cv2.inRange(image, np.array([100, 100, 100]), np.array([255, 255, 255]))
res_image = np.ndarray(simple_image.shape)

# print(res_image)
kernel = np.ones((11,11),np.uint8)
simple_image = cv2.erode(simple_image, kernel, iterations=1)


start = [height-2, width//2]
end = [2, width//2]
kernel = np.ones((3,3),np.uint8)

res_image[start[0]][start[1]] = 255
res_image[end[0]][end[1]] = 255
s = time()
res_image = cv2.dilate(res_image, kernel, iterations = 5)
for i in range(500):
    res_image = cv2.dilate(res_image, kernel, iterations = 1)
    res_image = np.where(res_image > 0, 255, 0)
    res_image = np.where(np.logical_and(simple_image, res_image), 255, 0).astype('uint8')
    print(res_image.shape)
e = time()
print(e-s)

fig, m_axs = plt.subplots(1, 2, figsize = (12, 9))
ax1, ax2 = m_axs

ax1.set_title('Исходная картинка', fontsize=15)
ax1.imshow(simple_image)
ax2.set_title('Как я это вижу', fontsize=15)
ax2.imshow(res_image)

plt.show()