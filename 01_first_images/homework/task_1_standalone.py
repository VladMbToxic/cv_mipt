import cv2
import numpy as np
import matplotlib.pyplot as plt
from time import time
from collections import deque
np.set_printoptions(threshold=np.inf)

def simplify(image, block_side):
    """
    simplifies maze to have walls and routes 1 pixel thick
    :param image: block-structured image
    :param block_side: one block of maze, left and up are walls
    :return: image with walls and frees spaces as 1 pixel
    """
    height, width = image.shape[0:2]

    indices = []
    for k in range(height // block_side):
        indices.append(block_side * k)  # One pixel from wall part of block
        indices.append((block_side - 1) + block_side * k)  # One pixel from freespace part of block
    indices.append(image.shape[0] - 1)
    indices = sorted(indices)
    res_image = image[indices, :]
    indices = []
    for k in range(width // block_side):
        indices.append(block_side * k)  # One pixel from wall part of block
        indices.append((block_side - 1) + block_side * k)  # One pixel from freespace part of block
    indices.append(image.shape[1] - 1)
    indices = sorted(indices)
    res_image = res_image[:, indices]
    return res_image

def find_opening_on_side(image, side):
    """
    INTENDED FOR WORK WITH SIMPLIFIED IMAGES
    :param image: simplified block-structured image of maze
    :param side: 1 - up, 2 - down, 3 - left, 4 -right
    :return: point with opening in the wall
    """
    height, width = image.shape[0:2]
    if side == 1 or side == 3:
        if side == 1:
            for i in range(width):
                if image[0][i] == 255:
                    return 0, i
        else:
            for i in range(width):
                if image[height - 1][i] == 255:
                    return height - 1, i
    else:
        if side == 2:
            for i in range(height):
                if image[i][0] == 255:
                    return i, 0
        else:
            for i in range(height):
                if image[i][width - 1] == 255:
                    return i, width - 1

def find_path(matrix, start, end):
    height, width = matrix.shape[0:2]
    visited = np.array([[False for _ in range(width)] for _ in range(height)])
    parent = {}
    directions = [(-1, 0), (1, 0), (0, -1), (0, 1)]  # Up, down, left, right

    queue = deque()
    queue.append(start)
    visited[start[0]][start[1]] = True

    while queue:
        current = queue.popleft()
        if current == end:
            break

        for dx, dy in directions:
            x = current[0] + dx
            y = current[1] + dy
            if 0 <= x < height and 0 <= y < width:
                if matrix[x][y] == 255 and not visited[x][y]:
                    visited[x][y] = True
                    parent[(x, y)] = current
                    queue.append((x, y))

    # Восстановление пути
    path = []
    current = end
    while current != start:
        path.append(current)
        current = parent[current]
    path.append(start)
    path.reverse()

    return path

def plot_maze_path(image: np.ndarray, coords: tuple) -> np.ndarray:
    """
    Нарисовать путь через лабиринт на изображении.
    Вспомогательная функция.

    :param image: изображение лабиринта
    :param coords: координаты пути через лабиринт типа (x, y) где x и y - массивы координат точек
    :return img_wpath: исходное изображение с отрисованными координатами
    """
    if image.ndim != 3:
        image = cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)

    img_wpath = image.copy()
    if coords:
        x, y = coords
        img_wpath[x, y, :] = [0, 0, 255]

    return img_wpath

image = cv2.imread('task_1/img_01.png')
image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

s = time()

block_side = 16 # Size of one maze block (which contains upper and left wall and free space)
res_image = simplify(image, block_side)

# print(res_image)
height, width = res_image.shape[0:2]

start = find_opening_on_side(res_image, 1)
end = find_opening_on_side(res_image, 3)

path = find_path(res_image, start, end)
for point in path:
    res_image[point[0]][point[1]] = 150
path_t = [point[0]*8 for point in path], [point[1]*8 for point in path]

# SOME BULLSHITTERY
# res_image = cv2.cvtColor(res_image, cv2.COLOR_GRAY2BGR)
# cv2.floodFill(res_image, None, (0, 0), (0, 255, 0), loDiff=(0, 0, 0, 0), upDiff=(0, 0, 0, 0))
# cv2.floodFill(res_image, None, (height-1, width-1), (255, 0, 0), loDiff=(0, 0, 0, 0), upDiff=(0, 0, 0, 0))

e = time()
print(e - s)

fig, m_axs = plt.subplots(1, 2, figsize = (12, 9))
ax1, ax2 = m_axs

ax1.set_title('Исходная картинка', fontsize=15)
ax1.imshow(image)
ax2.set_title('Как я это вижу', fontsize=15)
ax2.imshow(plot_maze_path(image, (path_t)))

plt.show()