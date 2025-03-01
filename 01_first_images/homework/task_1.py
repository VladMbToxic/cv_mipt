import cv2
import numpy as np
from collections import deque


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

def find_way_from_maze(image: np.ndarray) -> tuple:
    """
    Найти путь через лабиринт.

    :param image: изображение лабиринта
    :return: координаты пути из лабиринта в виде (x, y), где x и y - это массивы координат
    """
    image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    block_side = 16  # Size of one maze block (which contains upper and left wall and free space)
    res_image = simplify(image, block_side)

    start = find_opening_on_side(res_image, 1)
    end = find_opening_on_side(res_image, 3)

    path = find_path(res_image, start, end)
    for point in path:
        res_image[point[0]][point[1]] = 150
    coords = [point[0] * 8 for point in path], [point[1] * 8 for point in path]

    return coords
