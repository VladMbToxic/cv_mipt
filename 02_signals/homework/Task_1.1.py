import numpy as np
import matplotlib.pyplot as plt
import cv2

plt.rcParams['figure.figsize'] = (10.0, 8.0)  # set default size of plots
plt.rcParams['image.interpolation'] = 'nearest'
plt.rcParams['image.cmap'] = 'gray'


def conv_nested(image, kernel):
    """A naive implementation of convolution filter.

    This is a naive implementation of convolution using 4 nested for-loops.
    This function computes convolution of an image with a kernel and outputs
    the result that has the same shape as the input image.

    Args:
        image: numpy array of shape (Hi, Wi).
        kernel: numpy array of shape (Hk, Wk).

    Returns:
        out: numpy array of shape (Hi, Wi).
    """
    Hi, Wi = image.shape
    Hk, Wk = kernel.shape

    kernel_f = kernel[::-1, ::-1].astype(float)

    res_image = np.copy(image).astype(float)

    for i in range(1, Hi - 2):
        for j in range(1, Wi - 2):
            total = 0.0
            for k in range(Hk):
                for l in range(Wk):
                    total += image[i + k][j + l] * kernel_f[k][l]
            res_image[i + 1, j + 1] = total

    res_image = (res_image - res_image.min()) / (res_image.max() - res_image.min()) * 255

    return res_image


# Simple convolution kernel.
kernel = np.array(
    [
        [1, 0, 1],
        [0, 0, 0],
        [1, 0, 0]
    ])

# Create a test image: a white square in the middle
test_img = np.zeros((9, 9))
test_img[3:6, 3:6] = 1

# Run your conv_nested function on the test image
test_output = conv_nested(test_img, kernel)

# Build the expected output
expected_output = np.zeros((9, 9))
expected_output[2:7, 2:7] = 1
expected_output[5:, 5:] = 0
expected_output[4, 2:5] = 2
expected_output[2:5, 4] = 2
expected_output[4, 4] = 3

# Plot the test image
plt.subplot(1, 3, 1)
plt.imshow(test_img)
plt.title('Test image')
plt.axis('off')

# Plot your convolved image
plt.subplot(1, 3, 2)
plt.imshow(test_output)
plt.title('Convolution')
plt.axis('off')

# Plot the expected output
plt.subplot(1, 3, 3)
plt.imshow(expected_output)
plt.title('Exepected output')
plt.axis('off')
plt.show()

img = cv2.imread('img/dog.jpg', 0)

kernel = np.array(
[
    [1,0,-1],
    [2,0,-2],
    [1,0,-1]
])

out = conv_nested(img, kernel)
# out = cv2.filter2D(img, -1, kernel)

# Plot original image
plt.subplot(221), plt.imshow(img), plt.title('Original'), plt.axis('off')

# Plot your convolved image
plt.subplot(223), plt.imshow(out), plt.title('Convolution'), plt.axis('off')

# Plot what you should get
solution_img = cv2.imread('img/convoluted_dog.jpg', 0)
plt.subplot(224), plt.imshow(solution_img), plt.title('What you should get'), plt.axis('off')

print(out)

plt.show()
