import numpy as np
import matplotlib.pyplot as plt
import cv2
from numpy.lib.stride_tricks import sliding_window_view

plt.rcParams['figure.figsize'] = (10.0, 8.0) # set default size of plots
plt.rcParams['image.interpolation'] = 'nearest'
plt.rcParams['image.cmap'] = 'gray'

from filters import zero_pad

def zero_mean_cross_correlation(f, g):
    """ Zero-mean cross-correlation of f and g.

    Subtract the mean of g from g so that its mean becomes zero.

    Hint: you should look up useful numpy functions online for calculating the mean.

    Args:
        f: numpy array of shape (Hf, Wf).
        g: numpy array of shape (Hg, Wg).

    Returns:
        out: numpy array of shape (Hf, Wf).
    """
    Ht, Wt = g.shape

    temp_zero_mean = g.astype(np.float32) - np.mean(g.astype(np.float32))
    img_padded = np.pad(f.astype(np.float32), ((Ht // 2, Ht // 2), (Wt // 2, Wt // 2)))

    windows = sliding_window_view(img_padded, (Ht, Wt))
    res_map = np.sum(windows * temp_zero_mean, axis=(2, 3))

    return res_map

img = cv2.imread('img/shelf.jpg')
img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
img_grey = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
temp = cv2.imread('img/template_3.jpg')
temp = cv2.cvtColor(temp, cv2.COLOR_BGR2RGB)
temp_grey = cv2.cvtColor(temp, cv2.COLOR_BGR2GRAY)

# Perform cross-correlation between the image and the template
out = zero_mean_cross_correlation(img_grey, temp_grey)

# Find the location with maximum similarity
y, x = np.unravel_index(out.argmax(), out.shape)

plt.figure(figsize=(30, 20))

# Display product template
plt.subplot(311), plt.imshow(temp), plt.title('Template'), plt.axis('off')

# Display image
plt.subplot(312), plt.imshow(img), plt.title('Result (blue marker on the detected location)'), plt.axis('off')
plt.plot(x, y, 'bx', ms=20, mew=6)

# Display cross-correlation output
plt.subplot(313), plt.imshow(out), plt.title('Cross-correlation (white means more correlated)'), plt.axis('off')

# Draw marker at detected location
plt.plot(x, y, 'bx', ms=20, mew=6)

plt.show()