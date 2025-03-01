import numpy as np
from numpy.lib.stride_tricks import sliding_window_view


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
    Hi, Wi = image.shape[:2]
    Hk, Wk = kernel.shape[:2]

    kernel_f = kernel[::-1, ::-1].astype(float)

    res_image = np.copy(image).astype(float)

    for i in range(Hk // 2, Hi - Hk // 2 - 1):
        for j in range(Wk // 2, Wi - Wk // 2 - 1):
            total = 0.0
            for k in range(Hk):
                for l in range(Wk):
                    total += image[i + k][j + l] * kernel_f[k][l]
            res_image[i + 1, j + 1] = total

    # res_image = (res_image - res_image.min()) / (res_image.max() - res_image.min()) * 255

    return res_image


def zero_pad(image, pad_height, pad_width):
    """ Zero-pad an image.

    Ex: a 1x1 image [[1]] with pad_height = 1, pad_width = 2 becomes:

        [[0, 0, 0, 0, 0],
         [0, 0, 1, 0, 0],
         [0, 0, 0, 0, 0]]         of shape (3, 5)

    Args:
        image: numpy array of shape (H, W).
        pad_width: width of the zero padding (left and right padding).
        pad_height: height of the zero padding (bottom and top padding).

    Returns:
        out: numpy array of shape (H+2*pad_height, W+2*pad_width).
    """
    out = np.copy(image)
    # если нужно полное совпадение по значениям цветов на картинке убрать mode="edge"
    out = np.pad(out, ((pad_height, pad_height), (pad_width, pad_width)))

    return out


def conv_fast(image, kernel):
    """ An efficient implementation of convolution filter.

    This function uses element-wise multiplication and np.sum()
    to efficiently compute weighted sum of neighborhood at each
    pixel.

    Hints:
        - Use the zero_pad function you implemented above
        - There should be two nested for-loops
        - You may find np.flip() and np.sum() useful

    Args:
        image: numpy array of shape (Hi, Wi).
        kernel: numpy array of shape (Hk, Wk).

    Returns:
        out: numpy array of shape (Hi, Wi).
    """
    Hk, Wk = kernel.shape[:2]

    kernel_f = kernel[::-1, ::-1]
    out = zero_pad(image, Hk // 2, Wk // 2)

    windows = sliding_window_view(out, (Hk, Wk))

    out = np.tensordot(windows, kernel_f, axes=([2, 3], [0, 1]))

    # correlation_map = (out - out.min()) / (out.max() - out.min()) * 255

    # print(out.astype(np.uint8).shape)
    # print(image.shape)

    return out[Hk // 2: -(Hk // 2)][Wk // 2: -(Wk // 2)]


def conv_faster(image, kernel):
    """
    Args:
        image: numpy array of shape (Hi, Wi).
        kernel: numpy array of shape (Hk, Wk).

    Returns:
        out: numpy array of shape (Hi, Wi).
    """
    Hi, Wi = image.shape
    Hk, Wk = kernel.shape
    out = np.zeros((Hi, Wi))

    ### YOUR CODE HERE
    pass
    ### END YOUR CODE

    return out


def cross_correlation(image, template):
    Hk, Wk = template.shape[:2]

    res_map = zero_pad(image, Hk // 2, Wk // 2)

    windows = sliding_window_view(res_map, (Hk, Wk))

    res_map = np.sum(windows.astype(float) * template.astype(float), axis=(2, 3))

    res_map = (res_map - res_map.min()) / (res_map.max() - res_map.min()) * 255

    return res_map[Hk // 2: -(Hk // 2)][Wk // 2: -(Wk // 2)]


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


def normalized_cross_correlation(f, g):
    """ Normalized cross-correlation of f and g.

    Normalize the subimage of f and the template g at each step
    before computing the weighted sum of the two.

    Hint: you should look up useful numpy functions online for calculating 
          the mean and standard deviation.

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
    windows_norm = (windows - np.mean(windows, axis=(2, 3), keepdims=True)) / np.std(windows, axis=(2, 3), keepdims=True)
    temp_norm = (temp_zero_mean - np.mean(g)) / np.std(temp_zero_mean)
    res_map = np.sum(windows_norm * temp_norm, axis=(2, 3))

    return res_map
