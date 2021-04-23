import numpy as np
from scipy.signal import convolve2d

#Filter to make a series of convolve operations
f1 = np.array([[1, 0, 0], [0, -1, 0], [0, 0, 0]])
f2 = np.array([[0, 1, 0], [0, -1, 0], [0, 0, 0]])
f3 = np.array([[0, 0, 1], [0, -1, 0], [0, 0, 0]])
f4 = np.array([[0, 0, 0], [0, -1, 1], [0, 0, 0]])
f5 = np.array([[0, 0, 0], [0, -1, 0], [0, 0, 1]])
f6 = np.array([[0, 0, 0], [0, -1, 0], [0, 1, 0]])
f7 = np.array([[0, 0, 0], [0, -1, 0], [1, 0, 0]])
f8 = np.array([[0, 0, 0], [1, -1, 0], [0, 0, 0]])


def threshold(arr):
    aux = [1 if elem >= 0 else 0 for vec in arr for elem in vec]
    return np.reshape(aux, arr.shape)


def calculate_lbp(gray_img):
    ###Using a 3x3 kernel
    c1 = convolve2d(gray_img, f1, mode = 'same')
    c2 = convolve2d(gray_img, f2, mode = 'same')
    c3 = convolve2d(gray_img, f3, mode = 'same')
    c4 = convolve2d(gray_img, f4, mode = 'same')
    c5 = convolve2d(gray_img, f5, mode = 'same')
    c6 = convolve2d(gray_img, f6, mode = 'same')
    c7 = convolve2d(gray_img, f7, mode = 'same')
    c8 = convolve2d(gray_img, f8, mode = 'same')
    
    t1 = threshold(c1)
    t2 = threshold(c2)
    t3 = threshold(c3)
    t4 = threshold(c4)
    t5 = threshold(c5)
    t6 = threshold(c6)
    t7 = threshold(c7)
    t8 = threshold(c8)
    
    return np.uint8(t1 + t2 * 2 + t3 * 4 + t4 * 8 + t5 * 16 + t6 * 32 + t7 * 64 + t8 * 128)
