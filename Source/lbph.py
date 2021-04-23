import cv2
import numpy as np
from definitions import CHI_SQUARE, EUCLIDEAN, NORMALIZED_EUCLIDEAN, ABSOLUTE


def hist(mat):
    return cv2.calcHist([mat], [0], None, [256], [0, 255])


def class_hist(list_mat):
    aux = np.zeros((256, 1))
    for mat in list_mat:
        aux = aux + hist(mat)
    return aux / len(list_mat)


def compare(hist1, hist2, metric):
    # hist1 = np.array(hist1)
    # hist2 = np.array(hist2)
    if metric == CHI_SQUARE:
        pass
    if metric == EUCLIDEAN:
        p1 = np.sum(hist1 ** 2, axis = 1)[:, np.newaxis]
        p2 = np.sum(hist2 ** 2, axis = 1)
        p3 = -2 * np.dot(hist1, hist2.T)
        return np.round(np.sqrt(p1 + p2 + p3), 5)
    if metric == NORMALIZED_EUCLIDEAN:
        pass
    if metric == ABSOLUTE:
        return np.absolute(hist1 - hist2)


def find_closest_histogram(class_histograms, hist, metric = EUCLIDEAN):
    min_dist = 100000
    best_hist = None
    for k, h in class_histograms.items():
        dist = np.sum(compare(h, hist, metric))
        if metric == EUCLIDEAN:
            dist = np.sqrt(dist)
        if dist < min_dist:
            min_dist = dist
            best_hist = k
    return best_hist
