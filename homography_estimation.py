__author__ = 'DongwonShin'

import numpy as np
import math
import cv2


def draw_points(img, points, color):
    i = 0
    for p in points:
        cv2.circle(img, (int(p[0]), int(p[1])), 5, color, 1)
        cv2.putText(img, str(i), (int(p[0]), int(p[1])), cv2.FONT_HERSHEY_SIMPLEX, 1, color)
        i = i + 1

def homography_estimation(src_img, pattern_features, horizontal_unit_num, vertical_unit_num, unit_square_size, scale_factor):

    ideal_pattern_features = []
    pract_pattern_features = []

    for pf in pattern_features:
        pract_pattern_features.append(pf.pt)
    pract_pattern_features = np.array(pract_pattern_features)

    for j in range(0,horizontal_unit_num+1):
        for i in range(0,vertical_unit_num+1):
            ideal_pattern_features.append((unit_square_size * j, unit_square_size * i))
    ideal_pattern_features = np.array(ideal_pattern_features)
    ret_ideal_pattern_features = ideal_pattern_features

    ideal_pattern_features = np.array(ideal_pattern_features) * scale_factor
    ideal_pattern_features = ideal_pattern_features + (np.mean(pract_pattern_features, axis=0) - np.mean(ideal_pattern_features, axis=0))

    def distance(a,b):
        ax = a[0]
        ay = a[1]
        bx = b[0]
        by = b[1]

        return math.sqrt((ax-bx)**2 + (ay-by)**2)

    ideal_pattern_corr = []
    pract_pattern_corr = []

    t_ideal_pattern_features = ideal_pattern_features.copy()
    for ppf in pract_pattern_features:
        pract_pattern_corr.append(ppf)
        dist = np.zeros((len(t_ideal_pattern_features),1))

        i=0
        for ipf in t_ideal_pattern_features:
            dist[i] = distance(ppf, ipf)
            i = i+1

        ideal_pattern_corr.append(t_ideal_pattern_features[np.argmin(dist)])
        t_ideal_pattern_features = np.delete(t_ideal_pattern_features,np.argmin(dist), 0)

    ideal_pattern_corr = np.array(ideal_pattern_corr)
    pract_pattern_corr = np.array(pract_pattern_corr)
    H =  cv2.findHomography(ideal_pattern_corr, pract_pattern_corr)[0]

    draw_points(src_img, ideal_pattern_corr, (0,0,255))
    draw_points(src_img, pract_pattern_corr, (0,255,0))

    ideal_pattern_features = np.append(ideal_pattern_features, np.ones((len(ideal_pattern_features),1)),axis=1)
    ideal_pattern_features = np.dot(H,ideal_pattern_features.T).T
    for i in range(len(ideal_pattern_features)):
        ideal_pattern_features[i] = ideal_pattern_features[i]/ideal_pattern_features[i][2]
    ideal_pattern_features = ideal_pattern_features[:,0:2]

    # print pattern_features
    # i=0
    # for pf in ideal_pattern_features:
    #     cv2.circle(img, (int(pf[0]), int(pf[1])), 5, (0,255,0), 1)
    #     cv2.putText(img, str(i),(int(pf[0]), int(pf[1])),cv2.FONT_HERSHEY_SIMPLEX, 1, (0,255,0))
    #     i = i+1
    # i=0
    # for pf in pract_pattern_features:
    #     cv2.circle(img, (int(pf[0]), int(pf[1])), 5, (255,0,0), 3)
    #     cv2.putText(img, str(i),(int(pf[0]), int(pf[1])),cv2.FONT_HERSHEY_SIMPLEX, 1, (255,0,0))
    #     i = i+1

    return ideal_pattern_features, ret_ideal_pattern_features
