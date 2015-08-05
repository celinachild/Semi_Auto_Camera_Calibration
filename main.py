__author__ = 'DongwonShin'

import numpy as np
import cv2
from matplotlib import pyplot as plt
import math

param_FAST = 30
rad_circular_sampling = 10
cam_num = 4

def circle_check1(img_thresh, _kp, rad):

    flag = True

    cx = _kp.pt[0]
    cy = _kp.pt[1]

    lower_bound = 20
    upper_bound = 255-lower_bound

    for m in range(1,3):
        for k in range(3,7):
            if cy + rad < img_thresh.shape[0] and cx + rad < img_thresh.shape[1] and cy - rad > 0 and cx - rad > 0:
                 flag = (flag and (img_thresh[cy + (math.sin(math.pi/k)*rad*m)][cx + (math.cos(math.pi/2 - math.pi/k)*rad*m)] < lower_bound)
                              and (img_thresh[cy - (math.sin(math.pi/k)*rad*m)][cx - (math.cos(math.pi/2 - math.pi/k)*rad*m)] < lower_bound)
                              and (img_thresh[cy + (math.sin(math.pi/k)*rad*m)][cx - (math.cos(math.pi/2 - math.pi/k)*rad*m)] > upper_bound)
                              and (img_thresh[cy - (math.sin(math.pi/k)*rad*m)][cx + (math.cos(math.pi/2 - math.pi/k)*rad*m)] > upper_bound))

    return flag

def circle_check2(img_thresh, _kp, rad):

    flag = True

    cx = _kp.pt[0]
    cy = _kp.pt[1]

    lower_bound = 20
    upper_bound = 255-lower_bound

    for m in range(1,3):
        for k in range(3,7):
            if cy + rad < img_thresh.shape[0] and cx + rad < img_thresh.shape[1] and cy - rad > 0 and cx - rad > 0:
                 flag = (flag and (img_thresh[cy + (math.sin(math.pi/k)*rad*m)][cx + (math.cos(math.pi/2 - math.pi/k)*rad*m)] > lower_bound)
                              and (img_thresh[cy - (math.sin(math.pi/k)*rad*m)][cx - (math.cos(math.pi/2 - math.pi/k)*rad*m)] > lower_bound)
                              and (img_thresh[cy + (math.sin(math.pi/k)*rad*m)][cx - (math.cos(math.pi/2 - math.pi/k)*rad*m)] < upper_bound)
                              and (img_thresh[cy - (math.sin(math.pi/k)*rad*m)][cx + (math.cos(math.pi/2 - math.pi/k)*rad*m)] < upper_bound))

    return flag


def circular_sampling(img, kp, rad):
    img_gray = cv2.cvtColor(img, cv2.cv.CV_RGB2GRAY)
    ret, img_thresh = cv2.threshold(img_gray,100,255, cv2.THRESH_BINARY | cv2.THRESH_OTSU)
    #cv2.imshow("img_thresh", img_thresh)

    pattern_features = []
    for _kp in kp:
        if circle_check1(img_thresh, _kp, rad) or circle_check2(img_thresh, _kp, rad):
            pattern_features.append(_kp)

    #print pattern_features

    for pf in pattern_features:
        cv2.circle(img, (int(pf.pt[0]), int(pf.pt[1])), 5, (0,0,255))

    cv2.imshow("pattern_features", img)


# main
for i in range(1,10):
    filename = "seq/20150805/cam%d/IND000%d%02d.bmp" % (cam_num+1, cam_num ,i)

    img = cv2.imread(filename)
    img_gray = cv2.imread(filename,0)

    # Initiate FAST object with default values
    fast = cv2.FastFeatureDetector(param_FAST)

    # find and draw the keypoints
    kp = fast.detect(img,None)
    img_corners = cv2.drawKeypoints(img, kp, color=(255,0,0))
    cv2.imshow("img_corners", img_corners)

    circular_sampling(img, kp, rad_circular_sampling)


    cv2.waitKey()