__author__ = 'DongwonShin'

import numpy as np
import cv2
from matplotlib import pyplot as plt

def circular_sampling(img, kp):
    img_gray = cv2.cvtColor(img, cv2.cv.CV_RGB2GRAY)
    ret, thre_img = cv2.threshold(img_gray,100,255, cv2.THRESH_BINARY | cv2.THRESH_OTSU)
    #cv2.imshow("img_thresh", thre_img)

    for k in kp:
        print k.pt


for i in range(1,5):
    filename = "seq/20150805/cam1/IND0000%02d.bmp" % i

    img = cv2.imread(filename)
    img_gray = cv2.imread(filename,0)

    # Initiate FAST object with default values
    fast = cv2.FastFeatureDetector(40,True)

    # find and draw the keypoints
    kp = fast.detect(img,None)
    img_corners = cv2.drawKeypoints(img, kp, color=(255,0,0))

    circular_sampling(img, kp)

    cv2.imshow("img_corners", img_corners)
    cv2.waitKey()