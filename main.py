__author__ = 'DongwonShin'

import numpy as np
import cv2
from matplotlib import pyplot as plt
import math

from logging import FileHandler
from vlogging import VisualRecord
import logging

# Global Parameters
cam_num = 5
seq_num = 10
param_FAST = 20
rad_circular_sampling = 10
unit_square_size = 250 # mm
horizontal_unit_num = 5
vertical_unit_num = 3
scale_factor = 0.6
imageSize = (1920, 1080)

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


def circular_sampling(src_img, kp, rad):
    img = src_img.copy()
    img_gray = cv2.cvtColor(img, cv2.cv.CV_RGB2GRAY)
    ret, img_thresh = cv2.threshold(img_gray,100,255, cv2.THRESH_BINARY | cv2.THRESH_OTSU)
    #cv2.imshow("img_thresh", img_thresh)

    pattern_features = []
    for _kp in kp:
        cx = _kp.pt[0]
        cy = _kp.pt[1]

        if cy + rad < img_thresh.shape[0] and cx + rad < img_thresh.shape[1] and cy - rad > 0 and cx - rad > 0:
            if circle_check1(img_thresh, _kp, rad) or circle_check2(img_thresh, _kp, rad):
                pattern_features.append(_kp)

    def distance(a,b):
        ax = a.pt[0]
        ay = a.pt[1]
        bx = b.pt[0]
        by = b.pt[1]

        return math.sqrt((ax-bx)**2 + (ay-by)**2)

    # non-maximum supression
    new_pattern_features = []
    for pf1 in pattern_features:
        flag = True
        for pf2 in new_pattern_features:
            if distance(pf1, pf2) < 100:
                flag = False
                break

        if flag:
            new_pattern_features.append(pf1)

    # print pattern_features
    # for pf in new_pattern_features:
    #     cv2.circle(img, (int(pf.pt[0]), int(pf.pt[1])), 5, (0,0,255), 3)
    # cv2.imshow("pattern_features", img)
    # cv2.waitKey()

    return new_pattern_features

def homography_estimation(src_img, pattern_features):
    #img = src_img.copy()
    img_gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)

    ideal_pattern_features = []
    pract_pattern_features = []

    for pf in pattern_features:
        pract_pattern_features.append(pf.pt)
    pract_pattern_features = np.array(pract_pattern_features)

    for j in range(0,horizontal_unit_num+1):
        for i in range(0,vertical_unit_num+1):
            ideal_pattern_features.append((unit_square_size * j, unit_square_size * i))
    ideal_pattern_features = np.array(ideal_pattern_features )
    ret_ideal_pattern_features = ideal_pattern_features

    ideal_pattern_cetner = np.mean(ideal_pattern_features, axis=0)
    pract_pattern_cetner = np.mean(pract_pattern_features, axis=0)

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
    dist = np.zeros((len(ideal_pattern_features),1))

    for ppf in pract_pattern_features:
        pract_pattern_corr.append(ppf)

        i=0
        for ipf in ideal_pattern_features:
            dist[i] = distance(ppf, ipf)
            i = i+1

        ideal_pattern_corr.append(ideal_pattern_features[np.argmin(dist)])


    ideal_pattern_corr = np.array(ideal_pattern_corr)
    pract_pattern_corr = np.array(pract_pattern_corr)
    H =  cv2.findHomography(ideal_pattern_corr, pract_pattern_corr)[0]

    ideal_pattern_features = np.append(ideal_pattern_features, np.ones((len(ideal_pattern_features),1)),axis=1)
    ideal_pattern_features = np.dot(H,ideal_pattern_features.T).T
    for i in range(len(ideal_pattern_features)):
        ideal_pattern_features[i] = ideal_pattern_features[i]/ideal_pattern_features[i][2]
    ideal_pattern_features = ideal_pattern_features[:,0:2]

    #ret, corners = cv2.findChessboardCorners(img_gray, (6,4),None)
    # corners = np.zeros(24)
    # i = 0
    # for k in ideal_pattern_corr:
    #     corners[i] = (list[( list([k[0], k[1]])]))
    #     #corners[i] = list(np.array(list(np.array(list([k[0], k[1]])))))

    # term = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_COUNT, 30, 0.1)
    # cv2.cornerSubPix(img_gray, ideal_pattern_corr, (11,11), (-1,-1), term)
    # cv2.drawChessboardCorners(img, (vertical_unit_num+1, horizontal_unit_num+1), ideal_pattern_corr, 1)

    # print pattern_features
    i=0
    for pf in ideal_pattern_features:
        cv2.circle(img, (int(pf[0]), int(pf[1])), 5, (0,255,0), 1)
        cv2.putText(img, str(i),(int(pf[0]), int(pf[1])),cv2.FONT_HERSHEY_SIMPLEX, 1, (0,255,0))
        i = i+1
    i=0
    # for pf in pract_pattern_features:
    #     cv2.circle(img, (int(pf[0]), int(pf[1])), 5, (255,0,0), 3)
    #     cv2.putText(img, str(i),(int(pf[0]), int(pf[1])),cv2.FONT_HERSHEY_SIMPLEX, 1, (255,0,0))
    #     i = i+1
    # cv2.imshow("pattern_features", img)

    return ideal_pattern_features, ret_ideal_pattern_features


# main
# logger = logging.getLogger("demo")
# fh = FileHandler('log.html',mode="w")
# logger.setLevel(logging.DEBUG)
# logger.addHandler(fh)
# #logger.debug(VisualRecord("img_corners", img_corners,"image corners", fmt="png"))

for j in range(5,cam_num+1):
    # object_points = np.zeros((seq_num*6*4, 3), dtype=np.float32)
    # image_points = np.zeros((seq_num*6*4, 2), dtype=np.float32)

    pattern_size = (4, 6)
    pattern_points = np.zeros( (24, 3), np.float32 )
    #pattern_points = pattern_points*250
    for a in range(0,6):
        for b in range(0,4):
            pattern_points[a*4+b][0] = a*250
            pattern_points[a*4+b][1] = b*250
            pattern_points[a*4+b][2] = 0

    object_points = []
    image_points = []

    for i in range(0,seq_num):
        filename = "seq/20150805/cam%d/IND000%d%02d.bmp" % (j, j-1, i)
        img = cv2.imread(filename)
        img_gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)

        # Initiate FAST object with default values
        fast = cv2.FastFeatureDetector(param_FAST)
        key_points = fast.detect(img,None)
        # img_corners = cv2.drawKeypoints(img, key_points, color=(255,0,0))
        # cv2.imshow("img_corners", img_corners)

        pattern_features = circular_sampling(img, key_points, rad_circular_sampling)
        result_pattern_features, ret = homography_estimation(img, pattern_features)

        #ret2, corners = cv2.findChessboardCorners(img_gray, (4,6))

        my_corners = []
        idx=0
        for rpf in result_pattern_features:
            #corner[0][0] = result_pattern_features[idx][0]
            #corner[0][1] = result_pattern_features[idx][1]
            #print corner
            #print [[result_pattern_features[idx][0], result_pattern_features[idx][1]]]
            #corner = [[rpf[0], rpf[1]]]
            my_corners.append([[rpf[0], rpf[1]]])
            idx = idx+1

        #print my_corners
        my_corners = np.array(my_corners, dtype=np.float32)

        criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 100, 0.001)
        cv2.cornerSubPix(img_gray, my_corners ,(10,10), (-1,-1), criteria)
        # img_gray = cv2.drawChessboardCorners(img_gray, (4,6), corners,ret)

        # idx = 0
        # for corner in corners:
        #     cv2.circle(img, (corner[0][0], corner[0][1]), 3, (0,0,255), 3)
        #     cv2.putText(img, str(idx),(corner[0][0], corner[0][1]),cv2.FONT_HERSHEY_SIMPLEX, 1, (0,0,255))
        #     idx = idx+1
        #     #print (corner[0][0], corner[0][1])

        #image_points[i*24:(i+1)*24,0:2] = result_pattern_features
        #object_points[i*24:(i+1)*24,0:2] = ret

        image_points.append(my_corners.reshape(-1, 2))
        object_points.append(pattern_points)

        cv2.imwrite("result/cam%d-%02d.bmp" % (j,i), img)
        #cv2.imwrite("result/cam%d-%02d_gray.bmp" % (j,i), img_gray)
        #cv2.waitKey()

    # camera_matrix = np.zeros((3,3),'float32')
    # camera_matrix[0,0]= 1736.0
    # camera_matrix[1,1]= 1736.0
    # camera_matrix[2,2]= 1.0
    #
    # camera_matrix[0,2]= 981.0
    # camera_matrix[1,2]= 526.0
    #
    # dist_coefs = np.zeros(4,'float32')

    #image_points = np.array(image_points)
    #object_points = np.array(object_points)

    #rms,camera_matrix,dist_coefs,rvecs,tvecs = cv2.calibrateCamera([object_points],[image_points], imageSize, camera_matrix,dist_coefs,flags=cv2.CALIB_USE_INTRINSIC_GUESS)
    rms,camera_matrix,dist_coefs,rvecs,tvecs = cv2.calibrateCamera(object_points,image_points,imageSize)
    rmat = cv2.Rodrigues(rvecs[9])[0]
    tvec = tvecs[9]

    print camera_matrix, rms

    fp=open("result/cam%d_param.txt" % j, 'w')
    for i in range(0,3):
        for j in range(0,3):
            fp.write(str(camera_matrix[i][j]))
            fp.write(" ")
        fp.write("\n")
    fp.write("\n")

    for i in range(0,3):
        for j in range(0,3):
            fp.write(str(rmat[i][j]))
            fp.write(" ")
        fp.write("\n")
    fp.write("\n")
    for j in range(0,3):
        fp.write(str(tvec[j][0]))
        fp.write(" ")
    fp.write("\n")