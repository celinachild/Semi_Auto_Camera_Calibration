__author__ = 'DongwonShin'

import numpy as np
import cv2
from matplotlib import pyplot as plt
import math

# Global Parameters
seq_id = 5
corner_detect_mode = 'FAST' # FAST, ORB
param_FAST = 20
#imageSize = (1920,1080); scale_factor = 0.6; rad_circular_sampling = 20; non_maximum_thresh=50
imageSize = (1280,720); scale_factor = 0.4; rad_circular_sampling = 10; non_maximum_thresh=50
cir_num = 3
draw_mode = {'corner_detect':True, 'my_corners':False, 'Final':True}

cam_num = 5
seq_num = 10
unit_square_size = 250 # mm
horizontal_unit_num = 5
vertical_unit_num = 3
input_file_directory = ""
file_name=""


def sequence_setting(seq):
    global input_file_directory
    global file_name
    if (seq == 1):
        input_file_directory = "seq/20141106"
        file_name = "%s/cam%d/IND000%d%d.bmp" % (input_file_directory, cam_idx, cam_idx-1, seq_idx)
    elif (seq == 2):
        input_file_directory = "seq/20150129"
        file_name = "%s/cam%d/IND000%d%d.bmp" % (input_file_directory, cam_idx, cam_idx-1, seq_idx)
    elif (seq == 3):
        input_file_directory = "seq/20150131"
        file_name = "%s/cam%d/IND000%d%d.bmp" % (input_file_directory, cam_idx, cam_idx-1, seq_idx)
    elif (seq == 4):
        input_file_directory = "seq/20150805"
        file_name = "%s/cam%d/IND000%d%02d.bmp" % (input_file_directory, cam_idx, cam_idx-1, seq_idx)
    elif (seq == 5):
        input_file_directory = "seq/20150805_scale"
        file_name = "%s/cam%d/IND000%d%02d.bmp" % (input_file_directory, cam_idx, cam_idx-1, seq_idx)

def circle_check1(img_thresh, _kp, rad):

    flag = True

    cx = _kp.pt[0]
    cy = _kp.pt[1]

    lower_bound = 20
    upper_bound = 255-lower_bound

    for m in range(1,cir_num+1):
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

    for m in range(1,cir_num+1):
        for k in range(3,7):
            if cy + rad < img_thresh.shape[0] and cx + rad < img_thresh.shape[1] and cy - rad > 0 and cx - rad > 0:
                 flag = (flag and (img_thresh[cy + (math.sin(math.pi/k)*rad*m)][cx + (math.cos(math.pi/2 - math.pi/k)*rad*m)] > lower_bound)
                              and (img_thresh[cy - (math.sin(math.pi/k)*rad*m)][cx - (math.cos(math.pi/2 - math.pi/k)*rad*m)] > lower_bound)
                              and (img_thresh[cy + (math.sin(math.pi/k)*rad*m)][cx - (math.cos(math.pi/2 - math.pi/k)*rad*m)] < upper_bound)
                              and (img_thresh[cy - (math.sin(math.pi/k)*rad*m)][cx + (math.cos(math.pi/2 - math.pi/k)*rad*m)] < upper_bound))

    return flag


def circular_sampling(src_img, kp, rad):

    img_gray = cv2.cvtColor(img, cv2.cv.CV_RGB2GRAY)
    ret, img_thresh = cv2.threshold(img_gray,100,255, cv2.THRESH_BINARY | cv2.THRESH_OTSU)
    # cv2.imshow("img_thresh", img_thresh)
    # cv2.waitKey()

    pattern_features = []
    for _kp in kp:
        cx = _kp.pt[0]
        cy = _kp.pt[1]

        if cy + rad < img_thresh.shape[0] and cx + rad < img_thresh.shape[1] and cy - rad > 0 and cx - rad > 0:
            if circle_check1(img_thresh, _kp, rad):
                pattern_features.append(_kp)
            if circle_check2(img_thresh, _kp, rad):
                pattern_features.append(_kp)

    def distance(a,b):
        ax = a.pt[0]
        ay = a.pt[1]
        bx = b.pt[0]
        by = b.pt[1]

        return math.sqrt((ax-bx)**2 + (ay-by)**2)

    # non-maximum suppression
    new_pattern_features = []
    for pf1 in pattern_features:
        flag = True
        for pf2 in new_pattern_features:
            if distance(pf1, pf2) < non_maximum_thresh:
                flag = False
                break

        if flag:
            new_pattern_features.append(pf1)

    # for pf in new_pattern_features:
    #     cv2.circle(img, (int(pf.pt[0]), int(pf.pt[1])), 5, (0,0,255), 3)
    # cv2.imshow("pattern_features", img)
    # cv2.waitKey()

    return new_pattern_features

def homography_estimation(src_img, pattern_features):

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

    draw_points(ideal_pattern_corr, (0,0,255))
    draw_points(pract_pattern_corr, (0,255,0))

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


def draw_points(points, color):
    i = 0
    for p in points:
        cv2.circle(img, (int(p[0]), int(p[1])), 5, color, 1)
        cv2.putText(img, str(i), (int(p[0]), int(p[1])), cv2.FONT_HERSHEY_SIMPLEX, 1, color)
        i = i + 1


# main
rms_dict = dict()

for cam_idx in range(1,cam_num+1):

    pattern_size = (4, 6)
    pattern_points = np.zeros( (24, 3), np.float32 )

    for a in range(0,6):
        for b in range(0,4):
            pattern_points[a*4+b][0] = a*250
            pattern_points[a*4+b][1] = b*250
            pattern_points[a*4+b][2] = 0

    object_points = []
    image_points = []

    for seq_idx in range(0,seq_num):
        sequence_setting(seq_id)

        img = cv2.imread(file_name)
        img_gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)

        if (corner_detect_mode == 'FAST'):
            fast = cv2.FastFeatureDetector(param_FAST)
            kp = fast.detect(img,None)
            if draw_mode['corner_detect']:
                img_corners = cv2.drawKeypoints(img, kp, color=(255,0,0))
                cv2.imwrite('result/fast_result/'+str(cam_idx)+str(seq_idx)+'.png',img_corners)
        elif (corner_detect_mode == 'ORB'):
            orb = cv2.ORB()
            kp = orb.detect(img, None)
            kp, des = orb.compute(img, kp)
            if draw_mode['corner_detect']:
                img_corners = cv2.drawKeypoints(img,kp,color=(0,255,0), flags=0)
                cv2.imwrite('result/orb_result/'+str(cam_idx)+str(seq_idx)+'.png',img_corners)

        pattern_features = circular_sampling(img, kp, rad_circular_sampling)
        result_pattern_features, ret = homography_estimation(img, pattern_features)

        my_corners = []
        idx=0
        for rpf in result_pattern_features:
            my_corners.append([[rpf[0], rpf[1]]])
            idx = idx+1

        #print my_corners
        my_corners = np.array(my_corners, dtype=np.float32)

        criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 100, 0.001)
        cv2.cornerSubPix(img_gray, my_corners ,(10,10), (-1,-1), criteria)

        if draw_mode['my_corners']:
            i=0
            for corner in my_corners:
                cv2.circle(img, (int(corner[0][0]), int(corner[0][1])), 5, (0,0,255), 3)
                cv2.putText(img, str(i),(int(corner[0][0]), int(corner[0][1])),cv2.FONT_HERSHEY_SIMPLEX, 1, (0,0,255))
                i = i+1

        image_points.append(my_corners.reshape(-1, 2))
        object_points.append(pattern_points)

        if draw_mode['Final']:
            cv2.imwrite("result/final_result/cam%d-%02d.bmp" % (cam_idx,seq_idx), img)

    rms,camera_matrix,dist_coefs,rvecs,tvecs = cv2.calibrateCamera(object_points,image_points,imageSize)
    rmat = cv2.Rodrigues(rvecs[seq_num-1])[0]  # must use the last sequence's rotation mat
    tvec = tvecs[seq_num-1]                    # must use the last sequence's translation vec

    print "root-mean-square for cam param %d = %f" % (cam_idx, rms)
    rms_dict[cam_idx] = rms

    # cam param save
    fp=open("result/cam_param/cam%d_param.txt" % cam_idx, 'w')
    np.savetxt(fp, camera_matrix, fmt='%f'); fp.write("\n")
    np.savetxt(fp, rmat, fmt='%f'); fp.write("\n")
    np.savetxt(fp, tvec, fmt='%f')
    fp.close()

# rms log
fp_rms = open("result/cam_param/rms_result.txt", 'w')
fp_rms.write("seq_id : %d\n" % seq_id)
fp_rms.write("corner_detect_mode : %s\n" % corner_detect_mode)
fp_rms.write("image size : (%d, %d)\n" % (imageSize[0],imageSize[1]))
fp_rms.write("scale factor : %f\n" % scale_factor)
fp_rms.write("radius for circular sampling : %d\n\n" % rad_circular_sampling)
for rms in rms_dict:
    fp_rms.write("root-mean-square for cam param %d = %f\n" % (rms, rms_dict[rms]))
fp_rms.close()