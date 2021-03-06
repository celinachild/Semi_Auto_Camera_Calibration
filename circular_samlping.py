__author__ = 'DongwonShin'

import cv2
import math

def circle_check1(img_thresh, _kp, rad, cir_num):

    flag = True

    cx = _kp.pt[0]
    cy = _kp.pt[1]

    lower_bound = 20
    upper_bound = 255-lower_bound

    for m in range(1,cir_num+1):
        for deg in range(30,60,10):
            ay = cy + (math.sin(math.radians(deg))*rad*m)
            ax = cx + (math.cos(math.radians(90-deg))*rad*m)
            by = cy - (math.sin(math.radians(deg))*rad*m)
            bx = cx - (math.cos(math.radians(90-deg))*rad*m)
            if ay < img_thresh.shape[0] and ax < img_thresh.shape[1] and by > 0 and bx > 0:
                 flag = (flag and (img_thresh[cy + (math.sin(math.radians(deg))*rad*m)][cx + (math.cos(math.radians(90-deg))*rad*m)] < lower_bound)
                              and (img_thresh[cy - (math.sin(math.radians(deg))*rad*m)][cx - (math.cos(math.radians(90-deg))*rad*m)] < lower_bound)
                              and (img_thresh[cy + (math.sin(math.radians(deg))*rad*m)][cx - (math.cos(math.radians(90-deg))*rad*m)] > upper_bound)
                              and (img_thresh[cy - (math.sin(math.radians(deg))*rad*m)][cx + (math.cos(math.radians(90-deg))*rad*m)] > upper_bound))
            else:
                flag = False

    return flag

def circle_check2(img_thresh, _kp, rad, cir_num):

    flag = True

    cx = _kp.pt[0]
    cy = _kp.pt[1]

    lower_bound = 20
    upper_bound = 255-lower_bound

    for m in range(1,cir_num+1):
        for deg in range(30,60,10):
            ay = cy + (math.sin(math.radians(deg))*rad*m)
            ax = cx + (math.cos(math.radians(90-deg))*rad*m)
            by = cy - (math.sin(math.radians(deg))*rad*m)
            bx = cx - (math.cos(math.radians(90-deg))*rad*m)
            if ay < img_thresh.shape[0] and ax < img_thresh.shape[1] and by > 0 and bx > 0:
                 flag = (flag and (img_thresh[cy + (math.sin(math.radians(deg))*rad*m)][cx + (math.cos(math.radians(90-deg))*rad*m)] > lower_bound)
                              and (img_thresh[cy - (math.sin(math.radians(deg))*rad*m)][cx - (math.cos(math.radians(90-deg))*rad*m)] > lower_bound)
                              and (img_thresh[cy + (math.sin(math.radians(deg))*rad*m)][cx - (math.cos(math.radians(90-deg))*rad*m)] < upper_bound)
                              and (img_thresh[cy - (math.sin(math.radians(deg))*rad*m)][cx + (math.cos(math.radians(90-deg))*rad*m)] < upper_bound))
            else:
                flag = False

    return flag


def circular_sampling(img, kp, rad, cir_num, non_maximum_thresh,cam_idx, seq_idx,circular_results,prefix):

    img_gray = cv2.cvtColor(img, cv2.cv.CV_RGB2GRAY)
    ret, img_thresh = cv2.threshold(img_gray,150,255, cv2.THRESH_BINARY | cv2.THRESH_OTSU)
    # cv2.adaptiveThreshold(img_gray,100,)
    # cv2.imshow("img_thresh", img_thresh)
    # cv2.waitKey()

    pattern_features = []
    for _kp in kp:
        cx = _kp.pt[0]
        cy = _kp.pt[1]

        if cy + rad < img_thresh.shape[0] and cx + rad < img_thresh.shape[1] and cy - rad > 0 and cx - rad > 0:
            if circle_check1(img_thresh, _kp, rad, cir_num):
                pattern_features.append(_kp)
            if circle_check2(img_thresh, _kp, rad, cir_num):
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

    if circular_results == True:
        tmp_img = img.copy()
        for pf in new_pattern_features:
           cv2.circle(tmp_img, (int(pf.pt[0]), int(pf.pt[1])), 5, (0,0,255), 3)
        # cv2.imshow("pattern_features", tmp_img)
        cv2.imwrite("../result/circular_sampling/%s_%d%02d.bmp" % (prefix,cam_idx,seq_idx), tmp_img)
        # cv2.waitKey()

    return new_pattern_features
