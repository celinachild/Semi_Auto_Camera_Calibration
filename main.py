__author__ = 'DongwonShin'

from circular_samlping import *
from homography_estimation import *
import os
import ConfigParser
import shutil

# Global Parameters
draw_mode = {}
axis_change_flag = True
idx_rearrange = [20,21,22,23,16,17,18,19,12,13,14,15,8,9,10,11,4,5,6,7,0,1,2,3]     # coordinates for GIGA project 3*5 pattern
input_file_directory = ""
file_name=""

def color_sequence_setting(seq):
    global input_file_directory
    global file_name

    input_file_directory = "../seq/%d" % seq_id
    file_name = "%s/cam%d/%s%d_%d.%s" % (input_file_directory, cam_idx, color_prefix, cam_idx, seq_idx, color_file_ext)
    # print file_name
    # file_name = "%s/cam%d/%s_%d.%s" % (input_file_directory, cam_idx, color_prefix, seq_idx, file_ext)

def tof_sequence_setting(seq):
    global input_file_directory
    global file_name

    input_file_directory = "../seq/%d" % seq_id
    file_name = "%s/tof%d/%s%d_C%d.%s" % (input_file_directory, cam_idx, depth_prefix,cam_idx, seq_idx, depth_file_ext)
    # print file_name

def Calibration_For_Color_Cam():
    global param_FAST, scale_factor, rad_circular_sampling, non_maximum_thresh, cir_num, cam_num, imageSize, cam_idx, object_points, image_points, seq_idx, img, img_gray, fast, kp, img_corners, pattern_features, result_pattern_features, H, ret, my_corners, idx, rpf, criteria, i, corner, rms, camera_matrix, dist_coefs, rvecs, tvecs, rmat, tvec, fp, fp_rms

    param_FAST = config.getint('Parameters_for_color_cams', 'param_FAST')
    scale_factor = config.getfloat('Parameters_for_color_cams', 'scale_factor')
    rad_circular_sampling = config.getint('Parameters_for_color_cams', 'rad_circular_sampling')
    non_maximum_thresh = config.getint('Parameters_for_color_cams', 'non_maximum_thresh')
    cir_num = config.getint('Parameters_for_color_cams', 'cir_num')
    cam_num = config.getint('Parameters_for_color_cams', 'cam_num')

    rms_dict = dict()

    for cam_idx in range(cam_start_number, cam_start_number+cam_num):

        object_points = []
        image_points = []
        homographys = []

        for seq_idx in range(seq_start_number, seq_num):

            if missing_flag == True:
                continue_flag = False
                for mising_idx in missing_idxs:
                    if (seq_idx == mising_idx):
                        continue_flag = True
                        break
                if continue_flag == True:
                    continue

            color_sequence_setting(seq_id)

            img = cv2.imread(file_name)
            img_gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)

            fast = cv2.FastFeatureDetector(param_FAST,True)

            kp = fast.detect(img, None)
            print len(kp)
            if draw_mode['fast_results']:
                img_corners = cv2.drawKeypoints(img, kp, color=(0, 0, 255))
                cv2.imwrite('../result/fast_result/color_%d%02d.bmp' % (cam_idx, seq_idx), img_corners)

            pattern_features = circular_sampling(img, kp, rad_circular_sampling, cir_num, non_maximum_thresh, cam_idx,
                                                 seq_idx, draw_mode['circular_results'], 'color')
            # print len(pattern_features)
            result_pattern_features, H, ret = homography_estimation(img, pattern_features, horizontal_unit_num,
                                                                    vertical_unit_num, unit_square_size, scale_factor)

            homographys.append(H)

            my_corners = []
            idx = 0
            for rpf in result_pattern_features:
                my_corners.append([[rpf[0], rpf[1]]])
                idx = idx + 1

            # print my_corners
            my_corners = np.array(my_corners, dtype=np.float32)

            criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 100, 0.001)
            cv2.cornerSubPix(img_gray, my_corners, (15, 15), (-1, -1), criteria)

            # axis change
            if axis_change_flag == True:
                my_corners = my_corners[idx_rearrange]

            image_points.append(my_corners.reshape(-1, 2))
            object_points.append(pattern_points)

            if draw_mode['final_results']:
                for i, corner in enumerate(my_corners):
                    cv2.circle(img, (int(corner[0][0]), int(corner[0][1])), 5, (0, 0, 255), 3)
                    cv2.putText(img, str(i), (int(corner[0][0]), int(corner[0][1])), cv2.FONT_HERSHEY_SIMPLEX, 1,
                                (0, 0, 255))

                cv2.imwrite("../result/final_result/color_%d%02d.bmp" % (cam_idx, seq_idx), img)

            # fp = open("../result/object_points/color_%d_%d.txt" % (cam_idx, seq_idx), 'w')
            # for my_corner in my_corners:
            #     fp.write('%f\t%f' % (my_corner[0][0],my_corner[0][1]))
            #     fp.write('\n')
            # fp.close()


        rms, camera_matrix, dist_coefs, rvecs, tvecs = cv2.calibrateCamera(object_points, image_points, (img.shape[1],img.shape[0]))
        rmat = cv2.Rodrigues(rvecs[len(rvecs)-1])[0]  # must use the last sequence's rotation mat
        tvec = tvecs[len(rvecs)-1]  # must use the last sequence's translation vec

        if rms > 0.5:
            print "root-mean-square for color param %d = %f <---- please check" % (cam_idx, rms)
        else:
            print "root-mean-square for color param %d = %f" % (cam_idx, rms)
        rms_dict[cam_idx] = rms

        # cam param save
        fp = open("../result/cam_param/cam%d_param.txt" % cam_idx, 'w')
        np.savetxt(fp, camera_matrix, fmt='%.11f');
        fp.write("\n")
        np.savetxt(fp, rmat, fmt='%.11f');
        fp.write("\n")
        np.savetxt(fp, tvec, fmt='%.11f')
        fp.close()

        # cam param integration
        fp = open("../result/cam_param/cam_param.txt", 'a')
        fp.write("param_cam%d\n" % (cam_idx+1))
        np.savetxt(fp, camera_matrix, fmt='%.11f');
        fp.write("\n")
        np.savetxt(fp, rmat, fmt='%.11f');
        fp.write("\n")
        np.savetxt(fp, tvec, fmt='%.11f')
        fp.write("\n")
        fp.close()

    # result log
    fp_rms = open("../result/cam_param/color_rms_result.txt", 'w')
    fp_rms.write("seq_id : %d\n" % seq_id)
    fp_rms.write("image size : (%d, %d)\n" % (img.shape[1], img.shape[0]))
    fp_rms.write("scale factor : %f\n" % scale_factor)
    fp_rms.write("radius for circular sampling : %d\n\n" % rad_circular_sampling)
    for rms in rms_dict:
        fp_rms.write("root-mean-square for cam param %d = %f\n" % (rms, rms_dict[rms]))
    fp_rms.close()


def Calibration_For_Tof_Cam():
    global param_FAST, scale_factor, rad_circular_sampling, non_maximum_thresh, cir_num, cam_num, cam_idx, object_points, image_points, seq_idx, img, img_gray, fast, kp, img_corners, pattern_features, result_pattern_features, H, ret, my_corners, idx, rpf, criteria, i, corner, rms, camera_matrix, dist_coefs, rvecs, tvecs, rmat, tvec, fp, fp_rms

    param_FAST = config.getint('Parameters_for_tof_cams', 'param_FAST')
    scale_factor = config.getfloat('Parameters_for_tof_cams', 'scale_factor')
    rad_circular_sampling = config.getint('Parameters_for_tof_cams', 'rad_circular_sampling')
    non_maximum_thresh = config.getint('Parameters_for_tof_cams', 'non_maximum_thresh')
    cir_num = config.getint('Parameters_for_tof_cams', 'cir_num')
    cam_num = config.getint('Parameters_for_tof_cams', 'cam_num')

    rms_dict = dict()
    for cam_idx in range(cam_start_number, cam_start_number+cam_num):

        object_points = []
        image_points = []

        for seq_idx in range(seq_start_number, seq_num):

            if missing_flag == True:
                continue_flag = False
                for mising_idx in missing_idxs:
                    if (seq_idx == mising_idx):
                        continue_flag = True
                        break
                if continue_flag == True:
                    continue

            tof_sequence_setting(seq_id)

            img = cv2.imread(file_name)
            img_gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
            img_thre = cv2.adaptiveThreshold(img_gray,255,cv2.ADAPTIVE_THRESH_MEAN_C, cv2.THRESH_BINARY,11,2)
            # ret, img_thre = cv2.threshold(img_gray,128,255,cv2.THRESH_BINARY+cv2.THRESH_OTSU)
            # img = cv2.cvtColor(img_thre, cv2.COLOR_GRAY2BGR)
            # edges = cv2.Canny(img,100,200)
            # edges = inverte(edges)
            # img = img_thre + edges

            # cv2.imshow("img_thre", img_thre)
            # cv2.waitKey()

            fast = cv2.FastFeatureDetector(param_FAST)
            kp = fast.detect(img, None)
            if draw_mode['fast_results']:
                img_corners = cv2.drawKeypoints(img, kp, color=(255, 0, 0))
                cv2.imwrite('../result/fast_result/tof_%d%02d.bmp' % (cam_idx, seq_idx), img_corners)

            pattern_features = circular_sampling(img, kp, rad_circular_sampling, cir_num, non_maximum_thresh, cam_idx,
                                                 seq_idx, draw_mode['circular_results'], "tof")
            # print len(pattern_features)
            result_pattern_features, H, ret = homography_estimation(img, pattern_features, horizontal_unit_num,
                                                                    vertical_unit_num, unit_square_size, scale_factor)

            my_corners = []
            idx = 0
            for rpf in result_pattern_features:
                my_corners.append([[rpf[0], rpf[1]]])
                idx = idx + 1

            # print my_corners
            my_corners = np.array(my_corners, dtype=np.float32)

            criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001)
            cv2.cornerSubPix(img_gray, my_corners, (5, 5), (-1, -1), criteria)

            # axis change
            if axis_change_flag == True:
                my_corners = my_corners[idx_rearrange]

            image_points.append(my_corners.reshape(-1, 2))
            object_points.append(pattern_points)

            if draw_mode['final_results']:

                for i, corner in enumerate(my_corners):
                    cv2.circle(img, (int(corner[0][0]), int(corner[0][1])), 5, (255, 0, 0), 1)

                cv2.imwrite("../result/final_result/tof_%d%02d.bmp" % (cam_idx, seq_idx), img)

            # fp = open("../result/object_points/tof_%d_%d.txt" % (cam_idx, seq_idx), 'w')
            # for my_corner in my_corners:
            #     fp.write('%f\t%f' % (my_corner[0][0],my_corner[0][1]))
            #     fp.write('\n')
            # fp.close()


        rms, camera_matrix, dist_coefs, rvecs, tvecs = cv2.calibrateCamera(object_points, image_points, (img.shape[1],img.shape[0]))
        rmat = cv2.Rodrigues(rvecs[len(rvecs)-1])[0]  # must use the last sequence's rotation mat
        tvec = tvecs[len(rvecs)-1]  # must use the last sequence's translation vec

        fp = open("../result/cam_param/tof%d_distortion_param.txt" % cam_idx, 'w')
        np.savetxt(fp, dist_coefs, fmt='%.11f');
        fp.write("\n")
        fp.close()

        if draw_mode['undistorted_results']:
            for seq_idx in range(0, seq_num):
                tof_sequence_setting(seq_id)
                img = cv2.imread(file_name)
                undistort_img = img.copy()
                cv2.undistort(img, camera_matrix, dist_coefs, undistort_img)
                cv2.imwrite("../result/undistort/tof_%d%d.bmp" % (cam_idx, seq_idx), undistort_img)


        if rms > 0.5:
            print "root-mean-square for tof param %d = %f <---- please check" % (cam_idx, rms)
        else:
            print "root-mean-square for tof param %d = %f" % (cam_idx, rms)
        rms_dict[cam_idx] = rms

        # cam param save
        fp = open("../result/cam_param/tof%d_param.txt" % cam_idx, 'w')
        np.savetxt(fp, camera_matrix, fmt='%.11f')
        fp.write("\n")
        np.savetxt(fp, rmat, fmt='%.11f')
        fp.write("\n")
        np.savetxt(fp, tvec, fmt='%.11f')
        fp.write("\n")
        fp.write("%.11f %.11f %.11f" % (dist_coefs[0][0],dist_coefs[0][1],dist_coefs[0][4]))
        fp.write("\n")
        fp.close()

        # cam param integration
        fp = open("../result/cam_param/tof_param.txt", 'a')
        fp.write("param_tof%d\n" % (cam_idx+1))
        np.savetxt(fp, camera_matrix, fmt='%.11f');
        fp.write("\n")
        np.savetxt(fp, rmat, fmt='%.11f');
        fp.write("\n")
        np.savetxt(fp, tvec, fmt='%.11f')
        fp.write("\n")
        fp.write("%.11f %.11f %.11f" % (dist_coefs[0][0],dist_coefs[0][1],dist_coefs[0][4]))
        fp.write("\n\n")
        fp.close()


    # result log
    fp_rms = open("../result/cam_param/tof_rms_result.txt", 'w')
    fp_rms.write("seq_id : %d\n" % seq_id)
    fp_rms.write("image size : (%d, %d)\n" % (img.shape[1], img.shape[0]))
    fp_rms.write("scale factor : %f\n" % scale_factor)
    fp_rms.write("radius for circular sampling : %d\n\n" % rad_circular_sampling)
    for rms in rms_dict:
        fp_rms.write("root-mean-square for cam param %d = %f\n" % (rms, rms_dict[rms]))
    fp_rms.close()


def Make_Ideal_Pattern():
    global pattern_points
    pattern_points = np.zeros((hp_num * vp_num, 3), np.float32)
    for a in range(0, hp_num):
        for b in range(0, vp_num):
            pattern_points[a * vp_num + b][0] = a * unit_square_size
            pattern_points[a * vp_num + b][1] = b * unit_square_size
            pattern_points[a * vp_num + b][2] = 0
    if axis_change_flag == True:
        pattern_points = pattern_points[idx_rearrange]
    temp = np.copy(pattern_points[:, 0])
    pattern_points[:, 0] = pattern_points[:, 1]
    pattern_points[:, 1] = temp


def Dictionary_Setting():

    if os.access('../result/cam_param', os.F_OK):
        shutil.rmtree('../result/cam_param')
    if os.access('../result/circular_sampling', os.F_OK):
        shutil.rmtree('../result/circular_sampling')
    if os.access('../result/fast_result', os.F_OK):
        shutil.rmtree('../result/fast_result')
    if os.access('../result/final_result', os.F_OK):
        shutil.rmtree('../result/final_result')
    if os.access('../result/undistort', os.F_OK):
        shutil.rmtree('../result/undistort')

    if not os.access('../result', os.F_OK):
        os.mkdir('../result')
    if not os.access('../result/cam_param', os.F_OK):
        os.mkdir('../result/cam_param')
    if not os.access('../result/fast_result', os.F_OK):
        os.mkdir('../result/fast_result')
    if not os.access('../result/final_result', os.F_OK):
        os.mkdir('../result/final_result')
    if not os.access('../result/undistort', os.F_OK):
        os.mkdir('../result/undistort')
    if not os.access('../result/circular_sampling', os.F_OK):
        os.mkdir('../result/circular_sampling')


def Read_From_Configure_File():
    global config, seq_id, seq_num, color_prefix, depth_prefix, cam_start_number, seq_start_number, missing_flag, missing_idxs, \
        color_file_ext, depth_file_ext, unit_square_size, horizontal_unit_num, vertical_unit_num, hp_num, vp_num
    config = ConfigParser.RawConfigParser()
    config.read('../seq/config.cfg')
    seq_id = config.getint('Sequence_info', 'seq_id')
    seq_num = config.getint('Sequence_info', 'seq_num')
    color_prefix = config.get('Sequence_info','color_prefix')
    depth_prefix = config.get('Sequence_info','depth_prefix')
    color_file_ext = config.get('Sequence_info','color_file_ext')
    depth_file_ext = config.get('Sequence_info','depth_file_ext')
    cam_start_number = config.getint('Sequence_info', 'cam_start_number')
    seq_start_number = config.getint('Sequence_info', 'seq_start_number')
    missing_flag = config.getboolean('Sequence_info', 'missing_flag')
    if missing_flag == True:
        missing_idxs_str = config.get('Sequence_info', 'missing_idxs')
        missing_idxs_str = missing_idxs_str.split(',')
        missing_idxs = []
        for i in range(len(missing_idxs_str)):
            missing_idxs.append(int(missing_idxs_str[i]))
    else:
        missing_idxs = []

    unit_square_size = config.getint('Calibration_Pattern_info', 'unit_square_size')
    horizontal_unit_num = config.getint('Calibration_Pattern_info', 'horizontal_unit_num')
    vertical_unit_num = config.getint('Calibration_Pattern_info', 'vertical_unit_num')
    hp_num = horizontal_unit_num + 1
    vp_num = vertical_unit_num + 1
    draw_mode['fast_results'] = config.getboolean('Draw_mode', 'fast_results')
    draw_mode['final_results'] = config.getboolean('Draw_mode', 'final_results')
    draw_mode['undistorted_results'] = config.getboolean('Draw_mode', 'undistorted_results')
    draw_mode['circular_results'] = config.getboolean('Draw_mode', 'circular_results')

# main
if __name__ == "__main__":

    Read_From_Configure_File()
    Dictionary_Setting()
    Make_Ideal_Pattern()

    Calibration_For_Color_Cam()
    Calibration_For_Tof_Cam()