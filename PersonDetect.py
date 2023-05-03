import numpy as np
import cv2
import imutils
import pandas as pd
import time
import matplotlib.pyplot as plt

import poseModule

"""
Class Name  :   PersonDetectFromVideo
input       :   video file name
counts      :   frame numbers used to evaluate person's position
humanWidth  :   human Width in video frame


output      :   every person's position
"""
class PersonDetectFromVideo:
    def __init__(self, file, counts, humanWidth):
        self.file=file
        self.TEST_FRAME = counts
        self.PERSON_WIDTH=humanWidth

    def personPosition(self):
        # kernel
        kernel = np.ones((23, 23), np.uint8)
        # Use Background remover to split the foreground from background
        fgbg = cv2.createBackgroundSubtractorMOG2()
        # read video file
        videoCapture = cv2.VideoCapture(self.file)
        # CONSTANT
        HUMAN_HEIGHT = 50
        frame_count = 0
        person_position = []
        # data frame row number
        k = 0
        while True:
            success, frame = videoCapture.read()
            if success:
                # resize original video for better viewing performance
                frame = imutils.resize(frame, width=1000)
                # remove background
                deleted_background = fgbg.apply(frame)

                if frame_count < self.TEST_FRAME:
                    # dilation and erosion
                    opening_image = cv2.morphologyEx(deleted_background, cv2.MORPH_DILATE, kernel)
                    # cv2.imshow("deleted_background", opening_image)
                    # cv2.waitKey(1)
                    # detect moving anything
                    cnts, hierarchy = cv2.findContours(opening_image, cv2.RETR_TREE, cv2.CHAIN_APPROX_NONE)
                    result = frame.copy()
                    # detect moving anything with loop
                    for cnt in cnts:
                        x, y, w, h = cv2.boundingRect(cnt)
                        if (w > self.PERSON_WIDTH and h > HUMAN_HEIGHT):
                            person_position.append([x, y])
                            cv2.rectangle(result, (x, y), (x + w, y + h), (255, 0, 0), thickness=2)
                    frame_count += 1
                elif frame_count == self.TEST_FRAME:
                    # need to use classifying algorithm
                    xvalues = [start for start, end in person_position]  # person_position[:,0]
                    # remove zero values and sort and diff
                    xvalues = [x for x in xvalues if x != 0]
                    xvalues = np.msort(xvalues)
                    diff_x = np.diff(xvalues)
                    # find peak values
                    peak_inds = np.where(diff_x > self.PERSON_WIDTH-20)
                    # peak_inds=np.array(peak_inds)
                    person_num = len(peak_inds[0]) + 1
                    arrX = np.zeros([person_num, 1])
                    for i in range(person_num):
                        # average value
                        if i == 0:
                            arrX[i] = int(np.sum(xvalues[0:peak_inds[0][0]]) / peak_inds[0][0])
                        elif i == person_num - 1:
                            arrX[i] = int(np.sum(xvalues[peak_inds[0][i - 1]:len(xvalues)]) / (
                                    len(xvalues) - peak_inds[0][i - 1]))
                        else:
                            arrX[i] = int(np.sum(xvalues[peak_inds[0][i - 1]:peak_inds[0][i]]) / (
                                    peak_inds[0][i] - peak_inds[0][i - 1]))
                    break
                # cv2.imshow("Image", result)
                # cv2.waitKey(1)
        videoCapture.release()
        self.person_num = person_num
        return arrX, person_num
    def poseDetections(self, arrX, person_num):
        # read video file
        videoCapture = cv2.VideoCapture(self.file)
        MARGIN = 100
        detector = [None] * (person_num)
        arrDf = [None] * (person_num)
        pTime = time.time()
        arrTime = [[0 for x in range(0)] for y in range(person_num)]
        # initialize the PoseModule object
        for i in range(person_num):
            detector[i] = poseModule.PoseDetector()
            arrDf[i] = pd.DataFrame(columns=['KeyPoint' + str(i) for i in range(17)])

        k = 0
        while True:
            success, frame = videoCapture.read()
            if success:
                frame = imutils.resize(frame, width=1000)
                for i in range(person_num):
                    # Split frame according human position
                    if i == 0:
                        subFrame = frame[:, 0:(int)(arrX[i] + self.PERSON_WIDTH + MARGIN)]
                    else:
                        subFrame = frame[:, (int)(arrX[i - 1] + self.PERSON_WIDTH):(int)(arrX[i] + self.PERSON_WIDTH + MARGIN)]

                    img, lmList = detector[i].findPose(subFrame)
                    newList = []

                    if len(lmList) != 0:
                        for j in range(len(lmList)):
                            if j in (0, 2, 5, 7, 8, 11, 12, 13, 14, 15, 16,23, 24, 25, 26, 27, 28): #
                                newList.append(lmList[j])
                        arrTime[i].append(time.time() - pTime)
                        arrDf[i].loc[k] = newList
                    lmList = []
                k += 1
                cv2.imshow("Image", frame)
                cv2.waitKey(1)
            else:
                break
        videoCapture.release()
        person_key_points = {}
        for i in range(person_num):
            # arrDf[i].to_csv('_' + str(i) + '_' + 'key_points.csv', sep=',', encoding='utf-8', index=False)
            person_key_points["Musician" + str(i + 1)] = arrDf[i]
        self.person_key_points = person_key_points
        self.arrTime = arrTime

    def opticalFlowPose(self, draw=True):
        person_key_points_np = {}
        person_num = len(self.person_key_points)
        for musician in self.person_key_points:
            person_key_points_np[musician] = {}
            for keys in self.person_key_points[musician]:
                person_key_points_np[musician][keys] = np.array(
                    [np.array(xi) for xi in self.person_key_points[musician][keys]])

        xy_pose_diff = {}
        for musician in self.person_key_points:
            xy_pose_diff[musician] = {}
            for keys in self.person_key_points[musician]:
                xy_pose_diff[musician][keys] = np.diff(person_key_points_np[musician][keys][:, 0:2], axis=0)

        pose_diff_magnitudes = {}
        for musician in self.person_key_points:
            pose_diff_magnitudes[musician] = {}
            for keys in self.person_key_points[musician]:
                pose_diff_magnitudes[musician][keys] = np.sqrt(
                    np.square(xy_pose_diff[musician][keys][:, 0]) + np.square(xy_pose_diff[musician][keys][:, 1]))

        avg_pose_key_points = {}
        yp = [None] * person_num
        idx=0
        for musician in self.person_key_points:
            sum_keys = np.zeros(pose_diff_magnitudes[musician]['KeyPoint0'].shape)
            for keys in self.person_key_points[musician]:
                sum_keys += pose_diff_magnitudes[musician][keys]
            avg_pose_key_points[musician] = sum_keys / 17
            yp[idx] = avg_pose_key_points[musician]
            idx+=1
        if draw:
            legends = list(avg_pose_key_points.keys())
            for a_key in avg_pose_key_points.keys():
                plt.plot(avg_pose_key_points[a_key][2:-1], linewidth=3.0)
            plt.legend(legends, fontsize=8)
            plt.xlabel('Frames', fontsize=16)
            plt.ylabel('Average Motion', fontsize=16)
            plt.title('Average Motion - Frame per Musician', fontsize=16)
            plt.show(block=False)
        return yp, self.arrTime