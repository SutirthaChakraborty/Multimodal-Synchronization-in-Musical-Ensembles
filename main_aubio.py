# import librosa
import cv2
import time
import pandas as pd
import imutils
import numpy as np
import matplotlib.pyplot as plt

# from ffpyplayer.player import MediaPlayer
from scipy.signal import savgol_filter
from scipy.signal import find_peaks
from moviepy.editor import *
import aubio
import pyaudio
import wave
import sys


import poseModule
import PersonDetect

import soundcard as sc

# import winsound
duration = 100  # milliseconds
freq = 2080  # Hz
win_s = 1024
hop_s = win_s // 2


import subprocess

def mac_beep(freq, duration):
    script = f'''
    set volume 1
    set sound_path to (POSIX path of (path to library folder from user domain)) & "Sounds:Submarine.aiff"
    do shell script "afplay " & quoted form of sound_path & " -r " & quoted form of (("{0}" as integer) as string) & " -t " & quoted form of (("{1}" as integer) as string)
    '''
    script = script.format(freq, duration / 1000)
    subprocess.run(["osascript", "-e", script])

# Usage example:
freq = 440
duration = 1000
# mac_beep(freq, duration)


default_speaker = sc.default_speaker()

def shift5(arr, num, fill_value=np.nan):
    result = np.empty_like(arr)
    if num > 0:
        result[:num] = fill_value
        result[num:] = arr[:-num]
    elif num < 0:
        result[num:] = fill_value
        result[:num] = arr[-num:]
    else:
        result[:] = arr
    return result

def opticalFlow(person_key_points, draw=True):
    person_key_points_np = {}
    person_num = len(person_key_points)
    for musician in person_key_points:
        person_key_points_np[musician] = {}
        for keys in person_key_points[musician]:
            person_key_points_np[musician][keys] = np.array(
                [np.array(xi) for xi in person_key_points[musician][keys]])

    xy_pose_diff = {}
    for musician in person_key_points:
        xy_pose_diff[musician] = {}
        for keys in person_key_points[musician]:
            xy_pose_diff[musician][keys] = np.diff(person_key_points_np[musician][keys][:, 0:2], axis=0)

    pose_diff_magnitudes = {}
    for musician in person_key_points:
        pose_diff_magnitudes[musician] = {}
        for keys in person_key_points[musician]:
            pose_diff_magnitudes[musician][keys] = np.sqrt(
                np.square(xy_pose_diff[musician][keys][:, 0]) + np.square(xy_pose_diff[musician][keys][:, 1]))

    avg_pose_key_points = {}
    yp = [None] * person_num
    idx=0
    for musician in person_key_points:
        sum_keys = np.zeros(pose_diff_magnitudes[musician]['KeyPoint0'].shape)
        for keys in person_key_points[musician]:
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
        plt.draw()
        plt.show(block=False)
        plt.pause(0.01)
        plt.clf()
    return yp, arrTime
# input:
#   filename:   Read file or if value is 'live' then read from camera   [compulsory]
#   function:   "Kuramoto" or "swarmlator" or "janus" or "flock",  # [compulsory]
#   videoFlag:  if false then sync audio only   [compulsory]
#   output:     'sonicpi/tap',  ==> either it will produce tap or beat sound or send to sonic py   [compulsory]
#   ip:         'localhost',    ==> if sonicpy selected, then the ip address [optional]
#   port:       '1234'		    ==> if sonicpy selected, then the port number  [optional]
#   writeoutput:  False or True   ==> The time stamp at which the tap was produced to be written in a file
#   at completion of execution   [optional]
def CallRobot(filename, function, videoFlag, output, ip, writeoutput):
    return 0

def kuramoto(t_state, wn):
    K=5
    delta = 0.03
    num_osc = len(t_state[0])
    del_theta = np.zeros([1, num_osc])
    for i in range(num_osc):
        diff_phase = 0
        # Government equation
        for j in range(num_osc):
            diff_phase += np.math.sin(t_state[1][j] - t_state[1][i])
        diff_phase /= (num_osc - 1)
        t_state[0][i] = wn + K * diff_phase * delta
        t_state[1][i] += (wn + K * diff_phase) * delta
        # if t_state[1][i] > np.pi * 2:
        #     t_state[1][i] -= np.pi * 2
    return t_state

def janus(t_state, wn):
    num_osc = len(t_state[0])
    del_theta = np.zeros([1, num_osc])
    beta = 10
    sigma = 10
    vn = 2
    delta = 0.03
    for i in range(0, num_osc):
        if i - 1 == 0:
            j = num_osc - 1
        elif i + 1 == num_osc:
            j = 0
        else:
            j = i
        t_state[0][i] += (vn + beta * np.sin(t_state[1][i] - t_state[0][i]) + sigma * np.sin(
            t_state[1][j] - t_state[0][i])) * delta
        t_state[1][i] += (wn + beta * np.sin(t_state[0][i] - t_state[1][i]) + sigma * np.sin(
            t_state[0][j] - t_state[1][i])) * delta
        # if t_state[1][i] > np.pi * 2:
        #     t_state[1][i] -= np.pi * 2
    return t_state

def swarmalator(t_state, wn):
    c=1
    delta=0.03
    num_osc = len(t_state[0])
    del_theta = np.zeros([1, num_osc])
    del_r = np.zeros([1, num_osc])
    # the index of robot
    i = len(t_state[0]) - 1
    # for i in range(num_osc):
    diff_phase = 0
    diff_r = 0
    J = 1
    for j in range(num_osc):
        if (t_state[0][j] - t_state[0][i] > np.e):
            # second and third term of first equation
            diff_r += (t_state[0][j] - t_state[0][i]) / abs(t_state[0][j] - t_state[0][i]) * \
                      (1 + J * np.math.cos(t_state[1][j] - t_state[1][i])) - (t_state[0][j] - t_state[0][i]) \
                      / abs(t_state[0][j] - t_state[0][i]) ** 2

            # second term of second equation
            diff_phase += np.math.sin(t_state[1][j] - t_state[1][i]) / abs(t_state[0][j] - t_state[0][i])

    diff_r /= num_osc

    diff_phase /= num_osc - 1
    # first equation
    del_r[0][i] = 0.03 + diff_r
    # second equation
    del_theta[0][i] = wn + c * diff_phase
    del_all = np.append(del_r, del_theta, axis=0)
    t_state += del_all * delta

    # if t_state[1][i] > np.pi * 2:
    #     t_state[1][i] -= np.pi * 2
    return t_state

if __name__ == "__main__":
    startFlag=False
    #####################STEP 1=BEAT DETECTION #####################
    # create new audio file from a video file
    videoFileName= 'Vid_18_Nocturne_vn_fl_tpt 5.mp4'
    # Replace This with our algorithm phases
    # y, sr = librosa.load(videoFileName)
   # default_speaker.play(y, sr, None, 0.0000001)
    # tempo, beats = librosa.beat.beat_track(y=y, sr=sr)
    # beat_times = librosa.frames_to_time(beats, sr=sr)

    HUMAN_WIDTH = 70
    MARGIN = 100
    FRAMECOUNTS = 100

    ## person_detect
    objPersonDetect = PersonDetect.PersonDetectFromVideo(videoFileName, FRAMECOUNTS, HUMAN_WIDTH)
    arrX, person_num = objPersonDetect.personPosition()

    # Give the input manually
    # person_num=2
    # arrX = [210, 550]

    HUMAN_WIDTH = 100
    MARGIN = 100
    FRAMECOUNTS = 100

    # read video file
    videoCapture = cv2.VideoCapture(videoFileName)
    # frame count per second
    fps = videoCapture.get(cv2.CAP_PROP_FPS)

    audioFile = 'audio.wav'
    video = VideoFileClip(videoFileName)
    audio = video.audio
    audio.write_audiofile(audioFile)

    MARGIN = 100
    detector = [None] * (person_num)
    arrDf = [None] * (person_num)
    pTime = time.time()
    arrTime = [[0 for x in range(0)] for y in range(person_num)]
    # initialize the PoseModule object
    for i in range(person_num):
        detector[i] = poseModule.PoseDetector()
        arrDf[i] = pd.DataFrame(columns=['KeyPoint' + str(i) for i in range(17)])

    bufferSize=50
    k = 0

    state = np.zeros([2, person_num + 1])
    new_state = np.zeros([2, person_num + 1])
    person_key_points = {}

    currIndex=1

    a_source = aubio.source(audioFile, 44100, hop_s)
    a_tempo = aubio.tempo("default", win_s, hop_s, 44100)
    beat_times1 =[]
    # play the sound part

    CHUNK = int(44100//fps)
    wf = wave.open(audioFile, 'rb')
    p = pyaudio.PyAudio()
    stream = p.open(format=p.get_format_from_width(wf.getsampwidth()),
                    channels=wf.getnchannels(),
                    rate=wf.getframerate(),
                    output=True)
    data = wf.readframes(CHUNK)
    res_beat_times=[]
    while True:
        success, frame = videoCapture.read()
        if success:
            # play sound
            stream.write(data)
            data = wf.readframes(CHUNK)

            frame = imutils.resize(frame, width=1000)
            samples, read = a_source()
            is_beat = a_tempo(samples)
            if is_beat:
                beat_times1.append(videoCapture.get(cv2.CAP_PROP_POS_MSEC)/1000)
            for i in range(person_num):
                # Split frame according human position
                if i == 0:
                    subFrame = frame[:, 0:(int)(arrX[i] + HUMAN_WIDTH + MARGIN)]
                else:
                    subFrame = frame[:,
                               (int)(arrX[i - 1] + HUMAN_WIDTH):(int)(arrX[i] + HUMAN_WIDTH + MARGIN)]

                img, lmList = detector[i].findPose(subFrame)
                newList = []

                if len(lmList) != 0:
                    for j in range(len(lmList)):
                        if j in (0, 2, 5, 7, 8, 11, 12, 13, 14, 15, 16, 23, 24, 25, 26, 27, 28):
                            newList.append(lmList[j])
                    if k<bufferSize:
                        arrTime[i].append(videoCapture.get(cv2.CAP_PROP_POS_MSEC)/1000)
                        arrDf[i].loc[k] = newList
                    else:
                        startFlag=True
                        arrTime[i] = arrTime[i][1:]
                        arrTime[i].append(videoCapture.get(cv2.CAP_PROP_POS_MSEC)/1000)

                        arrDf[i]=arrDf[i].shift(-1)
                        arrDf[i].loc[k-1]=newList
                lmList = []
            if k<bufferSize:
                k += 1

            person_key_points = {}
            for i in range(person_num):
                person_key_points["Musician" + str(i + 1)] = arrDf[i]
            if startFlag:
                yp, arrTime=opticalFlow(person_key_points, False)
                # Finding peaks from the signal, to get the phase values
                phasePeakTimeList = [[0 for x in range(0)] for y in range(person_num)]
                peakCountList = [[0 for x in range(0)] for y in range(person_num)]
                angVelocityList = [[0 for x in range(0)] for y in range(person_num)]
                #
                for i in range(person_num):
                    person = (yp[i] - np.min(yp[i])) / np.ptp(yp[i])
                    smoothPerson = savgol_filter(person, 5, 3)
                    # Find indices of peaks
                    peak_times_idx, peak_values_dic = find_peaks(smoothPerson, 0)
                    arrTime_n_i = np.array(arrTime[i])
                    # peak time and peak values
                    peak_times = arrTime_n_i[peak_times_idx]
                    peak_values = np.array([list(item) for item in peak_values_dic.values()])[0]
                    # phasePeakTime list
                    # add zero value
                    phasePeakTimes = [0]
                    phasePeakTimes.extend(peak_times)
                    phasePeakTimeList[i].append(peak_times)
                    angVelocityList[i].append(2 * np.pi/ np.diff(phasePeakTimes, 1))
                    peakCountList[i].append(len(peak_times))

                # current state
                cur_t_pass=[time.time() - pTime - arrTime[i][-1] for i in range(person_num)]
                angvList = [angVelocityList[i][0][-1] for i in range(person_num)]
                # initialize the state value [[r1, r2],[phase1, phase2]]
                state[0][0:person_num] = [yp[i][-1]*1000 for i in range(person_num)]
                state[0][person_num] = new_state[0][person_num]
                state[1][0:person_num] = np.multiply(angvList, cur_t_pass)
                state[1][person_num] = new_state[1][person_num]
                if len(beat_times1)>1:
                    wn=2*np.pi/(beat_times1[-1]-beat_times1[-2])
                else:
                    wn=np.average(angvList)
                # new_state = swarmalator(state, wn)
                # new_state = kuramoto(state, wn)
                new_state = janus(state, wn)
                if new_state[1][person_num]>2*np.pi:
                    new_state[1][person_num]=new_state[1][person_num]-2*np.pi
                    # winsound.Beep(freq, duration)
                    mac_beep(freq, duration)
                    res_beat_times.append(videoCapture.get(cv2.CAP_PROP_POS_MSEC)/1000);


                # # Plot peaks (red) and valleys (blue)
                # plt.plot(arrTime[person_num - 1][1:-1], smoothPerson[1:])
                # plt.plot(peak_times, peak_values, 'r.')
                # plt.draw()
                # # plt.show(block=False)
                # plt.pause(0.0001)
                # plt.clf()
            #
            cv2.imshow("Image", frame)
            cv2.waitKey(1)
        else:
            break
    videoCapture.release()
    cv2.destroyAllWindows()
    person_key_points = {}
    import pandas as pd
    outputfilename=videoCapture+"_output.csv"
    pd.DataFrame(res_beat_times,columns=['BeatTimes']).to_csv(outputfilename,index=False)
        
 
     
    


