import numpy as np
from scipy.signal import savgol_filter
from scipy.signal import find_peaks
import matplotlib.pyplot as plt
import librosa
import cv2
import time

import PersonDetect
import AlgorithmTest

import soundfile as sf
# import soundcard as sc

# default_speaker = sc.default_speaker()

def getBPM(array, t, bpm):
    try:
        array = list(np.round(array, 3))
        index = array.index(round(t, 3))
        bpm = 60000 / (array[index] * 1000 - array[index - 1] * 1000)
    except:
        index = -1
    return bpm


if __name__ == "__main__":
    ## initial setting value
    HUMAN_WIDTH = 70
    MARGIN = 100
    FRAMECOUNTS = 50

    #####################STEP 1=BEAT DETECTION #####################
    # create new audio file from a video file
    videoFileName = 'Vid_18_Nocturne_vn_fl_tpt 5.mp4'
    # Replace This with our algorithm phases
    y, sr = librosa.load(videoFileName)

    vid = cv2.VideoCapture(videoFileName)
    height = vid.get(cv2.CAP_PROP_FRAME_HEIGHT)
    width = vid.get(cv2.CAP_PROP_FRAME_WIDTH)

    tempo, beats = librosa.beat.beat_track(y=y, sr=sr)
    beat_times = librosa.frames_to_time(beats, sr=sr)
    wn = tempo * 2 * np.pi / 60
    delta = 0.02

    print("Completed!  Estimated Beats Per Minute:", tempo, "angle velocity", wn)
    #####################STEP 2=Person DETECTION #####################
    ## create object of class PersonDetectFromVideo
    objPersonDetect = PersonDetect.PersonDetectFromVideo(videoFileName, FRAMECOUNTS, HUMAN_WIDTH)
    # arrX, person_num = objPersonDetect.personPosition()
    arrX = np.array([100, 400, 700])
    person_num = 3
    ## create object of class PersonDetectFromVideo
    simulate = AlgorithmTest.AlgorithmSimulate(delta, wn, person_num)
    # arrX = np.array([20, 250, 500, 750])
    print("Completed!  Estimated Person's number and Position:", person_num, arrX)
    #####################STEP 3=Optical flow pose #####################
    objPersonDetect.poseDetections(arrX, person_num)
    # print(time.time())
    yp, arrTime = objPersonDetect.opticalFlowPose()

    # calculate initial vn value
    vn = np.average(np.diff(yp[0], 1)) / delta

    # Finding peaks from the signal, to get the phase values
    phasePeakTimeList = [[0 for x in range(0)] for y in range(person_num)]
    peakCountList = [[0 for x in range(0)] for y in range(person_num)]
    angVelocityList = [[0 for x in range(0)] for y in range(person_num)]
    ## peak times, angle velocity
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
        angVelocityList[i].append(2 * np.pi / np.diff(phasePeakTimes, 1))
        peakCountList[i].append(len(peak_times))

    ## Plot peaks (red) and valleys (blue)
    # plt.plot(arrTime[person_num-1][1:-1], smoothPerson[1:])
    # plt.plot(peak_times, peak_values, 'r.')
    # plt.show()
    # initialize the machine signal
    conductor = []
    # find every person's phase for every time ; phaseValueList
    new_state = np.zeros([2, person_num + 1])

    currentIndex = [0] * person_num
    peakCurrentIndex = [0] * person_num
    angleVelocityValue = [0] * person_num

    rValueList = [0] * person_num
    phaseValueList = [0] * person_num

    state = np.zeros([2, person_num + 1])
    i = 0
    iter = True

    # # init the figure axis
    fig = plt.figure()
    ax = fig.add_subplot(111, xlim=(-100, 100), ylim=(-100, 100))

    color1 = np.random.random(person_num + 1)
    colors = ['r', 'g', 'b', 'y', 'k']
    scats = []
    for ii in range(person_num + 1):
        scat = ax.scatter([], [], color=colors[ii], label=('person' + str(ii)))
        scats.append(scat)
    plt.legend()
    timeScale = []
    tbpm = 0
    TrueBPM = []
    while iter:
        currentTime = delta * i
        timeScale.append(currentTime)

        tbpm = getBPM(beat_times, currentTime, tbpm)
        TrueBPM.append(tbpm)

        for j in range(person_num):
            if currentTime > arrTime[j][-1]:
                iter = False
                break
            if currentTime > arrTime[j][currentIndex[j]]:
                if currentIndex[j] < len(yp[j]) - 1:
                    currentIndex[j] += 1
            rValueList[j] = yp[j][currentIndex[j]] * 1000

            if currentTime < phasePeakTimeList[j][0][peakCurrentIndex[j]]:
                angleVelocityValue[j] = angVelocityList[j][0][peakCurrentIndex[j]]
                if peakCurrentIndex[j] == 0:
                    phaseValueList[j] = angleVelocityValue[j] * currentTime
                else:
                    phaseValueList[j] = angleVelocityValue[j] * (
                            currentTime - phasePeakTimeList[j][0][peakCurrentIndex[j] - 1])
            else:
                if peakCurrentIndex[j] < peakCountList[j][0] - 1:
                    peakCurrentIndex[j] += 1
                phaseValueList[j] = 0

        # initialize the state value [[r1, r2],[phase1, phase2]]
        state[0][0:person_num] = rValueList
        state[0][person_num] = new_state[0][person_num]
        state[1][0:person_num] = np.array(phaseValueList)
        state[1][person_num] = new_state[1][person_num]
        # algorithm implementation
        # new_state = simulate.swarmalator(state, vn)
        new_state = simulate.kuramoto(state)
        # new_state = simulate.janus(state)

        # substitute generated value
        conductor.append([new_state[0][person_num], new_state[1][person_num]])

        new_x = new_state[0][:] * np.cos(new_state[1][:])
        new_y = new_state[0][:] * np.sin(new_state[1][:])
        newState = np.column_stack([new_x, new_y])

        # particles.set_offsets(newState)
        # particles.set_array(color1)
        
        for ii in range(person_num + 1):
            sample = [new_x[ii], new_y[ii]]
            scats[ii].set_offsets(sample)
        plt.pause(0.01)
        i += 1

    ## predicted beat time
    predicted_beat_index = np.unique(np.round(np.array(conductor)[:, 1] / np.pi / 2), return_index=True)[1]
    predicted_beat_time = np.array(timeScale)[predicted_beat_index]
    np.savetxt("predicted_beat_time.csv", predicted_beat_time)
    # plot the result
    fig, ax = plt.subplots()
    ax.plot(np.array(conductor)[:, 1])
    plt.xlabel('Frames', fontsize=16)
    plt.ylabel('robot Motion', fontsize=16)
    plt.title('robot Motion', fontsize=16)
    plt.show()
    # evaluation part
    # angle velocity predicted
    w0 = np.diff(np.array(conductor)[:, 1]) / delta
    idxs = np.where(w0 > 0)
    figure, ax = plt.subplots()
    for i in range(person_num):
        ax.plot(np.array(phasePeakTimeList[i])[0], np.array(angVelocityList[i])[0] * 60 / 2 / np.pi, linewidth=1.0)

    ax.plot(np.array(timeScale)[idxs], w0[idxs] * 60 / 2 / np.pi)
    ax.plot(timeScale, TrueBPM)

    mse = np.round(np.sqrt(np.mean(((w0 - TrueBPM[1:]) * 60 / 2 / np.pi) ** 2)), 3)
    print('Mean Square Error = ', mse)

    plt.text(0.5, 0.7,
             "MSE = {}".format(mse),
             transform=plt.gca().transAxes)
    legend_val = ['Person' + str(i) for i in range(person_num)]
    legend_val.append('Predicted')
    legend_val.append('ground truth')
    plt.legend(legend_val)
    plt.xlabel('Time(s)', fontsize=16)
    plt.ylabel('Beat per Minute', fontsize=16)
    plt.show()
