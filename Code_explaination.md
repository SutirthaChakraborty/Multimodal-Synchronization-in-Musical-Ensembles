# Explanation of main_librosa.py

This Python script demonstrates the process of detecting beats from a video file, estimating the positions of people in the video, and simulating the behavior of a swarmalator system or Kuramoto model.

The script can be divided into the following sections:

1. **Import necessary libraries**: Import required libraries such as NumPy, SciPy, Matplotlib, Librosa, OpenCV, and others.

2. **Define helper function getBPM()**: This function calculates the beats per minute (BPM) given an array of beat times, the current time, and the previous BPM.

3. **Step 1 - Beat detection**: Load the video file and extract audio using Librosa. Calculate the tempo (BPM) and beat times using Librosa's beat tracking functionality.

4. **Step 2 - Person detection**: Create an instance of the `PersonDetectFromVideo` class from the `PersonDetect` module, providing the video file name, number of frames to process, and the human width as input parameters. Estimate the number of people and their positions using the `personPosition()` method.

5. **Step 3 - Optical flow pose**: Use the `poseDetections()` method to detect the poses of the people in the video. Calculate the optical flow pose of each person with the `opticalFlowPose()` method.

6. **Calculate initial velocity**: Compute the initial velocity `vn` based on the average difference in positions.

7. **Find peaks in the signal**: Detect the peaks in the position signal for each person using the `find_peaks()` function from SciPy. Calculate the phase peak times and angular velocities for each person.

8. **Initialize the swarmalator or Kuramoto model**: Create an instance of the `AlgorithmSimulate` class from the `AlgorithmTest` module, providing the time step `delta`, natural frequency `wn`, and the number of people as input parameters.

9. **Iterate through the video frames**: For each frame of the video, update the state of the swarmalator system or Kuramoto model using the `swarmalator()` or `kuramoto()` methods of the `AlgorithmSimulate` instance. Update the positions and phases of each person in the video based on the calculated values.

10. **Visualize the results**: Plot the positions and phases of each person in the video, as well as the predicted beats per minute and the ground truth beats per minute. Calculate the mean square error (MSE) between the predicted BPM and the ground truth BPM, and display it on the plot.

The script demonstrates how to combine beat detection, person detection, and swarmalator/Kuramoto model simulations to analyze and visualize the behavior of people in a video based on their positions and the beats of the audio track.





# Explanation of PersonDetect.py

This Python script imports the necessary libraries (`numpy`, `cv2`, `imutils`, `pandas`, `time`, `matplotlib.pyplot`, and `poseModule`) and defines a `PersonDetectFromVideo` class for detecting persons and their poses in a video using the previously defined `poseModule`.

The main features of this code are:

1. **Class initialization**: The `PersonDetectFromVideo` class is initialized with several parameters:
    - `file`: The video file name.
    - `counts`: The frame numbers used to evaluate a person's position.
    - `humanWidth`: The human width in the video frame.

2. **personPosition() method**: This method processes the input video to detect and track persons' positions. It performs the following steps:
    - Initialize the kernel, background remover, and video capture.
    - Loop through the video frames and apply the background remover.
    - Perform dilation and erosion on the resulting image.
    - Detect moving objects in the frame.
    - If the width and height of a moving object meet the defined criteria, add its position to the `person_position` list.
    - After the specified number of frames (`self.TEST_FRAME`), use the detected positions to classify the persons in the video.
    - Return the detected person positions and the number of persons.

3. **poseDetections() method**: This method takes the detected person positions and the number of persons, then performs pose detection on each person in the input video. It performs the following steps:
    - Initialize the video capture, detectors, and data frames.
    - Loop through the video frames and apply the pose detector to each person in the frame.
    - If pose landmarks are detected, store them in the data frame for each person.
    - Store the resulting pose data in the `person_key_points` dictionary.

4. **opticalFlowPose() method**: This method calculates the average motion for each person in the video based on their pose data. It performs the following steps:
    - Convert the pose data into NumPy arrays.
    - Calculate the differences in the x and y positions of the pose keypoints.
    - Calculate the magnitudes of the position differences.
    - Calculate the average pose keypoint motion for each person.
    - Optionally, plot the average motion per frame for each person.

The `PersonDetectFromVideo` class can be used to analyze videos and detect persons and their poses by creating an instance of the class, calling the `personPosition()` method to detect person positions, calling the `poseDetections()` method to detect poses, and calling the `opticalFlowPose()` method to calculate the average motion.


# Explanation of posemodule.py

This Python script imports the necessary libraries (`cv2`, `mediapipe`, and `time`) and defines a `PoseDetector` class for detecting human poses in images using the MediaPipe library. The main features of this code are:

1. **Class initialization**: The `PoseDetector` class is initialized with several parameters:
    - `mode`: A boolean indicating whether to use the model's static image mode (False) or video mode (True).
    - `upBody`: A boolean indicating whether to focus on the upper body only (True) or the whole body (False).
    - `smooth`: A boolean indicating whether to enable landmark smoothing (True) or not (False).
    - `detectionCon`: A float representing the minimum detection confidence threshold.
    - `trackCon`: A float representing the minimum tracking confidence threshold.

    The class also initializes the drawing utilities and pose estimation models from the MediaPipe library.

2. **findPose() method**: This method takes an input image and processes it to detect human poses. It performs the following steps:
    - Convert the input image to RGB format.
    - Apply the pose estimation model to the image.
    - If pose landmarks are detected, draw the landmarks and their connections on the input image.
    - Create a list (`lmList`) containing the x and y coordinates of each detected landmark.

    The method returns the modified input image with the drawn landmarks and the list of landmark coordinates.

The `PoseDetector` class can be used to analyze images and detect human poses by creating an instance of the class and calling the `findPose()` method with an input image.



# Explanation of AlgorithmTest.py

This Python script defines two classes, `AlgorithmSimulate` and `swarmalator`, to simulate a swarmalator algorithm and Kuramoto model using input parameters.

1. **Class AlgorithmSimulate**:
    - **Initialization**: The `AlgorithmSimulate` class is initialized with the following parameters:
        - `delta`: Time step for the simulation.
        - `wn`: The natural frequency of the oscillators.
        - `person_num`: The number of persons (or oscillators) in the system.
        - `velocities`: Randomly generated velocities for each person.

2. **swarmalator() method**: This method simulates the swarmalator algorithm for the input state `t_state` and given velocity `vn`. The method performs the following steps:
    - Initialize variables for the number of oscillators, phase differences, and position differences.
    - Loop through each oscillator and calculate the differences in positions and phases between the current oscillator and all other oscillators.
    - Update the position and phase differences based on the swarmalator equations.
    - Apply the time step `self.delta` to the updated position and phase differences.
    - Return the updated state `t_state`.

3. **kuramoto() method**: This method simulates the Kuramoto model for the input state `t_state`. The method performs the following steps:
    - Initialize a variable for the number of oscillators and phase differences.
    - Loop through each oscillator and calculate the differences in phases between the current oscillator and all other oscillators.
    - Update the phase differences based on the Kuramoto equation.
    - Apply the time step `self.delta` to the updated phase differences.
    - Return the updated state `t_state`.

The `AlgorithmSimulate` class can be used to simulate the behavior of a swarmalator system or Kuramoto model by creating an instance of the class and calling the `swarmalator()` or `kuramoto()` methods with the appropriate input parameters.


# Explanation of main_aubio.py

This script processes a given video file to detect human movements and generate synchronized beeps or send commands to Sonic Pi.

## Code Explanation

1. Import necessary libraries, such as `numpy`, `pandas`, `cv2`, `aubio`, and custom modules like `PersonDetect` and `poseModule`.
2. Define global variables and functions, such as `mac_beep()`, `shift5()`, and `opticalFlow()`.
3. Define synchronization algorithms, such as `kuramoto()`, `janus()`, and `swarmalator()`.
4. Define the main function, `CallRobot()`, which takes several arguments and returns the output based on the input arguments.
5. In the `__main__` section of the script:
   - Read the input video file and extract the audio.
   - Initialize the `PersonDetect` object and detect the number of persons in the video.
   - Initialize the `PoseDetector` object for each person.
   - Use the `aubio` library to detect beats in the audio.
   - Process the video frames and detect human pose for each person.
   - Calculate the optical flow using the `opticalFlow()` function.
   - Find peaks in the motion signal and synchronize the output according to the chosen algorithm (e.g., Kuramoto, Janus, or Swarmalator).
   - Generate beeps or send commands to Sonic Pi based on the synchronized output.

## Usage

To use this script, call the `CallRobot()` function with the appropriate input arguments, such as:

- `filename`: The video file name or 'live' for camera input.
- `function`: The synchronization algorithm to use ('Kuramoto', 'swarmlator', 'janus', or 'flock').
- `videoFlag`: A flag to indicate if video synchronization is needed.
- `output`: The output method ('sonicpi/tap' for Sonic Pi or tap sound).
- `ip`: The IP address for Sonic Pi (optional).
- `port`: The port number for Sonic Pi (optional).
- `writeoutput`: A flag to indicate if the output timestamps should be written to a file (optional).
