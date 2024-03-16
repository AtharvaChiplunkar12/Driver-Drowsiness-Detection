import webbrowser

import cv2
import math
import numpy as np
import dlib
import imutils
from imutils import face_utils
import vlc
import time


def euclideanDist(a, b):
    return math.sqrt(math.pow(a[0] - b[0], 2) + math.pow(a[1] - b[1], 2))


def ear(eye):
    return (euclideanDist(eye[1], eye[5]) + euclideanDist(eye[2], eye[4])) / (2 * euclideanDist(eye[0], eye[3]))


frame_thresh_1 = 10
# frame_thresh_2 = 10
# frame_thresh_3 = 5

close_thresh = 0.2
flag = 0
yawn_countdown = 0
map_counter = 0
map_flag = 1

alert = vlc.MediaPlayer('focus.mp3')
# video_path = '/Users/siddheshdhonde/Documents/CSCI_631/Drowsiness Detection/YawDD dataset/Dash/Male/9-MaleNoGlasses.avi'
# capture = cv2.VideoCapture(video_path)
#
# avgEAR = 0
# detector = dlib.get_frontal_face_detector()
# predictor = dlib.shape_predictor('shape_predictor_68_face_landmarks.dat')
# (leStart, leEnd) = face_utils.FACIAL_LANDMARKS_IDXS["left_eye"]
# (reStart, reEnd ) = face_utils.FACIAL_LANDMARKS_IDXS["right_eye"]
# (mStart, mEnd) = face_utils.FACIAL_LANDMARKS_IDXS["mouth"]


capture = cv2.VideoCapture(1)
avgEAR = 0
detector = dlib.get_frontal_face_detector()
predictor = dlib.shape_predictor('shape_predictor_68_face_landmarks.dat')
(leStart, leEnd) = face_utils.FACIAL_LANDMARKS_IDXS["left_eye"]
(reStart, reEnd) = face_utils.FACIAL_LANDMARKS_IDXS["right_eye"]
(mStart, mEnd) = face_utils.FACIAL_LANDMARKS_IDXS["mouth"]

# Initialize variables for FPS calculation
fps_start_time = time.time()
fps_frame_count = 0

# Capture every 2nd frame
frame_skip = 2

while True:
    if cv2.waitKey(1) == ord('Q') or cv2.waitKey(1) == ord('q'):
        break
    ret, frame = capture.read()

    # Resize the frame
    frame = imutils.resize(frame, width=1000, height=1000)  # Adjust the width as needed

    size = frame.shape
    gray = frame
    rects = detector(gray, 0)

    if len(rects):
        shape = face_utils.shape_to_np(predictor(gray, rects[0]))
        leftEye = shape[leStart:leEnd]
        rightEye = shape[reStart:reEnd]
        leftEyeHull = cv2.convexHull(leftEye)
        rightEyeHull = cv2.convexHull(rightEye)

        leftEAR = ear(leftEye)
        rightEAR = ear(rightEye)
        avgEAR = (leftEAR + rightEAR) / 2.0
        eyeContourColor = (255, 255, 255)

        if avgEAR < close_thresh:
            flag += 1
            eyeContourColor = (0, 255, 255)
            print(flag)
            if flag >= frame_thresh_1:
                eyeContourColor = (147, 20, 255)
                cv2.putText(gray, "Drowsiness Detected", (50, 50), cv2.FONT_HERSHEY_COMPLEX, 1, (0, 255, 127), 2)
                alert.play()

        elif avgEAR > close_thresh and flag:
            print("Flag reseted to 0")
            alert.stop()
            yawn_countdown = 0
            map_flag = 1
            flag = 0

        cv2.drawContours(gray, [leftEyeHull], -1, eyeContourColor, 2)
        cv2.drawContours(gray, [rightEyeHull], -1, eyeContourColor, 2)

    if avgEAR > close_thresh:
        alert.stop()
    cv2.imshow('Driver', gray)

capture.release()
cv2.destroyAllWindows()