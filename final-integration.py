import cv2
import math
import numpy as np
import dlib
from imutils import face_utils
import imutils
from matplotlib import pyplot as plt

import train as train
import sys
import webbrowser
import datetime


def yawn(mouth):
    return ((euclideanDist(mouth[2], mouth[10]) + euclideanDist(mouth[4], mouth[8])) / (
            2 * euclideanDist(mouth[0], mouth[6])))


def getFaceDirection(shape, size):
    if len(shape) < 6:
        return 0  # Handle the case when shape is not available

    image_points = np.array([
        shape[33],  # Nose tip
        shape[8],  # Chin
        shape[45],  # Left eye left corner
        shape[36],  # Right eye right corne
        shape[54],  # Left Mouth corner
        shape[48]  # Right mouth corner
    ], dtype="double")

    # 3D model points.
    model_points = np.array([
        (0.0, 0.0, 0.0),  # Nose tip
        (0.0, -330.0, -65.0),  # Chin
        (-225.0, 170.0, -135.0),  # Left eye left corner
        (225.0, 170.0, -135.0),  # Right eye right corne
        (-150.0, -150.0, -125.0),  # Left Mouth corner
        (150.0, -150.0, -125.0)  # Right mouth corner
    ])

    # Camera internals
    focal_length = size[1]
    center = (size[1] / 2, size[0] / 2)
    camera_matrix = np.array(
        [[focal_length, 0, center[0]],
         [0, focal_length, center[1]],
         [0, 0, 1]], dtype="double"
    )
    dist_coeffs = np.zeros((4, 1))  # Assuming no lens distortion

    # Solve PnP
    (success, rotation_vector, translation_vector) = cv2.solvePnP(model_points, image_points, camera_matrix,
                                                                  dist_coeffs, flags=cv2.SOLVEPNP_ITERATIVE)

    if not success:
        return 0  # Handle the case when solvePnP fails to find a solution

    return translation_vector[1][0]


def euclideanDist(a, b):
    return math.sqrt(math.pow(a[0] - b[0], 2) + math.pow(a[1] - b[1], 2))


# EAR -> Eye Aspect ratio
def ear(eye):
    return (euclideanDist(eye[1], eye[5]) + euclideanDist(eye[2], eye[4])) / (2 * euclideanDist(eye[0], eye[3]))


def writeEyes(a, b, img):
    y1 = max(a[1][1], a[2][1])
    y2 = min(a[4][1], a[5][1])
    x1 = a[0][0]
    x2 = a[3][0]

    # Ensure that the specified ROI is within the bounds of the image
    if 0 <= y1 < y2 <= img.shape[0] and 0 <= x1 < x2 <= img.shape[1]:
        cv2.imwrite('left-eye.jpg', img[y1:y2, x1:x2])

    y1 = max(b[1][1], b[2][1])
    y2 = min(b[4][1], b[5][1])
    x1 = b[0][0]
    x2 = b[3][0]

    # Ensure that the specified ROI is within the bounds of the image
    if 0 <= y1 < y2 <= img.shape[0] and 0 <= x1 < x2 <= img.shape[1]:
        cv2.imwrite('right-eye.jpg', img[y1:y2, x1:x2])


# open_avg = train.getAvg()
# close_avg = train.getAvg()

#alert = vlc.MediaPlayer('focus.mp3')

frame_thresh_1 = 15
frame_thresh_2 = 10
frame_thresh_3 = 7

close_thresh = 0.3
flag = 0
map_counter = 0

capture = cv2.VideoCapture(1)
avgEAR = 0
detector = dlib.get_frontal_face_detector()
predictor = dlib.shape_predictor('shape_predictor_68_face_landmarks.dat')
(leStart, leEnd) = face_utils.FACIAL_LANDMARKS_IDXS["left_eye"]
(reStart, reEnd) = face_utils.FACIAL_LANDMARKS_IDXS["right_eye"]
(mStart, mEnd) = face_utils.FACIAL_LANDMARKS_IDXS["mouth"]

while True:
    ret, frame = capture.read()
    size = frame.shape
    gray = frame
    rects = detector(gray, 0)

    if len(rects) > 0:
        shape = face_utils.shape_to_np(predictor(gray, rects[0]))
        leftEye = shape[leStart:leEnd]
        rightEye = shape[reStart:reEnd]
        leftEyeHull = cv2.convexHull(leftEye)
        rightEyeHull = cv2.convexHull(rightEye)
        leftEAR = ear(leftEye)
        rightEAR = ear(rightEye)
        avgEAR = (leftEAR + rightEAR) / 2.0
        eyeContourColor = (255, 255, 255)

        print('Average Ear:', avgEAR)
        
        if avgEAR < close_thresh:
            flag += 1
            eyeContourColor = (0, 255, 255)
            # print(flag)
            if flag >= frame_thresh_3:
                eyeContourColor = (147, 20, 255)
                cv2.putText(gray, "Drowsy", (50, 50), cv2.FONT_HERSHEY_COMPLEX, 1, (0, 255, 127), 2)
                #alert.play()

            elif flag >= frame_thresh_1:
                eyeContourColor = (0, 0, 255)
                cv2.putText(gray, "Drowsy (Normal)", (50, 50), cv2.FONT_HERSHEY_COMPLEX, 1, (0, 255, 127), 2)
                #alert.play()

        elif avgEAR > close_thresh and flag:
            # print("Flag reseted to 0")
            #alert.stop()
            yawn_countdown = 0
            map_flag = 1
            flag = 0

        cv2.drawContours(gray, [leftEyeHull], -1, eyeContourColor, 2)
        cv2.drawContours(gray, [rightEyeHull], -1, eyeContourColor, 2)

        writeEyes(leftEye, rightEye, frame)

        face_direction = getFaceDirection(shape, size)
        # print('Face Direction:', face_direction)

    """if avgEAR > close_thresh:
        #alert.stop()"""

    cv2.imshow('Driver', gray)
    if cv2.waitKey(1) == 27:
        break

capture.release()
cv2.destroyAllWindows()
