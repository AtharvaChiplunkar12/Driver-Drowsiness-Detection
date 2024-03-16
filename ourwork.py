import cv2
import dlib
import imutils
from imutils import face_utils
#import vlc
from scipy.spatial import distance



def eye_aspect_ratio(eye):
    return ((distance.euclidean(eye[1], eye[5]) + distance.euclidean(eye[2], eye[4])) / (2 * distance.euclidean(eye[0], eye[3])))

def mouth_aspect_ratio(mouth):
    mar = (distance.euclidean(mouth[2], mouth[10]) + distance.euclidean(mouth[4], mouth[8])) / (
            2 * distance.euclidean(mouth[0], mouth[6]))
    print(mar)
    return mar
    
def extracting_eye_and_face_features():
    predictor = dlib.shape_predictor('shape_predictor_68_face_landmarks.dat')
    left_eye_start, left_eye_end = face_utils.FACIAL_LANDMARKS_68_IDXS["left_eye"]
    right_eye_start, right_eye_end = face_utils.FACIAL_LANDMARKS_68_IDXS["right_eye"]
    mouth_start, mouth_end = face_utils.FACIAL_LANDMARKS_68_IDXS["mouth"]
    return predictor, (left_eye_start, left_eye_end), (right_eye_start, right_eye_end), (mouth_start, mouth_end)



def dorwsiness_detection():
    captured_frame = cv2.VideoCapture(1)
    detector = dlib.get_frontal_face_detector()
    predictor, left_eye_points, right_eye_points, mouth_points = extracting_eye_and_face_features()
    drowsy_eye_threshold = 0.20
    continues_frame_threshold = 10
    mouth_frame_threshold = 5
    yawn_count = 0
    flag = 0
    
    while True:
        if cv2.waitKey(1) == ord('Q') or cv2.waitKey(1) == ord('q'):
            break
        ret, frame = captured_frame.read()
        frame = imutils.resize(frame, width=1000, height=1000)
        face_found = detector(frame, 0)
    
        if len(face_found) > 0:
            face_shape = face_utils.shape_to_np(predictor(frame, face_found[0]))
            left_eye = face_shape[left_eye_points[0]:left_eye_points[1]]
            right_eye = face_shape[right_eye_points[0]:right_eye_points[1]]
            mouth = face_shape[mouth_points[0]:mouth_points[1]]
            leftEyeBorder = cv2.convexHull(left_eye)
            rightEyeBorder = cv2.convexHull(right_eye)
            mouthBorder = cv2.convexHull(mouth)
            left_EAR = eye_aspect_ratio(left_eye)
            right_EAR = eye_aspect_ratio(right_eye)
            average_EAR = (left_EAR + right_EAR) / 2.0
            eye_color = (0, 255, 0)
            mouth_color = (0, 255, 0)

            if mouth_aspect_ratio(mouth) > 0.6:
                cv2.putText(frame, "Yawn Detected", (50, 50), cv2.FONT_HERSHEY_COMPLEX, 1, (0, 255, 127), 2)
                yawn_count = 1
                mouth_color = (147, 20, 255)

            if average_EAR < drowsy_eye_threshold:
                flag += 1
                eye_color = (0,255,255)
                if yawn_count and flag>=mouth_frame_threshold:
                        eye_color = (147, 20, 255)
                        mouth_color = (147, 20, 255)
                        cv2.putText(frame, "Drowsy after yawn", (100,100), cv2.FONT_HERSHEY_COMPLEX, 1,(0,255,127),2)
                        #vlc.MediaPlayer('focus.mp3').play()
                elif flag >= continues_frame_threshold:
                    eye_color = (255,0,0)
                    cv2.putText(frame, "Drowsiness Detected", (50, 50), cv2.FONT_HERSHEY_COMPLEX, 1, (0, 255, 127), 2)
                    #vlc.MediaPlayer('focus.mp3').play()
            elif average_EAR > drowsy_eye_threshold and flag:
                flag = 0
                eye_color = (0, 255, 0)
                yawn_count = 0
                #vlc.MediaPlayer('focus.mp3').stop()
            cv2.drawContours(frame, [leftEyeBorder], -1, eye_color, 2)
            cv2.drawContours(frame, [rightEyeBorder], -1, eye_color, 2)
            cv2.drawContours(frame, [mouthBorder], -1, mouth_color, 2)
        #if average_EAR > drowsy_eye_threshold:
            #vlc.MediaPlayer('focus.mp3').stop()
        cv2.imshow('Drivers Drowsiness', frame)
    
    captured_frame.release()
    cv2.destroyAllWindows()
            

dorwsiness_detection()
    
