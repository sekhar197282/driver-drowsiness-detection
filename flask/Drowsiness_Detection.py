from scipy.spatial import distance as dist
from imutils import face_utils
from pygame import mixer
import imutils
import dlib
import cv2

mixer.init()
mixer.music.load("C:/Users/sekha/OneDrive/Desktop/music.wav")

def eye_aspect_ratio(eye):
    # Compute the euclidean distances between the two sets of vertical eye landmarks (x, y)-coordinates
    A = dist.euclidean(eye[1], eye[5])
    B = dist.euclidean(eye[2], eye[4])

    # Compute the euclidean distance between the horizontal eye landmark (x, y)-coordinates
    C = dist.euclidean(eye[0], eye[3])

    # Compute the eye aspect ratio
    ear = (A + B) / (2.0 * C)
    return ear

def mouth_aspect_ratio(mouth):
    A = dist.euclidean(mouth[13], mouth[19])
    B = dist.euclidean(mouth[14], mouth[18])
    C = dist.euclidean(mouth[15], mouth[17])
    MAR = (A + B + C) / 3.0
    return MAR

thresh = 0.18
frame_check = 21
detect = dlib.get_frontal_face_detector()
predict = dlib.shape_predictor("models/shape_predictor_68_face_landmarks.dat")

(lStart, lEnd) = face_utils.FACIAL_LANDMARKS_IDXS["left_eye"]
(rStart, rEnd) = face_utils.FACIAL_LANDMARKS_IDXS["right_eye"]
(mStart, mEnd) = face_utils.FACIAL_LANDMARKS_IDXS["mouth"]

cap = cv2.VideoCapture(0)
flag = 0
flagd = 0
flaga = 0
count_yawn = 0

while True:
    ret, frame = cap.read()
    frame = imutils.resize(frame, width=600)
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    subjects = detect(gray, 0)
    for subject in subjects:
        shape = predict(gray, subject)
        shape = face_utils.shape_to_np(shape)
        leftEye = shape[lStart:lEnd]
        rightEye = shape[rStart:rEnd]
        mouth = shape[mStart:mEnd]
        leftEAR = eye_aspect_ratio(leftEye)
        rightEAR = eye_aspect_ratio(rightEye)
        ear = (leftEAR + rightEAR) / 2.0
        mouthMAR = mouth_aspect_ratio(mouth)
        
        leftEyeHull = cv2.convexHull(leftEye)
        rightEyeHull = cv2.convexHull(rightEye)
        mouthHull = cv2.convexHull(mouth)

        cv2.drawContours(frame, [leftEyeHull], -1, (0, 255, 0), 1)
        cv2.drawContours(frame, [rightEyeHull], -1, (0, 255, 0), 1)

        
        cv2.drawContours(frame, [mouthHull], -1, (0, 0, 255), 1)

        if ear < 0.19:
            flag += 1
            print ("s",flag)
            if flag >= frame_check:      
                flaga = 0
                flagd = 0
                count_yawn=0
                cv2.putText(frame, "****************ALERT!****************", (10, 30),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
                mixer.music.play()
                
        elif mouthMAR > 14:
            count_yawn += 1
            print ("y-",count_yawn)
            if count_yawn >= 16:
                cv2.putText(frame, "YAWNING ALERT!", (270, 30),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
                
        elif 0.25 > ear > 0.19:
            flagd += 1
            print ("d",flagd)
            if flagd >= 25:
                flag = 0
                flaga = 0
                cv2.putText(frame, "***DROWSY TAKE BREAK***", (10, 30),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
                
        elif ear > 0.25:
            flaga += 1
            count_yawn=0
            if flaga >= 5:
                flagd = 0
                flag = 0
                cv2.putText(frame, "***ACTIVE***", (10, 30),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)

    cv2.imshow("Frame", frame)
    key = cv2.waitKey(1) & 0xFF
    if key == ord("q"):
        break

cv2.destroyAllWindows()
cap.release()
