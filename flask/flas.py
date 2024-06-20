from flask import Flask, render_template, Response
import cv2
from imutils import face_utils
from scipy.spatial import distance as dist
import dlib
from pygame import mixer
import imutils

app = Flask(__name__)

mixer.init()
mixer.music.load("music.wav")

def eye_aspect_ratio(eye):
    A = dist.euclidean(eye[1], eye[5])
    B = dist.euclidean(eye[2], eye[4])
    C = dist.euclidean(eye[0], eye[3])
    return (A + B) / (2.0 * C)

def mouth_aspect_ratio(mouth):
    A = dist.euclidean(mouth[13], mouth[19])
    B = dist.euclidean(mouth[14], mouth[18])
    C = dist.euclidean(mouth[15], mouth[17])
    return (A + B + C) / 3.0

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

def detect_drowsiness_and_yawning(frame):
    global flag, flagd, flaga, count_yawn

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
            if flag >= frame_check:      
                flaga = 0
                flagd = 0
                count_yawn = 0
                cv2.putText(frame, "****************ALERT!****************", (10, 30),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
                mixer.music.play()
                
        elif mouthMAR > 14:
            count_yawn += 1
            if count_yawn >= 16:
                cv2.putText(frame, "YAWNING ALERT!", (270, 30),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
                
        elif 0.25 > ear > 0.19:
            flagd += 1
            if flagd >= 25:
                flag = 0
                flaga = 0
                cv2.putText(frame, "***DROWSY TAKE BREAK***", (10, 30),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
                
        elif ear > 0.25:
            flaga += 1
            count_yawn = 0
            if flaga >= 5:
                flagd = 0
                flag = 0
                cv2.putText(frame, "***ACTIVE***", (10, 30),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)

    return frame

def generate_frames():
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        else:
            frame = detect_drowsiness_and_yawning(frame)
            ret, buffer = cv2.imencode('.jpg', frame)
            frame = buffer.tobytes()
            yield (b'--frame\r\n'
                b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')

@app.route('/')
def index():
    return render_template('base.html')

@app.route('/video_feed')
def video_feed():
    return Response(generate_frames(), mimetype='multipart/x-mixed-replace; boundary=frame')

if __name__ == '__main__':
    app.run(debug=True)
