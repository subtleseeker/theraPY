import cv2
import numpy as np
import dlib
import csv
from math import hypot
import pickle
import pandas as pd
from pygame import mixer

list123=[]
irises=[]
my_irises=[]
d=[]
mean=[]
counter=0

cap= cv2.VideoCapture(0)
detector=dlib.get_frontal_face_detector()
predictor = dlib.shape_predictor("shape_predictor_68_face_landmarks.dat")

font = cv2.FONT_HERSHEY_PLAIN


song_dict={
    "0":r"C:\Users\Pulkit Pahuja\Desktop\ML\Dr.BonnAI\Music\1.mp3",
    "1":r"C:\Users\Pulkit Pahuja\Desktop\ML\Dr.BonnAI\Music\2.mp3",
    "2":r"C:\Users\Pulkit Pahuja\Desktop\ML\Dr.BonnAI\Music\3.mp3",
    "3":r"C:\Users\Pulkit Pahuja\Desktop\ML\Dr.BonnAI\Music\4.mp3",
    "4":r"C:\Users\Pulkit Pahuja\Desktop\ML\Dr.BonnAI\Music\5.mp3",
    "5":r"C:\Users\Pulkit Pahuja\Desktop\ML\Dr.BonnAI\Music\6.mp3",
    "6":r"C:\Users\Pulkit Pahuja\Desktop\ML\Dr.BonnAI\Music\7.mp3",
    "7":r"C:\Users\Pulkit Pahuja\Desktop\ML\Dr.BonnAI\Music\8.mp3",
    "8":r"C:\Users\Pulkit Pahuja\Desktop\ML\Dr.BonnAI\Music\9.mp3",
    "9":r"C:\Users\Pulkit Pahuja\Desktop\ML\Dr.BonnAI\Music\10.mp3"
    }



def face_sentiment():

    path = "./face_classification/src/video_emotion_color_demo.py"



def get_irises_location(frame_gray):

    eye_cascade=cv2.CascadeClassifier(r'haarcascades/haarcascade_eye_tree_eyeglasses.xml')
    eyes = eye_cascade.detectMultiScale(frame_gray, 1.3, 5)  # if not empty - eyes detected

    irises=[]
    for (ex, ey, ew, eh) in eyes:
        iris_w = int(ex + float(ew / 2))
        iris_h = int(ey + float(eh / 2))
        irises.append([np.float32(iris_w), np.float32(iris_h)])
        my_irises.append([np.float32(iris_w), np.float32(iris_h)])

    return irises

def get_blinking_ratio(eye_points, facial_landmarks):
    left_point = (facial_landmarks.part(eye_points[0]).x, facial_landmarks.part(eye_points[0]).y)
    right_point = (facial_landmarks.part(eye_points[3]).x, facial_landmarks.part(eye_points[3]).y)
    center_top = midpoint(facial_landmarks.part(eye_points[1]), facial_landmarks.part(eye_points[2]))
    center_bottom = midpoint(facial_landmarks.part(eye_points[5]), facial_landmarks.part(eye_points[4]))

    #hor_line = cv2.line(frame, left_point, right_point, (0, 255, 0), 2)
    #ver_line = cv2.line(frame, center_top, center_bottom, (0, 255, 0), 2)

    hor_line_lenght = hypot((left_point[0] - right_point[0]), (left_point[1] - right_point[1]))
    ver_line_lenght = hypot((center_top[0] - center_bottom[0]), (center_top[1] - center_bottom[1]))

    ratio = hor_line_lenght / ver_line_lenght
    return ratio

def midpoint(p1 ,p2):
    return int((p1.x + p2.x)/2), int((p1.y + p2.y)/2)

def distance(v1, v2):
    return np.sqrt(np.sum((v1 - v2)**2))



while True:
    d=[]
    _, frame = cap.read()
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    irises=get_irises_location(gray)

    faces = detector(gray)
    for face in faces:
        #x, y = face.left(), face.top()
        #x1, y1 = face.right(), face.bottom()
        #cv2.rectangle(frame, (x, y), (x1, y1), (0, 255, 0), 2)

        landmarks = predictor(gray, face)

        left_eye_ratio = get_blinking_ratio([36, 37, 38, 39, 40, 41], landmarks)
        right_eye_ratio = get_blinking_ratio([42, 43, 44, 45, 46, 47], landmarks)
        blinking_ratio = (left_eye_ratio + right_eye_ratio) / 2

        if blinking_ratio > 5.7:
            cv2.putText(frame, "BLINKING", (50, 150), font, 7, (255, 0, 0))

        left_eye_region = np.array([(landmarks.part(36).x, landmarks.part(36).y),
                                    (landmarks.part(37).x, landmarks.part(37).y),
                                    (landmarks.part(38).x, landmarks.part(38).y),
                                    (landmarks.part(39).x, landmarks.part(39).y),
                                    (landmarks.part(40).x, landmarks.part(40).y),
                                    (landmarks.part(41).x, landmarks.part(41).y)], np.int32)

        right_eye_region = np.array([(landmarks.part(42).x, landmarks.part(42).y),
                                    (landmarks.part(43).x, landmarks.part(43).y),
                                    (landmarks.part(44).x, landmarks.part(44).y),
                                    (landmarks.part(45).x, landmarks.part(45).y),
                                    (landmarks.part(46).x, landmarks.part(46).y),
                                    (landmarks.part(47).x, landmarks.part(47).y)], np.int32)

        height, width, _ = frame.shape
        mask = np.zeros((height, width), np.uint8)
        cv2.polylines(mask, [left_eye_region], True, 255, 2)
        cv2.fillPoly(mask, [left_eye_region], 255)
        left_eye = cv2.bitwise_and(gray, gray, mask=mask)

        height1, width1, _ = frame.shape
        mask1 = np.zeros((height1, width1), np.uint8)
        cv2.polylines(mask1, [right_eye_region], True, 255, 2)
        cv2.fillPoly(mask1, [right_eye_region], 255)
        right_eye = cv2.bitwise_and(gray, gray, mask=mask1)


        min_x = np.min(left_eye_region[:, 0])
        max_x = np.max(left_eye_region[:, 0])
        min_y = np.min(left_eye_region[:, 1])
        max_y = np.max(left_eye_region[:, 1])

        min_x1 = np.min(right_eye_region[:, 0])
        max_x1 = np.max(right_eye_region[:, 0])
        min_y1 = np.min(right_eye_region[:, 1])
        max_y1 = np.max(right_eye_region[:, 1])

        for w,h in irises:
            cv2.circle(frame, (w, h), 7, (0, 255, 0), 2)

        gray_eye_left = left_eye[min_y: max_y, min_x: max_x]
        gray_eye_right = right_eye[min_y1: max_y1, min_x1: max_x1]

        _, threshold_eye_left = cv2.threshold(gray_eye_left, 70, 255, cv2.THRESH_BINARY)
        try:
            threshold_eye_left = cv2.resize(threshold_eye_left, None, fx=5, fy=5)
        except:
            pass
        _, threshold_eye_right = cv2.threshold(gray_eye_right, 70, 255, cv2.THRESH_BINARY)
        threshold_eye_right = cv2.resize(threshold_eye_right, None, fx=5, fy=5)

        bl_left = cv2.GaussianBlur(threshold_eye_left,(7,7),0)
        bl_right = cv2.GaussianBlur(threshold_eye_right,(7,7),0)

        contours,hierachy=cv2.findContours(bl_left,cv2.RETR_TREE,cv2.CHAIN_APPROX_SIMPLE)
        contours1,hierarchy1 = cv2.findContours(bl_right, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

        contours = sorted(contours, key=lambda x: cv2.contourArea(x), reverse=True)

        for i in range(len(my_irises)-2):
            d.append(distance(np.asarray(my_irises[i+2]),np.asarray(my_irises[i])))
        a1=np.mean(d)
        mean.append(a1)


        for cnt in contours:
            if cv2.contourArea(cnt)<400 and cv2.contourArea(cnt)>300:
                a=cv2.contourArea(cnt)
                list123.append(a)

            cv2.drawContours(frame, [cnt], 0, (0, 0, 255), 3)

        cv2.imshow("Threshold left", threshold_eye_left)
        cv2.imshow("Threshold right", threshold_eye_right)
        cv2.imshow("Left eye", left_eye)
        cv2.imshow("Right eye", right_eye)

    cv2.imshow("Frame", frame)
    key = cv2.waitKey(1)
    if key == 27:
        break;

cap.release()
cv2.destroyAllWindows()
