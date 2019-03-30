import dlib
import numpy as np
import cv2

irises=[]

cap= cv2.VideoCapture(0)
detector=dlib.get_frontal_face_detector()
predictor = dlib.shape_predictor("shape_predictor_68_face_landmarks.dat")

font = cv2.FONT_HERSHEY_PLAIN

def get_irises_location(frame_gray):

    eye_cascade=cv2.CascadeClassifier(r'haarcascades/haarcascade_eye_tree_eyeglasses.xml')
    eyes = eye_cascade.detectMultiScale(frame_gray, 1.3, 5)  # if not empty - eyes detected

    irises=[]
    for (ex, ey, ew, eh) in eyes:
        iris_w = int(ex + float(ew / 2))
        iris_h = int(ey + float(eh / 2))
        irises.append([np.float32(iris_w), np.float32(iris_h)])
        #my_irises.append([np.float32(iris_w), np.float32(iris_h)])

    return irises


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

        gray_eye_left = left_eye[min_y: max_y, min_x: max_x]
        gray_eye_right = right_eye[min_y1: max_y1, min_x1: max_x1]

        _, threshold_eye_left = cv2.threshold(gray_eye_left, 70, 255, cv2.THRESH_BINARY)
        threshold_eye_left = cv2.resize(threshold_eye_left, None, fx=5, fy=5)

        _, threshold_eye_right = cv2.threshold(gray_eye_right, 70, 255, cv2.THRESH_BINARY)
        threshold_eye_right = cv2.resize(threshold_eye_right, None, fx=5, fy=5)

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
