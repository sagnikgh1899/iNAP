import cv2
import numpy as np
import dlib
import imutils
from imutils import face_utils

cap = cv2.VideoCapture(0)
#path = 'face1.jpg'
#image_init = cv2.imread(path)


detector = dlib.get_frontal_face_detector()
predictor = dlib.shape_predictor("shape_predictor_68_face_landmarks.dat")

while (True):
    ret,frame = cap.read()
    #frame = image_init
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    overlay = frame.copy()

    faces = detector(gray)
    for face in faces:
        x1 = face.left()
        y1 = face.top()
        x2 = face.right()
        y2 = face.bottom()
        #cv2.rectangle(frame, (x1,y1), (x2,y2), (0,255,0), 3)

        landmarks = predictor(gray, face)
        for n in range(0,68):
            x = landmarks.part(n).x
            y = landmarks.part(n).y
            cv2.circle(frame, (x,y), 2, (255,0,0), -1)

        landmarks = face_utils.shape_to_np(landmarks)

        # Right Eye ROI Extraction
        (x1, y1, w1, h1) = cv2.boundingRect(np.array([landmarks[36:42]]))
        roi_reqd_1 = frame[y1:y1 + h1, x1:x1 + w1]
        roi_reqd_1 = imutils.resize(roi_reqd_1, width=250, inter=cv2.INTER_CUBIC)
        #overlay_1 = roi_reqd_1.copy()
        pts_1 = landmarks[36:42]
        hull_1 = cv2.convexHull(pts_1)
        cv2.drawContours(overlay, [hull_1], 0, (0, 255, 0), 2)
        #cv2.imshow("RIGHT EYE", overlay)
        #cv2.waitKey(0)


        # Left Eye ROI Extraction
        (x2, y2, w2, h2) = cv2.boundingRect(np.array([landmarks[42:48]]))
        roi_reqd_2 = frame[y2:y2 + h2, x2:x2 + w2]
        roi_reqd_2 = imutils.resize(roi_reqd_2, width=250, inter=cv2.INTER_CUBIC)
        #overlay = roi_reqd_2.copy()
        pts_2 = landmarks[42:48]
        hull_2 = cv2.convexHull(pts_2)
        cv2.drawContours(overlay, [hull_2], 0, (0, 255, 0), 2)
        cv2.imshow("LEFT EYE", overlay)
        #cv2.waitKey(0)

    key = cv2.waitKey(1)
    if key % 256 == 27:
        # ESC pressed
        print("Escape hit, closing...")
        break

#cv2.imshow("Frame",frame)
#cv2.waitKey(0)

cap.release()
cv2.destroyAllWindows()