import cv2
import mediapipe as mp
import time

cap = cv2.VideoCapture("videos/1.mp4")
pTime = 0

while True:
    success, img = cap.read()

    cTime = time.time()
    fps = 1 / (cTime - pTime)
    pTime = cTime
    cv2.putText(img, f'FPS: {int(fps)}', (20, 140), 
            cv2.FONT_HERSHEY_PLAIN, 10, (0, 255, 0), 2)

    # resize img
    resized_img = cv2.resize(img, (640, 480))

    cv2.imshow("Image", resized_img)
    
    cv2.waitKey(1)
