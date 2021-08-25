import cv2
import mediapipe as mp
import time

cap = cv2.VideoCapture("videos/3.mp4")
pTime = 0

mpFaceDetection = mp.solutions.face_detection
mpDraw = mp.solutions.drawing_utils
faceDetection = mpFaceDetection.FaceDetection()

while True:
    success, img = cap.read()

    imgRGB = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    results = faceDetection.process(imgRGB)

    if results.detections:
        for id, detection in enumerate(results.detections):
            # Default function to draw the bounding box
            # mpDraw.draw_detection(img, detection)

            bboxC = detection.location_data.relative_bounding_box
            ih, iw, ic = img.shape
            bbox = int(bboxC.xmin * iw), int(bboxC.ymin * ih), \
                   int(bboxC.width * iw), int(bboxC.height * ih)
            cv2.rectangle(img, bbox, (0, 255, 0), 10)

    cTime = time.time()
    fps = 1 / (cTime - pTime)
    pTime = cTime
    cv2.putText(img, f'FPS: {int(fps)}', (20, 140), 
            cv2.FONT_HERSHEY_PLAIN, 10, (0, 255, 0), 20)

    # resize img
    resized_img = cv2.resize(img, (640, 480))

    cv2.imshow("Image", resized_img)
    
    cv2.waitKey(1)
