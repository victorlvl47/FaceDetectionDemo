import cv2
import mediapipe as mp
import time

class FaceDetector():
    def __init__(self, minDetectionCon=0.5):
        self.minDetectionCon = minDetectionCon
        self.mpFaceDetection = mp.solutions.face_detection
        self.mpDraw = mp.solutions.drawing_utils
        self.faceDetection = self.mpFaceDetection.FaceDetection()

    def findFaces(self, img, draw=True):
        imgRGB = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        self.results = self.faceDetection.process(imgRGB)

        bboxs = []
        if self.results.detections:
            for id, detection in enumerate(self.results.detections):
                # Default function to draw the bounding box
                # mpDraw.draw_detection(img, detection)

                bboxC = detection.location_data.relative_bounding_box
                ih, iw, ic = img.shape
                bbox = int(bboxC.xmin * iw), int(bboxC.ymin * ih), \
                       int(bboxC.width * iw), int(bboxC.height * ih)
                bboxs.append([id, bbox, detection.score])
                cv2.rectangle(img, bbox, (0, 255, 0), 5)
                if draw: 
                    img = self.fancyDraw(img, bbox)
                    cv2.putText(img, 
                        f'{int(detection.score[0]*100)}%', 
                        (bbox[0], bbox[1]-20), 
                        cv2.FONT_HERSHEY_PLAIN, 8, (0, 255, 0), 10)

        return img, bboxs

    def fancyDraw(self, img, bbox, l=100, t=20, rt=1):
        x, y, w, h = bbox
        x1, y1, = x + w, y + h

        cv2.rectangle(img, bbox, (0, 255, 0), rt)
        # top left x,y
        cv2.line(img, (x, y), (x + l, y), (0, 255, 0), t)
        cv2.line(img, (x, y), (x, y + l), (0, 255, 0), t)
        # top right x1,y
        cv2.line(img, (x1, y), (x1 - l, y), (0, 255, 0), t)
        cv2.line(img, (x1, y), (x1, y + l), (0, 255, 0), t)
        # bottom left x,y
        cv2.line(img, (x, y1), (x + l, y1), (0, 255, 0), t)
        cv2.line(img, (x, y1), (x, y1 - l), (0, 255, 0), t)
        # bottom right x1,y1
        cv2.line(img, (x1, y1), (x1 - l, y1), (0, 255, 0), t)
        cv2.line(img, (x1, y1), (x1, y1 - l), (0, 255, 0), t)

        return img


def main():
    print("FaceDetectionModule")

    cap = cv2.VideoCapture("> PATH TO VIDEO HERE <")
    pTime = 0
    detector = FaceDetector()

    while True:
        success, img = cap.read()
        img, bboxs = detector.findFaces(img)
        
        cTime = time.time()
        fps = 1 / (cTime - pTime)
        pTime = cTime
        cv2.putText(img, f'FPS: {int(fps)}', (20, 140), 
                cv2.FONT_HERSHEY_PLAIN, 10, (0, 255, 0), 20)

        # resize img
        resized_img = cv2.resize(img, (832, 624))

        cv2.imshow("Image", resized_img)
        
        cv2.waitKey(1)


if __name__ == "__main__":
    main()
