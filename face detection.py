# libraries
import cv2
import mediapipe as mp

# opening the original image
bgr_image = cv2.imread("D:\AI\OpenCV\dataset\Celebrity Faces Dataset\Jennifer Lawrence/001_21a7d5e6.jpg")


# converting bgr to rgb
rgb_image = cv2.cvtColor(bgr_image, cv2.COLOR_BGR2RGB)


# face_detection  module which includes a pre-trained face detector under hood
# it will return 1.confidentScore 2.bbox 3.keyFacialLandMarks
mp_face_detector = mp.solutions.face_detection

# also we can easly draw bbox and landmarks with draw_utils
mp_drawing = mp.solutions.drawing_utils


# we need to access FaceDetection class from mp_face_detector which uses a pre-trained ML model
with mp_face_detector.FaceDetection(model_selection=0, min_detection_confidence=0.5) as face_detection:
    output = face_detection.process(rgb_image)

    if output.detections:
        for detection in output.detections:
            mp_drawing.draw_detection(bgr_image, detection,
                                      mp_drawing.DrawingSpec(color=(0,255,0), thickness=2),
                                      mp_drawing.DrawingSpec(color=(255,0,0), thickness=2))

cv2.imshow("Face Detection", bgr_image)
cv2.imwrite("detected_face.jpg", bgr_image)
cv2.waitKey(0)
cv2.destroyAllWindows()