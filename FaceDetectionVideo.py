# Libraries
import cv2
import argparse
import mediapipe as mp

args = argparse.ArgumentParser()
args.add_argument("--filePath", default="D:/AI/OpenCV/selena gomez1.MP4",
                  help="path to input image")
args = args.parse_args()

# Dataset : video
cap = cv2.VideoCapture(args.filePath)

# Check if video opened correctly
if not cap.isOpened():
    print("Error: Cannot open video.")
    exit()

# Video writing
fps = cap.get(cv2.CAP_PROP_FPS)
width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
result_video = cv2.VideoWriter("SelenaGomez_openCV.avi", cv2.VideoWriter_fourcc(*"XVID"), fps,(width, height))

#  Model : mediapipe
mp_face_detector = mp.solutions.face_detection
mp_drawing = mp.solutions.drawing_utils

# Reading the video and Face Detection
with mp_face_detector.FaceDetection(model_selection=0, min_detection_confidence=0.5) as face_detection:
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        output = face_detection.process(rgb_frame)

        if output.detections:
            for detection in output.detections:
                mp_drawing.draw_detection(
                    frame, detection,
                    mp_drawing.DrawingSpec(color=(0, 255, 0), thickness=2),
                    mp_drawing.DrawingSpec(color=(255, 0, 0), thickness=2)
                )

        result_video.write(frame)
        cv2.imshow("Face Detection", frame)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

# Release resources
cap.release()
result_video.release()
cv2.destroyAllWindows()