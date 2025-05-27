# Libraries
import cv2
import mediapipe as mp

# Dataset : video
cap = cv2.VideoCapture("D:/AI/OpenCV/selena gomez1.MP4")

# Check if video opened correctly
if not cap.isOpened():
    print("Error: Cannot open video.")
    exit()

# Video writing
fps = cap.get(cv2.CAP_PROP_FPS)
width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
result_video = cv2.VideoWriter("SelenaGomez_blured.avi", cv2.VideoWriter_fourcc(*"XVID"), fps,(width, height))

#  Model : mediapipe
mp_face_detector = mp.solutions.face_detection

# Reading the video and Face Detection
with mp_face_detector.FaceDetection(model_selection=0, min_detection_confidence=0.5) as face_detection:
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break
        H,W,_ = frame.shape
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        output = face_detection.process(rgb_frame)

        if output.detections is not None:
            for detection in output.detections:
                location_data = detection.location_data
                bbox = location_data.relative_bounding_box

                x1, y1, w, h = bbox.xmin, bbox.ymin, bbox.width, bbox.height

                x1 = int(x1 * W)
                y1 = int(y1 * H)
                w = int(w * W)
                h = int(h * H)

                # blur faces
                frame[y1:y1 + h, x1:x1 + w, :] = cv2.blur(frame[y1:y1 + h, x1:x1 + w, :], (50, 50))

    

        result_video.write(frame)
        cv2.imshow("Face Detection", frame)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

# Release resources
cap.release()
result_video.release()
cv2.destroyAllWindows()