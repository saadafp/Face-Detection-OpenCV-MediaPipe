# Face-Detection-OpenCV-MediaPipe
In this notebook I used openCV and Mediapipe to detect faces.

# Face Detection in Video using MediaPipe & OpenCV

This project detects human faces in a video using [MediaPipe's Face Detection](https://google.github.io/mediapipe/solutions/face_detection.html) and overlays detection annotations using OpenCV. The processed video is saved locally with bounding boxes and keypoints drawn around detected faces.

---

## 📽️Dataset 
selena gomez instagrame : selenagomez
google image : jeniffer Jennifer Lawrence

---

## ✅ Features

- Face detection using MediaPipe's lightweight face detection model.
- Real-time annotation with bounding boxes and keypoints.
- Saves the result video with overlaid detections.
- Frame-by-frame display with a quit option (`q`).

---

## 🛠️ Requirements

Install the following Python packages before running the code:

```bash
pip install opencv-python mediapipe
```

---

## 🗂️ File Structure

```plaintext
.
├── face_detection_video.py      # Main Python script
├── SelenaGomez_openCV.mp4       # Output video (generated after running)
└── selena gomez1.MP4            # Input video (local file path required)
```

---

## 🚀 How to Run

1. **Change the Input Video Path**  
   Update this line in the code to reflect your actual video file path:
   
2. **Run the Script**
   ```bash
   python face_detection_video.py
   ```

3. **View and Save Output**  
   The processed video with face annotations will be saved as:
   ```
   SelenaGomez_openCV.mp4
   ```

4. **Exit the Live Window**  
   Press **`q`** to stop the video display and save the output.

---

## 🔍 Sample Output (Visualization)

- Green bounding box for detected face.
- Blue keypoints such as eyes, nose, and mouth corners.

---

## 🧠 How it Works

1. Reads video frame-by-frame.
2. Converts each frame to RGB for MediaPipe compatibility.
3. Detects faces using the `FaceDetection` model (model 0: short-range).
4. Draws detected face bounding boxes and keypoints on the frame.
5. Writes the annotated frame to an output video.
6. Displays real-time preview with OpenCV.

---

## 📌 Notes

- Works best on high-quality and well-lit videos.
- `model_selection=0` is optimized for faces within 2 meters.
- Ensure the resolution `(640, 480)` in `cv2.VideoWriter()` matches the input video resolution or adjust it accordingly.

---

