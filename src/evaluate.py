import cv2
import numpy as np
import mediapipe as mp

# FaceMesh 5-point indices (standard landmarks from refine_landmarks=True)
IDX_LEFT_EYE    = 33
IDX_RIGHT_EYE   = 263
IDX_NOSE_TIP    = 1
IDX_MOUTH_LEFT  = 61
IDX_MOUTH_RIGHT = 291


def main():
    # 1. Load Haar cascade for face detection
    cascade_path = cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
    face_cascade = cv2.CascadeClassifier(cascade_path)
    
    if face_cascade.empty():
        raise RuntimeError(f"Failed to load cascade classifier: {cascade_path}")

    # 2. Initialize MediaPipe FaceMesh (refined landmarks → iris & mouth corners)
    mp_face_mesh = mp.solutions.face_mesh
    face_mesh = mp_face_mesh.FaceMesh(
        static_image_mode=False,
        max_num_faces=1,
        refine_landmarks=True,
        min_detection_confidence=0.5,
        min_tracking_confidence=0.5,
    )

    # 3. Open webcam
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        raise RuntimeError("Cannot open camera. Try changing index to 1 or 2.")

    print("Haar + FaceMesh 5-point landmarks demo. Press 'q' to quit.")

    while True:
        ret, frame = cap.read()
        if not ret:
            print("Failed to grab frame. Exiting...")
            break

        H, W = frame.shape[:2]
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        # Haar face detection
        faces = face_cascade.detectMultiScale(
            gray,
            scaleFactor=1.1,
            minNeighbors=5,
            minSize=(60, 60)
        )

        # Draw all detected Haar faces (green rectangle)
        for (x, y, w, h) in faces:
            cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)

        # FaceMesh processing (on full RGB frame)
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = face_mesh.process(rgb_frame)

        if results.multi_face_landmarks:
            landmarks = results.multi_face_landmarks[0].landmark

            # Extract the 5 key points
            idxs = [
                IDX_LEFT_EYE,
                IDX_RIGHT_EYE,
                IDX_NOSE_TIP,
                IDX_MOUTH_LEFT,
                IDX_MOUTH_RIGHT
            ]

            pts = []
            for i in idxs:
                p = landmarks[i]
                pts.append([p.x * W, p.y * H])

            kps = np.array(pts, dtype=np.float32)  # shape: (5, 2)

            # Enforce correct left/right ordering (just in case)
            if kps[0, 0] > kps[1, 0]:  # left eye should be left of right eye
                kps[[0, 1]] = kps[[1, 0]]
            if kps[3, 0] > kps[4, 0]:  # left mouth should be left of right mouth
                kps[[3, 4]] = kps[[4, 3]]

            # Draw the 5 points
            for (px, py) in kps.astype(int):
                cv2.circle(frame, (int(px), int(py)), 4, (0, 255, 0), -1)

            cv2.putText(
                frame,
                "5pt FaceMesh",
                (10, 30),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.9,
                (0, 255, 0),
                2
            )

        # Show result
        cv2.imshow("Haar + 5pt FaceMesh Landmarks", frame)

        if cv2.waitKey(1) & 0xFF == ord("q"):
            break

    # Cleanup
    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()