import cv2
import mediapipe as mp
import time
import json
from datetime import datetime

def main():
    mp_face_mesh = mp.solutions.face_mesh
    mp_drawing = mp.solutions.drawing_utils
    mp_drawing_styles = mp.solutions.drawing_styles

    webcam = cv2.VideoCapture(0)
    if not webcam.isOpened():
        print("Error: Unable to open webcam.")
        return

    # Initialize FaceMesh model and drawing specifications
    face_mesh = mp_face_mesh.FaceMesh(
        max_num_faces=2,
        refine_landmarks=True,
        min_detection_confidence=0.7,
        min_tracking_confidence=0.6
    )
    drawing_spec_tesselation = mp_drawing_styles.get_default_face_mesh_tesselation_style()
    drawing_spec_contours = mp_drawing_styles.get_default_face_mesh_contours_style()
    drawing_spec_iris = mp_drawing_styles.get_default_face_mesh_iris_connections_style()

    landmarks_data = []
    frame_count = 0
    start_time = time.time()

    while True:
        ret, img = webcam.read()
        if not ret:
            break

        # Convert image to RGB
        rgb_image = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

        # Process image with FaceMesh
        results = face_mesh.process(rgb_image)

        # Convert image back to BGR for rendering
        img = cv2.cvtColor(rgb_image, cv2.COLOR_RGB2BGR)

        if results.multi_face_landmarks:
            for face_landmarks in results.multi_face_landmarks:
                # Draw face mesh annotations
                mp_drawing.draw_landmarks(
                    image=img,
                    landmark_list=face_landmarks,
                    connections=mp_face_mesh.FACEMESH_TESSELATION,
                    landmark_drawing_spec=None,
                    connection_drawing_spec=drawing_spec_tesselation
                )
                mp_drawing.draw_landmarks(
                    image=img,
                    landmark_list=face_landmarks,
                    connections=mp_face_mesh.FACEMESH_CONTOURS,
                    landmark_drawing_spec=None,
                    connection_drawing_spec=drawing_spec_contours
                )
                mp_drawing.draw_landmarks(
                    image=img,
                    landmark_list=face_landmarks,
                    connections=mp_face_mesh.FACEMESH_IRISES,
                    landmark_drawing_spec=None,
                    connection_drawing_spec=drawing_spec_iris
                )

                # Get the current timestamp
                current_timestamp = datetime.fromtimestamp(start_time + (frame_count / webcam.get(cv2.CAP_PROP_FPS))).strftime('%Y-%m-%d %H:%M:%S.%f')

                # Save face landmarks data along with the timestamp
                landmark_data = {
                    "frame": frame_count,
                    "timestamp": current_timestamp,
                    "landmarks": [
                        {"x": landmark.x, "y": landmark.y, "z": landmark.z}
                        for landmark in face_landmarks.landmark
                    ]
                }
                landmarks_data.append(landmark_data)

        cv2.imshow("Face Mesh", img)
        frame_count += 1

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    webcam.release()
    cv2.destroyAllWindows()

    # Save the landmarks data to a JSON file
    with open("landmarks_data.json", "w") as f:
        json.dump(landmarks_data, f, indent=2)

if __name__ == "__main__":
    main()