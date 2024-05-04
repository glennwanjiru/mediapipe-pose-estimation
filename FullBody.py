import cv2
import mediapipe as mp
import time
import json
from datetime import datetime

def main():
    # Initialize MediaPipe modules
    mp_face_mesh = mp.solutions.face_mesh
    mp_hands = mp.solutions.hands
    mp_pose = mp.solutions.pose
    mp_drawing = mp.solutions.drawing_utils
    mp_drawing_styles = mp.solutions.drawing_styles

    # Initialize FaceMesh model
    face_mesh = mp_face_mesh.FaceMesh(
        max_num_faces=2,
        refine_landmarks=True,
        min_detection_confidence=0.7,
        min_tracking_confidence=0.6
    )

    # Initialize Hands model
    hands = mp_hands.Hands()

    # Initialize Pose model
    pose = mp_pose.Pose()

    # Initialize webcam
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("Error: Unable to open webcam.")
        return

    # Define drawing styles for face mesh
    drawing_spec_tesselation = mp_drawing_styles.get_default_face_mesh_tesselation_style()
    drawing_spec_contours = mp_drawing_styles.get_default_face_mesh_contours_style()
    drawing_spec_iris = mp_drawing_styles.get_default_face_mesh_iris_connections_style()

    landmarks_data = []
    frame_count = 0
    start_time = time.time()

    while True:
        ret, img = cap.read()
        if not ret:
            break

        # Flip the image horizontally for a later selfie-view display.
        img = cv2.flip(img, 1)

        # Convert image to RGB
        rgb_image = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

        # Process image with FaceMesh
        face_results = face_mesh.process(rgb_image)

        # Process image with Hands
        hands_results = hands.process(rgb_image)

        # Process image with Pose
        pose_results = pose.process(rgb_image)

        # Convert image back to BGR for rendering
        img = cv2.cvtColor(rgb_image, cv2.COLOR_RGB2BGR)

        # Draw face mesh landmarks
        if face_results.multi_face_landmarks:
            for face_landmarks in face_results.multi_face_landmarks:
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

        # Draw hand landmarks
        if hands_results.multi_hand_landmarks:
            for hand_landmarks in hands_results.multi_hand_landmarks:
                mp_drawing.draw_landmarks(
                    image=img,
                    landmark_list=hand_landmarks,
                    connections=mp_hands.HAND_CONNECTIONS
                )

        # Draw pose landmarks and connections
        if pose_results.pose_landmarks:
            draw_landmarks(img, pose_results.pose_landmarks)

        # Display the annotated frame
        cv2.imshow('Combined Detection', img)
        frame_count += 1

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    # Release resources
    cap.release()
    cv2.destroyAllWindows()

def draw_landmarks(image, landmarks):
    # Draw all landmarks
    for landmark in landmarks.landmark:
        # Convert normalized landmark coordinates to pixel values
        h, w, c = image.shape  # Height, width, channels
        cx, cy = int(landmark.x * w), int(landmark.y * h)
        cv2.circle(image, (cx, cy), 5, (255, 0, 0), -1)  # Draw a blue circle

    # Define connections between all landmarks
    connections = mp.solutions.pose.POSE_CONNECTIONS

    # Draw connections between all landmarks
    for connection in connections:
        part_from = connection[0]
        part_to = connection[1]
        if landmarks.landmark[part_from].visibility > 0 and landmarks.landmark[part_to].visibility > 0:
            # Get pixel coordinates of landmarks
            x1, y1 = int(landmarks.landmark[part_from].x * w), int(landmarks.landmark[part_from].y * h)
            x2, y2 = int(landmarks.landmark[part_to].x * w), int(landmarks.landmark[part_to].y * h)
            cv2.line(image, (x1, y1), (x2, y2), (0, 255, 0), 2)  # Draw a green line

if __name__ == "__main__":
    main()
