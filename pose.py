import cv2
import mediapipe as mp

# Function to detect poses using MediaPipe
def detect_pose():
    # Initialize MediaPipe Pose model
    mp_pose = mp.solutions.pose
    pose = mp_pose.Pose()

    # Initialize webcam
    cap = cv2.VideoCapture(0)

    while cap.isOpened():
        # Read frame from webcam
        ret, frame = cap.read()
        if not ret:
            break

        # Convert the BGR image to RGB
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

        # Process the frame with MediaPipe Pose
        results = pose.process(rgb_frame)

        # If pose landmarks are detected
        if results.pose_landmarks:
            # Draw pose landmarks and connections on the frame
            draw_landmarks(frame, results.pose_landmarks)

        # Display the frame
        cv2.imshow('Pose Detection', frame)

        # Break the loop if 'q' is pressed
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    # Release resources
    cap.release()
    cv2.destroyAllWindows()
    

# Function to draw landmarks and connections on the frame using OpenCV
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
    detect_pose()
