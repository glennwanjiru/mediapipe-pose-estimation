import cv2
import mediapipe as mp

def main():
    # Initialize MediaPipe Hands.
    mp_hands = mp.solutions.hands
    hands = mp_hands.Hands()
    mp_drawing = mp.solutions.drawing_utils  # Import drawing utilities from MediaPipe.

    # Initialize OpenCV webcam capture.
    cap = cv2.VideoCapture(0)
    while cap.isOpened():
        # Read each frame from the webcam.
        ret, frame = cap.read()
        if not ret:
            print("Ignoring empty camera frame.")
            continue

        # Flip the image horizontally for a later selfie-view display.
        frame = cv2.flip(frame, 1)

        # Convert the BGR image to RGB.
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

        # Process the image with MediaPipe Hands.
        results = hands.process(rgb_frame)

        # If hand landmarks are detected, draw them on the frame.
        if results.multi_hand_landmarks:
            for hand_landmarks in results.multi_hand_landmarks:
                # Draw hand landmarks.
                mp_drawing.draw_landmarks(frame, hand_landmarks, mp_hands.HAND_CONNECTIONS)

        # Display the annotated frame.
        cv2.imshow('Hand Tracking', frame)

        # Press 'q' to quit.
        if cv2.waitKey(10) & 0xFF == ord('q'):
            break

    # Release the webcam and close any OpenCV windows.
    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
