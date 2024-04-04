import cv2
import dlib
from imutils import face_utils

# Initialize dlib's face detector and the facial landmark predictor
detector = dlib.get_frontal_face_detector()
predictor = dlib.shape_predictor("shape_predictor_68_face_landmarks.dat")

# Initialize video capture
cap = cv2.VideoCapture(0)

# Define the codec and create VideoWriter object
fourcc = cv2.VideoWriter_fourcc(*'XVID')
out = cv2.VideoWriter('cheeks_output.avi', fourcc, 30.0, (200, 100))  # Adjust the size as needed

# Variable to store the initial cheek regions
initial_cheek_left = None
initial_cheek_right = None

while cap.isOpened():
    ret, frame = cap.read()
    # frame = cv2.rotate(frame, cv2.ROTATE_180)
    if not ret:
        break

    # If the initial cheek regions are not defined, detect them
    if initial_cheek_left is None or initial_cheek_right is None:
        # Convert the frame to grayscale
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        # Detect faces in the grayscale frame
        rects = detector(gray, 0)

        for rect in rects:
            # Determine the facial landmarks for the face region
            shape = predictor(gray, rect)
            shape = face_utils.shape_to_np(shape)

            # Define the initial cheek regions
            initial_cheek_right = [shape[29][1], shape[33][1], shape[54][0], shape[12][0]]
            initial_cheek_left = [shape[29][1], shape[33][1], shape[4][0], shape[48][0]]
            break  # Only use the first detected face

    # If the initial regions have been defined, use them to extract cheeks
    if initial_cheek_left is not None and initial_cheek_right is not None:
        y1, y2, x1, x2 = initial_cheek_right
        cheek_right = frame[y1:y2, x1:x2]
        
        y1, y2, x1, x2 = initial_cheek_left
        cheek_left = frame[y1:y2, x1:x2]

        # Resize cheek images to have consistent output size, adjust as needed
        cheek_right = cv2.resize(cheek_right, (100, 100))
        cheek_left = cv2.resize(cheek_left, (100, 100))

        # Combine both cheek images horizontally
        cheeks = cv2.hconcat([cheek_left, cheek_right])

        # Write the frame
        out.write(cheeks)

        # Display the resulting frame (optional)
        cv2.imshow('Cheeks', cheeks)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

# Release everything if job is finished
cap.release()
out.release()
cv2.destroyAllWindows()
