import cv2

# Create a VideoCapture object to access the webcam (by default, it uses the first available camera)
cap = cv2.VideoCapture(0)

# Check if the camera opened successfully
if not cap.isOpened():
    print("Error: Could not open the webcam.")
    exit()

# Set the window name
cv2.namedWindow("Webcam", cv2.WINDOW_NORMAL)

while True:
    # Read a frame from the webcam
    ret, frame = cap.read()

    if not ret:
        print("Error: Could not read a frame from the webcam.")
        break

    # Display the frame in a window
    cv2.imshow("Webcam", frame)

    # Exit the loop if the 'q' key is pressed
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release the VideoCapture object and close the window
cap.release()
cv2.destroyAllWindows()
