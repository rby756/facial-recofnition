import face_recognition
import cv2
import time
import pygame
from pygame import mixer
import time

def play_sound(sound_file):
    # Initialize Pygame mixer
    pygame.mixer.init()

    try:
        # Load the sound file
        pygame.mixer.music.load(sound_file)

        # Play the sound
        pygame.mixer.music.play()

        # Wait for the sound to finish playing
        while pygame.mixer.music.get_busy():
            time.sleep(1)

    except pygame.error as e:
        print(f"Error: {e}")

def encode_faces(image_path):
    # Load the image with face_recognition
    image = face_recognition.load_image_file(image_path)

    # Find face locations and encodings
    face_locations = face_recognition.face_locations(image)
    face_encodings = face_recognition.face_encodings(image, face_locations)

    return face_locations, face_encodings

def detect_faces(known_faces):
    # Flags to control sound playback
    person_detected = False
    unknown_detected = False
    person_consecutive_counter = 0
    unknown_consecutive_counter = 0
    last_recognized_face = None
    
    # Open a connection to the webcam (you may need to adjust the index)
    cap = cv2.VideoCapture(0)

    # Variables for fps calculation
    start_time = time.time()
    frame_count = 0

    while True:
        # Capture video frame-by-frame
        ret, frame = cap.read()

        # Resize the frame for faster processing
        small_frame = cv2.resize(frame, (0, 0), fx=0.25, fy=0.25)

        # Process every other frame to increase fps
        if frame_count % 3 == 0:
            # Find face locations and encodings in the current frame
            face_locations = face_recognition.face_locations(small_frame)
            face_encodings = face_recognition.face_encodings(small_frame, face_locations)

            # Loop through each face found in the frame
            for (top, right, bottom, left), face_encoding in zip(face_locations, face_encodings):
                # Check if the face matches any known faces
                matches = face_recognition.compare_faces(known_faces, face_encoding,tolerance=0.8)
                name = "Unknown"

                # If a match is found, use the name of the known face
                if True in matches:
                    first_match_index = matches.index(True)
                    name = f"person"
                    
                    
                    
                # Set flags based on detected faces
                if name == 'person' :
                    person_detected = True
                    unknown_detected = False
                    person_consecutive_counter +=1
                    print("person_consecutive_counter",person_consecutive_counter)
                elif name == 'Unknown' :
                    person_detected = False
                    unknown_detected = True
                    unknown_consecutive_counter +=1
                    print("unknown_consecutive_counter",unknown_consecutive_counter)
                else:
                    pass
                        
                 # Update the last recognized face
                last_recognized_face = name


                # Draw rectangles around the faces
                cv2.rectangle(frame, (left * 4, top * 4), (right * 4, bottom * 4), (0, 255, 0), 2)
                font = cv2.FONT_HERSHEY_DUPLEX
                cv2.putText(frame, name, (left * 4 + 6, bottom * 4 - 6), font, 0.5, (255, 255, 255), 1)

            # Play the appropriate sound based on the detected faces
        if person_detected and person_consecutive_counter > 2:
            print(name)
            play_sound('audios/person.mp3')
            person_consecutive_counter = 0
            time.sleep(2)
        elif unknown_detected and unknown_consecutive_counter > 2:
            print(name)
            play_sound('audios/unknown.mp3')
            unknown_consecutive_counter = 0
            time.sleep(2)
        else:
            pass
            
        # Calculate and display fps
        frame_count += 1
        if time.time() - start_time >= 1.0:
            fps = frame_count / (time.time() - start_time)
            print(f"FPS: {fps:.2f}")
            frame_count = 0
            start_time = time.time()

        # Display the frame with rectangles and names
        cv2.imshow('Faces Detected', frame)

        # Break the loop when 'q' key is pressed
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    # Release the webcam and close the window
    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    # Replace 'path/to/known_face.jpg' with the path to the known face image
    known_face_path = 'faces/reby_face.jpg'
    
    

    # Encode the known face
    known_face_locations, known_face_encodings = encode_faces(known_face_path)

    # Pass the known face encodings to the detection function
    detect_faces(known_face_encodings)
