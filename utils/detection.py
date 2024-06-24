import face_recognition.api
import cv2
import numpy as np
import sys
import math
import os 
import time

def face_confidence(face_distance,face_match_threshold=0.6):
    range=(1.0 - face_match_threshold)
    linear_val=(1.0 - face_distance)/(range * 2.0)

    if face_distance> face_match_threshold:
        return str(round(linear_val * 100 , 2)) + '%'
    
    else:

        value=(linear_val + ((1.0 - linear_val) * math.pow((linear_val - 0.5)*2 , 0.2)))*100
        return str(round(value,2))+'%'


class FaceRecognition:
    face_locations=[]
    face_encodings=[]
    face_names=[]
    known_face_encodings=[]
    known_face_names=[]
    process_current_frames=True


    def __init__(self):
        self.encode_faces()
        
    
    def encode_faces(self):
        for image in os.listdir('faces'):
            face_image=face_recognition.api.load_image_file(f"faces/{image}")
            face_encoding=face_recognition.api.face_encodings(face_image)

            self.known_face_encodings.append(face_encoding)
            self.known_face_names.append(image.split('.')[0].split("_")[0])



        # print(self.known_face_names)


    def run_recognition(self):
        video_capture=cv2.VideoCapture(0)

        if not video_capture.isOpened():
            sys.exit("video source is not found .....")

        while True:
            
            ret , frame = video_capture.read()
            # print(ret)
            if ret:
                small_frame=cv2.resize(frame,(0,0),fx=0.25,fy=0.25)
                rgb_small_frame = np.ascontiguousarray(small_frame[:, :, ::-1])

                #Find all the faces in current_frame
                
                self.face_locations=face_recognition.api.face_locations(rgb_small_frame)
                # self.face_locations=list(self.face_locations)
                print("face locations",self.face_locations)
                self.face_encodings=face_recognition.api.face_encodings(rgb_small_frame)
                

                self.face_names=[]
                # print(self.face_encodings)
                # print("self.face_encodings",len(self.face_encodings))
                # print("self.known_face_encodings",len(self.known_face_encodings))

                # print(self.face_encodings)
                for face_encoding in self.face_encodings:
                    matches=face_recognition.api.compare_faces(self.known_face_encodings,face_encoding)
                    name='Unknown'
                    confidence="NA"

                    face_distances = face_recognition.face_distance(self.known_face_encodings,face_encoding)
                    print(face_distances)
                    best_match_index=np.argmin(face_distances)

                    # print("checking len of best match index")
                    # print(len(face_distances[0]))
                    # print(len(face_distances[1]))     

                    print(matches)
                    print("seperation")
                    print(len(face_distances),best_match_index)


                    # if matches[best_match_index]:
                    #     name=self.known_face_names[best_match_index]
                    #     confidence=face_confidence(face_distances[best_match_index])

                    # If a match was found in known_face_encodings, just use the first one.
                    if True in matches:
                        first_match_index = matches.any().index(True)
                        print(first_match_index)
                        name = known_face_names[first_match_index]

                    self.face_names.append(f'{name} ({confidence})')

                # self.process_current_frames=not self.process_current_frames

                for (top,right,bottom,left),name in zip(self.face_locations,self.face_names):
                    top *= 4
                    right *= 4
                    bottom *= 4
                    left *= 4

                    cv2.rectangle(frame, (left,top), (right,bottom) ,(0,0,255),2)
                    cv2.rectangle(frame, (left,bottom-35), (right,bottom) ,(0,0,255),2)
                    cv2.putText(frame,name,(left+6,bottom-6),cv2.FONT_HERSHEY_COMPLEX,0.8,(234,234,123),1)
                
                
                cv2.imshow('Face Recognition',frame)
                # time.sleep(2)

                if cv2.waitKey(1) == ord('q'):
                    break
            else: 
                break
        video_capture.release()
        cv2.destroyAllWindows()





if __name__ == "__main__":
    fr=FaceRecognition()
    fr.run_recognition()







