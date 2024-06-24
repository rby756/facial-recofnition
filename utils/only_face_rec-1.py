
import imutils
import time
from imutils import face_utils
import dlib
import cv2
import face_recognition 
import numpy as np
import traceback
import numpy as np
import os
import numpy as np
import traceback




# path imagenes folder
path_images = "faces"

# threshold 
threshold = 0.53


# In[12]:


def get_features(img,box):
    features = face_recognition.face_encodings(img,box)
    return features

def compare_faces(face_encodings,db_features,db_names):
    match_name = []
    names_temp = db_names
    Feats_temp = db_features  

    for face_encoding in face_encodings:
        dist = face_recognition.face_distance(Feats_temp,face_encoding)
        index = np.argmin(dist)
        if dist[index] <= threshold:
            
            match_name = match_name + [names_temp[index]]
        else:
            match_name = match_name + ["unknown"]
    return match_name, dist[index]


# In[13]:


class rec():
    def __init__(self):
        
        print("Creating DataBase ...")
        self.db_names, self.db_features = load_images_to_database()
        print("DataBase created ...")

    def recognize_face(self,im):
    
        box_faces = face_recognition.face_locations(im)
        actual_features = get_features(im,box_faces)

        # conditional in case no face is detected
        if  not box_faces:
            res = {
                'status':'ok',
                'faces':[],
                'names':[],
                'dist face':[]}
            return res
        else:
            if not self.db_names:
                res = {
                    'status':'ok',
                    'faces':box_faces,
                    'names':['unknow']*len(box_faces),
                    'dist face': face_dis}
                return res
            else:
                # (continued) extract features
                actual_features = get_features(im,box_faces)
                face_dis= compare_faces(actual_features,self.db_features,self.db_names)[1]
                # compare actual_features with those stored in the database
                match_names = compare_faces(actual_features,self.db_features,self.db_names)[0]

                # save
                res = {
                    'status':'ok',
                    'faces':box_faces,
                    'names':match_names,
                    'dist face': face_dis}
                return res


def bounding_box(img,box,match_name=[]):
    for i in np.arange(len(box)):
        y0,x1,y1,x0 = box[i]
        img = cv2.rectangle(img,
                      (x0,y0),
                      (x1,y1),
                      (0,255,0),3);
        if not match_name:
            continue
        else:
            cv2.putText(img, match_name[i], (x0, y0-10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0,255,0), 2)
    return img


# In[14]:


def load_images_to_database():
    list_images = os.listdir(path_images)
    # I filter the files that are not images
    list_images = [File for File in list_images if File.endswith(('.jpg','.jpeg','JPEG','.png'))]

    name = []
    Feats = []

    # image ingest
    for file_name in list_images:
        im = cv2.imread(path_images+os.sep+file_name)

        # I get the features of the face
        box_face = face_recognition.face_locations(im)
        
        feat = get_features(im,box_face)
        if len(feat)!=1:
            '''
            this means that there are no faces or there is more than one face
            '''
            continue
        else:
            
            # insert the new features to the database
            
            new_name = file_name.split(".")[0]
            new_name=new_name.split("_")[0]
            if new_name == "":
                continue
            name.append(new_name)
            if len(Feats)==0:
                Feats = np.frombuffer(feat[0], dtype=np.float64)
            else:
                Feats = np.vstack((Feats,np.frombuffer(feat[0], dtype=np.float64)))
    return name, Feats


# In[15]:


cap=cv2.VideoCapture(0)
recognizer=rec()
while True:
    (grabbed, image)=cap.read()
    if not grabbed:
        break
#     image=imutils.resize(image,720)
    star_time = time.time()
    ref = recognizer.recognize_face(image)
    print(ref)
    image = bounding_box(image,ref["faces"],ref["names"])

    end_time = time.time() - star_time    
    FPS = 1/end_time
    cv2.putText(image,f"FPS: {round(FPS,3)}",(10,50),cv2.FONT_HERSHEY_COMPLEX,1,(0,0,255),2)
    cv2.imshow("Detection",image)
    if cv2.waitKey(1) &0xFF == ord('q'):
        
            break
    
        
cap.release()
cv2.destroyAllWindows()

