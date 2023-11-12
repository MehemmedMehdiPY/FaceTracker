import numpy as np
import json
import cv2
from PIL import Image
from mtcnn.mtcnn import MTCNN
import mediapipe as mp

class FaceTracker():
    def __init__(self):
        mp_face_mesh =  mp.solutions.face_mesh
        self.face_mesh = mp_face_mesh.FaceMesh()
        self.face_detector = MTCNN()
        self.frame_number = 0

    def __call__(self):
        video = cv2.VideoCapture(0)
        while True:
            
            ret, frame = video.read()
            
            if self.frame_number % 10 == 0:       
#                 print(frame)
                faces = self.face_detector.detect_faces(frame)
                
                try:
                    new_frame = self.get_face(frame, faces)
                    new_frame = self.get_landmarks(new_frame, faces[0])
                except:
                    new_frame = frame
                
            cv2.imshow('frame', new_frame)
            
            if cv2.waitKey(1) & 0xFF == ord('q'): 
                video.release() 
                # Destroy all the windows 
                cv2.destroyAllWindows() 
                break
            
            self.frame_number += 1

    def get_face(self, frame, faces):
        for index, face in enumerate(faces):

            x, y, width, height = face['box']

            start_coord = (x, y)
            end_coord = (x + width, y + height)

            new_frame = cv2.rectangle(frame, start_coord, end_coord, color = (0, 255, 0), thickness=2)
        return new_frame    

    def get_landmarks(self, frame, face):

        left_eye_coordinates = np.array(face['keypoints']['left_eye']).reshape(-1, 2)
        right_eye_coordinates = np.array(face['keypoints']['right_eye']).reshape(-1, 2)

        results = self.face_mesh.process(frame)
        landmarks = results.multi_face_landmarks

        height, width, channels = frame.shape

        num_landmarks = len(landmarks[0].landmark)
        
        points = []

        for i in range(num_landmarks):

            x = int(landmarks[0].landmark[i].x * width)
            y = int(landmarks[0].landmark[i].y * height)

            points.append([x, y])
           
        points = np.array(points)

        right_indexes = np.sqrt((points - right_eye_coordinates) ** 2).sum(axis = 1).argsort()[:10].tolist()
        left_indexes = np.sqrt((points - left_eye_coordinates) ** 2).sum(axis = 1).argsort()[:10].tolist()
        indexes = right_indexes + left_indexes

        for x, y in points[indexes]:
            new_frame = cv2.circle(frame, (x, y), radius = 1, color = (0, 0, 255), thickness = 1)

        return new_frame
