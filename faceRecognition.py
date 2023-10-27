import os
import cv2
from PIL import Image
import torch
from torchvision import transforms
import numpy as np
from facenet_pytorch import MTCNN, InceptionResnetV1, fixed_image_standardization
import time
import glob
from datetime import datetime
import json
from torch.nn import CosineSimilarity
import math
from sklearn.metrics import pairwise


#load yolov7
from detector import Detector

import torch.backends.cudnn as cudnn
from numpy import random
import copy
import time
from pathlib import Path
from collections import Counter

from tracker import Tracker

class faceNet:
    def __init__(self, embeddingsPath=None):
        self.device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
        print("device:",self.device)
        self.model = InceptionResnetV1(
                classify=False,
                pretrained="casia-webface"
            ).to(self.device)
        
        self.detector = MTCNN(thresholds= [0.7, 0.7, 0.8] ,keep_all=True, device = self.device)
        self.model.eval()
        self.embeddingsPath = embeddingsPath
        if embeddingsPath is not None:
            self.employee_ids, self.faceEmbeddings, self.faceNames = self.load_faceslist(embeddingsPath)       
        self.power = pow(10, 6)
        #self.employee_ids = np.empty(shape=[0])
        #self.faceNames = np.empty(shape=[0])
        #self.faceEmbeddings = []

    def process(self, face, bbox=None):
        if bbox is not None: 
            face = self.extract_face(bbox, face)
            
        face = Image.fromarray(face)
        idx, score = self.inference(self.model, face, self.faceEmbeddings)

        if idx != -1:
            score = torch.Tensor.cpu(score[0]).detach().numpy()*self.power
            name = self.faceNames[idx]
            employee_id = self.employee_ids[idx]
        else:
            name = "Unknown"
            employee_id = -1

        return [name, employee_id, score] 


    def extract_face(self,box, img, margin=20):
        face_size = 160
        img_size = (img.shape[1], img.shape[0])
        margin = [
            margin * (box[2] - box[0]) / (face_size - margin),
            margin * (box[3] - box[1]) / (face_size - margin),
        ] #tạo margin bao quanh box cũ
        box = [
            int(max(box[0] - margin[0] / 2, 0)),
            int(max(box[1] - margin[1] / 2, 0)),
            int(min(box[2] + margin[0] / 2, img_size[0])),
            int(min(box[3] + margin[1] / 2, img_size[1])),
        ]
        
        img = img[box[1]:box[3], box[0]:box[2]]
        face = cv2.resize(img,(face_size, face_size), interpolation=cv2.INTER_AREA)
        #face = Image.fromarray(face)
        return face
    
    def load_faceslist(self, DATA_PATH):
        if self.device == 'cpu':
            embeds = torch.load(DATA_PATH+'/faceslistCPU.pth')
        else:
            embeds = torch.load(DATA_PATH+'/faceslist.pth')
        names = np.load(DATA_PATH+'/usernames.npy', allow_pickle=True)
        employee_ids = np.load(DATA_PATH+'/employee_ids.npy', allow_pickle=True)
        return employee_ids, embeds, names
    
    def trans(self,img):
        transform = transforms.Compose([
                transforms.ToTensor(),
                fixed_image_standardization
            ])
        return transform(img)
    
    def inference(self, model, face, local_embeds, threshold = 1):
        #local: [n,512] voi n la so nguoi trong faceslist
        embeds = []
        embeds.append(model(self.trans(face).to(self.device).unsqueeze(0)))
        detect_embeds = torch.cat(embeds) #[1,512]
        #print("local_embeds.shape:",local_embeds.shape)
        #print("detect_embeds.shape:",detect_embeds.shape)
                        #[1,512,1]                                      [1,512,n]
        norm_diff = detect_embeds.unsqueeze(-1) - torch.transpose(local_embeds, 0, 1).unsqueeze(0)
        #norm_diff = CosineSimilarity().forward(detect_embeds.unsqueeze(-1), torch.transpose(local_embeds, 0, 1).unsqueeze(0))
        #print("norm_diff:",norm_diff)
        norm_score = torch.sum(torch.pow(norm_diff, 2), dim=1) #(1,n), moi cot la tong khoang cach euclide so vs embed moi
        #print("norm_score:",norm_score)
        
        min_dist, embed_idx = torch.min(norm_score, dim = 1)
        #print(min_dist*self.power, self.faceNames[embed_idx])
        # print(min_dist.shape)
        if min_dist*self.power > threshold:
            return -1, -1
        else:
            return embed_idx, min_dist.double()

    def inference2(self, model, face, local_embeds, threshold = 1):
        embeds = []
        embeds.append(model(self.trans(face).to(self.device).unsqueeze(0)))
        detect_embeds = torch.cat(embeds) #[1,512]
        # norm_score = CosineSimilarity().forward(detect_embeds.unsqueeze(-1), torch.transpose(local_embeds, 0, 1).unsqueeze(0))
        # norm_diff = detect_embeds.unsqueeze(-1) - torch.transpose(local_embeds, 0, 1).unsqueeze(0)
        # norm_score = torch.sum(torch.pow(norm_diff, 2), dim=1) #(1,n)
        distances = []
        for local_embed in local_embeds:
            distances.append(self.distance(detect_embeds, local_embed, distance_metric=1))
            
        norm_score = torch.cat(distances).unsqueeze(0)
        min_dist, embed_idx = torch.min(norm_score, dim = 1)
        #print(min_dist*self.power, names[embed_idx])
        if min_dist*self.power > threshold:
            return -1, -1
        else:
            return embed_idx, (min_dist*self.power).double()
        
    def distance_origin(self, embeddings1, embeddings2, distance_metric=0):
        if distance_metric == 0:
            # Euclidian distance
            embeddings1 = embeddings1/np.linalg.norm(embeddings1, axis=1, keepdims=True)
            embeddings2 = embeddings2/np.linalg.norm(embeddings2, axis=1, keepdims=True)
            dist = np.sqrt(np.sum(np.square(np.subtract(embeddings1, embeddings2))))
            return dist      
        else:
            # Distance based on cosine similarity
            dot = np.sum(np.multiply(embeddings1, embeddings2), axis=1)
            norm = np.linalg.norm(embeddings1, axis=1) * np.linalg.norm(embeddings2, axis=1)
            similarity = dot/norm
            dist = np.arccos(similarity) / math.pi
            return dist[0]

    def distance2(self, embeddings1, embeddings2, distance_metric=0):
        embeddings1 = embeddings1.to("cpu")
        embeddings2 = embeddings2.to("cpu")    
        embeddings1 = embeddings1.detach().numpy()
        embeddings2 = embeddings2.detach().numpy()
        if distance_metric == 0:
            # Euclidian distance
            embeddings1 = embeddings1/np.linalg.norm(embeddings1, axis=1, keepdims=True)
            embeddings2 = embeddings2/np.linalg.norm(embeddings2, keepdims=True)
            dist = np.sqrt(np.sum(np.square(np.subtract(embeddings1, embeddings2))))
            return torch.from_numpy(np.array([dist]))
        else:
            # Distance based on cosine similarity
            dot = np.sum(np.multiply(embeddings1, embeddings2), axis=1)
            norm = np.linalg.norm(embeddings1, axis=1) * np.linalg.norm(embeddings2)
            similarity = dot/norm
            dist = np.arccos(similarity) / math.pi
            return torch.from_numpy(np.array([dist[0]]))
        
            
    def face2vec(self,face):
        face = Image.fromarray(face)
        vec = self.model(self.trans(face).to(self.device).unsqueeze(0))
        #print("vec.shape:",vec.shape)
        return vec

    def register(self, face, name, id, get_vector= False):    
        boxes, _ = self.detector.detect(face)
        if boxes is None:
            return 1, None
        for box in boxes:
            bbox = list(map(int,box.tolist()))
            face = self.extract_face(bbox, face)
            vec = self.face2vec(face)
            break

        #face = cv2.resize(face,(160, 160), interpolation=cv2.INTER_AREA)
        #vec = self.face2vec(face)

        if self.employee_ids.shape[0] > 0:
            self.faceEmbeddings = torch.cat((self.faceEmbeddings,vec),0)
        else:
            self.faceEmbeddings = vec
        self.faceNames = np.append(self.faceNames, name)
        self.employee_ids = np.append(self.employee_ids, id)
        print("faceEmbeddings.shape:",self.faceEmbeddings.shape)
        print("faceNames.shape:",self.faceNames.shape)
        print("employee_ids.shape:",self.employee_ids.shape)

        self.saveInfo()
        print('Update Completed! There are {0} people in FaceLists'.format(self.faceNames.shape[0]))

        if get_vector:
            with torch.no_grad():
                vec = vec.to("cpu").numpy()
            return 0, json.dumps(vec.tolist())
               
    def updateFace(self, face, name, id, get_vector= False):
        out = np.where(self.employee_ids == id)[0]
        if len(out) < 1:
            print("employee_ids was not exist:",id)
            return 1, None
        
        boxes, _ = self.detector.detect(face)
        if boxes is None:
            return 1, None
        for box in boxes:
            bbox = list(map(int,box.tolist()))
            face = self.extract_face(bbox, face)
            vec = self.face2vec(face)
            break

        vec = self.face2vec(face)
        with torch.no_grad():
            self.faceEmbeddings[out] = vec

        embeddingPath = "outs/data/faceEmbedings"
        device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
        if device == 'cpu':
            torch.save(self.faceEmbeddings, embeddingPath+"/faceslistCPU.pth")
        else:
            torch.save(self.faceEmbeddings, embeddingPath+"/faceslist.pth")

        self.saveInfo()
        print("Update Face Completed!")

        if get_vector:
            with torch.no_grad():
                vec = vec.to("cpu").numpy()
            return 0, json.dumps(vec.tolist())
        
    def removeId(self, id):
        out = np.where(self.employee_ids == id)[0]
        if len(out) < 1:
            return 1
        
        self.employee_ids = np.delete(self.employee_ids,out)
        self.faceNames = np.delete(self.faceNames,out)
        #self.faceEmbeddings = self.faceEmbeddings[self.faceEmbeddings!=self.faceEmbeddings[out]]
        self.faceEmbeddings = torch.cat([self.faceEmbeddings[:out], self.faceEmbeddings[out+1:]])
        
        self.saveInfo()
        print("faceEmbeddings.shape:",self.faceEmbeddings.shape)
        print("faceNames.shape:",self.faceNames.shape)
        print("employee_ids.shape:",self.employee_ids.shape)
        print('Update Completed! There are {0} people in FaceLists'.format(self.employee_ids.shape[0]))
        return 0
        

        
    def update_faceEmbeddings(self,imagePath, embeddingPath):
        embeddings = []
        names = []
        employee_ids = []
        device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

        idx = 0
        for usr in os.listdir(imagePath):
            idx +=1
            embeds = []
            for file in glob.glob(os.path.join(imagePath, usr)+'/*.jpg'):
                # print(usr)
                try:
                    img = Image.open(file)
                except:
                    continue
                img = np.array(img)
                img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
                boxes, _ = self.detector.detect(img)
                if boxes is None:
                    continue
                for box in boxes:
                    bbox = list(map(int,box.tolist()))
                    face = self.extract_face(bbox, img)
                face = Image.fromarray(face)
                with torch.no_grad():
                    # print('smt')
                    embeds.append(self.model(self.trans(face).to(device).unsqueeze(0))) #1 anh, kich thuoc [1,512]
            if len(embeds) == 0:
                continue
            embedding = torch.cat(embeds).mean(0, keepdim=True) #dua ra trung binh cua 30 anh, kich thuoc [1,512]
            embeddings.append(embedding) # 1 cai list n cai [1,512]
            # print(embedding)
            names.append(usr)
            employee_ids.append(str(idx))
            
        embeddings = torch.cat(embeddings) #[n,512]
        names = np.array(names)
        employee_ids = np.array(employee_ids)

        if device == 'cpu':
            torch.save(embeddings, embeddingPath+"/faceslistCPU.pth")
        else:
            torch.save(embeddings, embeddingPath+"/faceslist.pth")
        np.save(embeddingPath+"/usernames", names)
        np.save(embeddingPath+"/employee_ids", employee_ids)
        print('Update Completed! There are {0} people in FaceLists'.format(names.shape[0]))

    def saveInfo(self, _embeddingPath=None):
        embeddingPath = self.embeddingsPath
        if _embeddingPath is not None:
            embeddingPath = _embeddingPath

        device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
        if device == 'cpu':
            torch.save(self.faceEmbeddings, embeddingPath+"/faceslistCPU.pth")
        else:
            torch.save(self.faceEmbeddings, embeddingPath+"/faceslist.pth")
        np.save(embeddingPath+"/usernames", self.faceNames)
        np.save(embeddingPath+"/employee_ids", self.employee_ids)
        print('Update Completed! There are {0} people in FaceLists'.format(self.faceNames.shape[0]))
        

# DeepSORT -> Importing DeepSORT.
#from deep_sort.application_util import preprocessing
#from deep_sort.deep_sort import nn_matching
#from deep_sort.deep_sort.detection import Detection
#from deep_sort.deep_sort.tracker import Tracker
#from deep_sort.tools import generate_detections as gdet

def test_recognition():
    prev_frame_time = 0
    new_frame_time = 0
    power = pow(10, 6)
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

    detector = Detector(classes = [0])
    #model_path = "weights/yolov7/face_50epoch.pt"
    model_path = 'weights/yolov7-face/yolov7-face.pt' 
    detector.load_model(model_path)

    faceRecognition = faceNet("outs/data/faceEmbedings")
    tracker = Tracker()
    detection_threshold = 0.5
    colors = [(random.randint(0, 255), random.randint(0, 255), random.randint(0, 255)) for j in range(10)]


    size = (640, 480)  
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    writer = cv2.VideoWriter('track.mp4',fourcc, 25.0, (1280,960))

    video = "video/vlc-record.mp4"
    video = "video/face_video.mp4"
    cap = cv2.VideoCapture(video)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH,640)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT,480)
    frame_idx = 0
    while cap.isOpened():
        isSuccess, frame = cap.read()
        if isSuccess:
            frame_idx +=1
            yolo_dets = detector.detect(frame.copy())  
            if yolo_dets is not None:
                bbox = yolo_dets[:,:4]
                pros = yolo_dets[:,4]
                classes = yolo_dets[:,-1]
                num_objects = yolo_dets.shape[0]

                bboxes = []
                scores = []
                detections = []
                for box, score in zip(bbox, pros):
                    bbox = list(map(int,box.tolist()))
                    #frame = cv2.rectangle(frame, (bbox[0],bbox[1]), (bbox[2],bbox[3]), (0,0,255), 6)
                    #cv2.putText(frame, '{:.3f}'.format(score), (bbox[0],bbox[3]+20), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0,255,0), 1)

                    # DeepSORT -> Extracting Bounding boxes and its confidence scores.
                    if score > detection_threshold:
                        detections.append([bbox[0],bbox[1], bbox[2],bbox[3], score])

                tracker.update(frame, detections)

                
                for track in tracker.tracker.tracks:
                    if len(track.faces) > 0:
                        #print("track.face:",track.face)
                        name, idx , score,  = faceRecognition.process(frame,track.faces[0])
                        #track.names.append([name,round(score,3)])
                        track.names.append(name)
                        #print("track.names:",track.names)
                        #print('track ID {} : {}'.format(track.track_id,track.names))
                        counter = Counter(track.names)
                        most_common = counter.most_common()
                        print('track ID {} : {}'.format(track.track_id,most_common))
                        if len(track.names) >= 30:
                            #counter = Counter(track.names)
                            #most_common = counter.most_common()
                            #print('track ID {} : {}'.format(track.track_id,most_common))
                            if most_common[0][1] >  30:
                                track.name = most_common[0][0]
                                if track.face is None and name == most_common[0][0]:
                                    x1, y1, x2, y2 = track.faces[0]
                                    scale_x = int((x2 - x1)*0.3)
                                    scale_y = int((y2 - y1)*0.3)

                                    x1 = max(x1 - scale_x,0)
                                    y1 = max(y1 - scale_y,0)
                                    x2 = min(x2 + scale_x,frame.shape[1])
                                    y2 = min(y2 + scale_y,frame.shape[0])

                                    track.face = frame[y1:y2, x1:x2].copy()
                        track.faces = []
                
                for track in tracker.tracks:
                    bbox = track.bbox
                    x1, y1, x2, y2 = bbox
                    track_id = track.track_id

                    cv2.rectangle(frame, (int(x1), int(y1)), (int(x2), int(y2)), (colors[track_id % len(colors)]), 3)
                    txt = 'id:' + str(track.track_id) + "-" + track.name
                    org = (int(x1), int(y1)- 10)
                    cv2.putText(frame, txt, org, cv2.FONT_HERSHEY_SIMPLEX, 1.1, (0,0,255), 2)

                '''
                    face = faceRecognition.extract_face(bbox, frame)
                    name, idx , score,  = faceRecognition.process(face)
                    if idx != -1:
                        frame = cv2.rectangle(frame, (bbox[0],bbox[1]), (bbox[2],bbox[3]), (0,0,255), 6)
                        #score = torch.Tensor.cpu(score[0]).detach().numpy()*power
                        frame = cv2.putText(frame, name + '_{:.2f}'.format(score), (bbox[0],bbox[1]), cv2.FONT_HERSHEY_DUPLEX, 2, (0,255,0), 2, cv2.LINE_8)
                    else:
                        frame = cv2.rectangle(frame, (bbox[0],bbox[1]), (bbox[2],bbox[3]), (0,0,255), 6)
                        frame = cv2.putText(frame,'Unknown', (bbox[0],bbox[1]), cv2.FONT_HERSHEY_DUPLEX, 2, (0,255,0), 2, cv2.LINE_8)
                '''
                #frame = cv2.resize(frame, (1280,960), interpolation = cv2.INTER_LINEAR)
                #frame = cv2.resize(frame, (960,720), interpolation = cv2.INTER_LINEAR)
                #cv2.imshow("test",frame)
                #if cv2.waitKey(1) & 0xFF == ord('q'):
                #    break

                new_frame_time = time.time()
                fps = 1/(new_frame_time-prev_frame_time)
                prev_frame_time = new_frame_time
                fps = str(int(fps))
                cv2.putText(frame, fps, (7, 120), cv2.FONT_HERSHEY_DUPLEX, 2, (100, 255, 0), 3, cv2.LINE_AA)

            #image_name = "outs/track/image" + '{:05}'.format(frame_idx) + ".jpg"
            #image_name = "outs/check/check.jpg"
            #cv2.imwrite(image_name, frame)          
            #if frame_idx > 1300:
            #    break
        else:
            break
    cap.release()
    writer.release()

def updateFaceEmbeddings():
    faceRecognition = faceNet()
    faceimages = "data/faces_register/"
    embeddings = "outs/data/faceEmbedings"
    faceRecognition.update_faceEmbeddings(faceimages, embeddings)


def test_facedetection():
    images = "/home/vdc/project/computervison/python/face/faceSystem/images/vlc-record/000350.jpg"
    img = cv2.imread(images)

    outpath = "outs/faces"
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    mtcnn = MTCNN(thresholds= [0.7, 0.7, 0.8] ,keep_all=True, device = device)
    faceRecognition = faceNet("outs/data/faceEmbedings")
    
    boxes, _ = mtcnn.detect(img)
    if boxes is not None:
        for idx, box in enumerate(boxes):
            bbox = list(map(int,box.tolist()))
            face = faceRecognition.extract_face(bbox, img)
            img_name = f'{outpath}/{idx:06d}.jpg'
            cv2.imwrite(img_name,face)

def test_facerecognition():
    name_face = "girl"
    facespath = "outs/faces/000002.jpg"
    #facespath = "outs/tracks/1/00023.jpg"
    img = cv2.imread(facespath)

    embeddingPath = "outs/data/faceEmbedings"
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    faceRecognition = faceNet(embeddingPath)

    #faceRecognition.register(img, name_face, len(faceRecognition.faceNames)+10)

    out = faceRecognition.process(img)
    print("out:",out)
    #exit()

    '''
    #if name_face in faceRecognition.faceNames:
    #    print("Name exist")
    #    return
    #faceRecognition.register(img, name_face)


    if device == 'cpu':
        torch.save(faceRecognition.faceEmbeddings, embeddingPath+"/faceslistCPU.pth")
    else:
        torch.save(faceRecognition.faceEmbeddings, embeddingPath+"/faceslist.pth")
    np.save(embeddingPath+"/usernames", faceRecognition.faceNames)
    print('Update Completed! There are {0} people in FaceLists'.format(faceRecognition.faceNames.shape[0]))
    '''

from multiprocessing import Process

def f(name):
    while True:
        print('hello', name)
        time.sleep(1)


if __name__ == "__main__":
    test_recognition()
    #updateFaceEmbeddings()
    #test_facedetection()
    #test_facerecognition()
