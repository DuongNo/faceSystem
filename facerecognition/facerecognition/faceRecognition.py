import os
import cv2
from PIL import Image
import torch
from torchvision import transforms
import numpy as np
import glob
from datetime import datetime
import json
import math
import torch.backends.cudnn as cudnn
import copy
import time
from pathlib import Path
from collections import Counter
from ultralytics import YOLO

from .InsightFace_Pytorch.config import get_config
from .InsightFace_Pytorch.Learner import face_learner
from .InsightFace_Pytorch.utils import load_facebank, prepare_facebank

class faceRecogner:
    def __init__(self, embeddingsPath=None, clearInfo=False):
        self.device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
        print("device:",self.device)
        
        self.conf = get_config(False)
        self.learner = face_learner(self.conf, True)
        if self.conf.device.type == 'cpu':
            self.learner.load_state(self.conf, 'cpu_final.pth', True, True)
        else:
            self.learner.load_state(self.conf, 'ir_se50.pth', False, True)
        self.learner.model.eval()
        print('learner loaded')
        
        #update = False
        #if update:
        #    self.targets, self.names = prepare_facebank(self.conf, self.learner.model, mtcnn, False)
        #    print('facebank updated')
        #else:
        #    self.targets, self.names = load_facebank(self.conf)
        #    print('facebank loaded')
        
        self.model = None
        self.embeddingsPath = embeddingsPath
        
        if clearInfo:
            self.employee_ids = np.empty(shape=[0])
            self.faceNames = np.empty(shape=[0])
            self.faceEmbeddings = []
            self.saveInfo()
            print("Clear Info")
        else:   
            self.employee_ids, self.faceEmbeddings, self.faceNames = self.load_faceslist(embeddingsPath)   
            print("Load Info:",len(self.employee_ids))    
        self.power = pow(10, 6)
        print("len of self.faceNames:",len(self.faceNames))
        
        self.detector = YOLO('/home/vdc/project/computervision/python/VMS/faceprocess/faceSystem/weights/yolov8l_face.pt')

    def process(self, face, bbox=None):
        if bbox is not None: 
            face = self.extract_face(bbox, face)
            
        face = Image.fromarray(face)
        faces = [face]
        results, score = self.learner.infer(self.conf, faces, self.faceEmbeddings, True)
        
        if results[0] == -1:
            name = "Unknown"
            employee_id = -1
        else:
            name = self.faceNames[results[0]]
            employee_id = self.employee_ids[results[0]]
        
        return [name, employee_id, score] 

    def extract_face(self,box, img, margin=20, crop_size = 112):
        face_size = crop_size
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
        embeds = torch.load(DATA_PATH+'/faceslist.pth')
        names = np.load(DATA_PATH+'/usernames.npy', allow_pickle=True)
        employee_ids = np.load(DATA_PATH+'/employee_ids.npy', allow_pickle=True)
        return employee_ids, embeds, names
    
    def clearInfo(self):
        self.employee_ids = np.empty(shape=[0])
        self.faceNames = np.empty(shape=[0])
        self.faceEmbeddings = []
    
    def trans(self,img):
        transform = transforms.Compose([
                transforms.ToTensor(),
                fixed_image_standardization
            ])
        return transform(img)
    
    def inference(self, model, face, local_embeds, threshold = 1):
        #local: [n,512] voi n la so nguoi trong faceslist
        embeds = []
        embeds.append(model(self.trans(face).to(self.device).unsqueeze(0).permute(0, 3, 2, 1)))
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
        vec = self.learner.model(self.conf.test_transform(face).to(self.conf.device).unsqueeze(0))
        #print("vec.shape:",vec.shape)
        return vec
    
    def detect(self, image):
        results = self.detector(image)  # predict on an image
        if results is not None: 
            boxes = results[0].boxes.xyxy.cpu()
            clss = results[0].boxes.cls.cpu().tolist()
            confs = results[0].boxes.conf.float().cpu().tolist()
            
            detections = []
            for box, cls_name, conf in zip(boxes, clss, confs):
                bbox = list(map(int,box.tolist()))
                cls_name = int(cls_name)
                #x1, y1, x2, y2 = bbox
                #im = cv2.rectangle(frame, (int(x1), int(y1)), (int(x2), int(y2)), (0,255,0), 3)
                #im = cv2.resize(im, (960,720), interpolation = cv2.INTER_LINEAR)
                #cv2.imshow("frame detect",im)
                
                if conf > 0.3:
                    detections.append([bbox[0],bbox[1], bbox[2],bbox[3]])
        return detections

    def register(self, face, name, id, get_vector= False):
        out = np.where(self.employee_ids == id)[0]
        if len(out) > 0:
            print("employee_ids: {} was exist:".format(id))
            return 1
            
        boxes = self.detect(face)
        print("boxes register:",boxes)
        if boxes is None:
            print("Dont have any faces")
            return 1, None
        
        bboxOK = False
        for box in boxes:
            if box[2] - box[0] > 50 and box[3] - box[1] > 50:
                face = self.extract_face(box, face)
                vec = self.face2vec(face)
                bboxOK = True
                break
        if not bboxOK:
            print("boxes size too small:",boxes)
            return 1, None

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
        out = out[0]
        
        boxes = self.detect(face)
        print("boxes updateFace:",boxes)
        if boxes is None:
            print("Dont have any faces")
            return 1, None
        bboxOK = False
        for box in boxes:
            if box[2] - box[0] > 80 and box[3] - box[1] > 80:
                face = self.extract_face(box, face)
                vec = self.face2vec(face)
                bboxOK = True
                break
        if not bboxOK:
            print("boxes size too small:",boxes)
            return 1, None

        self.faceNames[out] = name
        with torch.no_grad():
            self.faceEmbeddings[out] = vec
            
        self.saveInfo()
        print("Update Face Completed!")

        if get_vector:
            with torch.no_grad():
                vec = vec.to("cpu").numpy()
            return 0, json.dumps(vec.tolist())
        
    def removeId(self, id):
        out = np.where(self.employee_ids == id)[0]
        print("employee_ids:",out)
        if len(out) < 1:
            return 1
        out = out[0]
        
        self.employee_ids = np.delete(self.employee_ids,out)
        self.faceNames = np.delete(self.faceNames,out)
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
                    embeds.append(self.model(self.trans(face).to(device).unsqueeze(0).permute(0, 3, 2, 1))) #1 anh, kich thuoc [1,512]
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

        torch.save(self.faceEmbeddings, embeddingPath+"/faceslist.pth")
        np.save(embeddingPath+"/usernames", self.faceNames)
        np.save(embeddingPath+"/employee_ids", self.employee_ids)
        print('Update Completed! There are {0} people in FaceLists'.format(self.faceNames.shape[0]))
        



    
