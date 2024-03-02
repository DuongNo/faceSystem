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
from hopenet import headPose

from .InsightFace_Pytorch.config import get_config
from .InsightFace_Pytorch.Learner import face_learner
from .InsightFace_Pytorch.utils import load_facebank, prepare_facebank

class faceRecogner:
    def __init__(self, config, register=False):
        self.device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
        print("device:",self.device)
        
        self.config = config
        
        self.conf = get_config(False)
        self.learner = face_learner(self.conf, True)
        if self.conf.device.type == 'cpu':
            self.learner.load_state(self.conf, 'cpu_final.pth', True, True)
        else:
            self.learner.load_state(self.conf, 'ir_se50.pth', False, True)
        self.learner.model.eval()
        print('learner loaded')
    
        self.model = None
        self.embeddingsPath = self.config.embeddings_path
        
        self.employee_ids, self.faceEmbeddings, self.faceNames = self.load_faceslist(self.embeddingsPath)   
        print("Load Info:",len(self.employee_ids))    
        print("len of self.faceNames:",len(self.faceNames))
        self.power = pow(10, 6)
        
        if register:
            self.detector = YOLO(self.config.weights.faceDetection)
            
            self.head_pose_path = self.config.weights.headPose
            self.headpose = headPose(self.head_pose_path)

    def process(self, face, bbox=None):
        if bbox is not None: 
            face = self.extract_face(bbox, face)
            
        face = Image.fromarray(face)
        faces = [face]
        if len(self.faceEmbeddings) > 0:
            results, score = self.learner.infer(self.conf, faces, self.faceEmbeddings, True)
        else:
            return "Unknown", -1, -1
        
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
            print("Detected face:",boxes)
            
            detections = []
            for box, cls_name, conf in zip(boxes, clss, confs):
                bbox = list(map(int,box.tolist()))
                print("bbox:",bbox)
                cls_name = int(cls_name)
                #x1, y1, x2, y2 = bbox
                #im = cv2.rectangle(image, (int(x1), int(y1)), (int(x2), int(y2)), (0,255,0), 3)
                #im = cv2.resize(im, (960,720), interpolation = cv2.INTER_LINEAR)
                #cv2.imshow("frame detect",im)
                #cv2.waitKey(0)
                
                if conf > 0.3:
                    detections.append([bbox[0],bbox[1], bbox[2],bbox[3]])
        else:
            print("Detect can't find face")
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
            if box[2] - box[0] > 50 and box[3] - box[1] > 50:
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
            print("Can't find employee_ids:",id)
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
    
    def evaluate(self,imagePath, embeddingPath):
        employee_ids, faceEmbeddings, faceNames = self.load_faceslist(embeddingPath) 
        idx = 0
        correct = 0
        for name in os.listdir(imagePath):
            for file in glob.glob(os.path.join(imagePath, name)+'/*.png'):
                #print(name)
                try:
                    img = Image.open(file)
                    #img = cv2.resize(img, (112,112), interpolation = cv2.INTER_LINEAR)
                except:
                    continue 
                
                img = np.array(img)
                img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
                
                results = self.detector(img)
                if results is None:
                    continue           
                boxes = results[0].boxes.xyxy.cpu()
                if boxes.nelement() == 0:
                    print("empty boxes")
                    continue

                headposeOK = False
                for box in boxes:
                    bbox = list(map(int,box.tolist()))
                    face = self.extract_face(bbox, img)
                    
                    yaw_predicted, pitch_predicted, roll_predicted, _  = self.headpose.getHeadPose(bbox, img) 
                    if yaw_predicted is None or pitch_predicted is None or roll_predicted is None:
                        continue                  
                    if (yaw_predicted > 20 or yaw_predicted < -20) or (pitch_predicted > 20 or pitch_predicted < -20) or (roll_predicted > 20 or roll_predicted < -20):
                        continue
                    headposeOK = True        
                if not headposeOK:
                    continue
                #cv2.imshow("face",face)
                #cv2.waitKey(0)
                #return
                idx +=1
                face = Image.fromarray(face)
                faces = [face]
                results, score = self.learner.infer(self.conf, faces, faceEmbeddings, True)
                if results[0] == -1:
                    usr = "Unknown"
                else:
                    usr = faceNames[results[0]]
                
                if name == usr:
                    correct +=1
                print(f"accuracy = {correct}/{idx}")

                     
    def createFaceBank(self,imagePath, embeddingPath):
        embeddings = []
        names = []
        employee_ids = []
        device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

        idx = 0
        for usr in os.listdir(imagePath):
            idx +=1
            embeds = []
            for file in glob.glob(os.path.join(imagePath, usr)+'/*.png'):
                #print(usr)
                try:
                    img = Image.open(file)
                    #img = cv2.resize(img, (112,112), interpolation = cv2.INTER_LINEAR)
                except:
                    continue
                
                img = np.array(img)
                img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
                
                results = self.detector(img)
                if results is None:
                    print("empty result")
                    continue
                boxes = results[0].boxes.xyxy.cpu()
                print("boxes",boxes)
                if boxes.nelement() == 0:
                    print("empty boxes")
                    continue
                headposeOK = False
                for box in boxes:
                    bbox = list(map(int,box.tolist()))
                    face = self.extract_face(bbox, img)  
                                  
                    yaw_predicted, pitch_predicted, roll_predicted, _  = self.headpose.getHeadPose(bbox, img) 
                    if yaw_predicted is None or pitch_predicted is None or roll_predicted is None:
                        continue                  
                    if (yaw_predicted > 20 or yaw_predicted < -20) or (pitch_predicted > 20 or pitch_predicted < -20) or (roll_predicted > 20 or roll_predicted < -20):
                        continue
                    headposeOK = True        
                if not headposeOK:
                    continue
                face = Image.fromarray(face)
                with torch.no_grad():
                    #embeds.append(self.model(self.trans(face).to(device).unsqueeze(0).permute(0, 3, 2, 1))) #1 anh, kich thuoc [1,512]
                    embeds.append(self.learner.model(self.conf.test_transform(face).to(self.conf.device).unsqueeze(0)))
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
        
    def clearInfo(self):
        self.employee_ids = np.empty(shape=[0])
        self.faceNames = np.empty(shape=[0])
        self.faceEmbeddings = []   
        



    
