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

class faceNet:
    def __init__(self, embeddingsPath=None):
        self.device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
        self.model = InceptionResnetV1(
                classify=False,
                pretrained="casia-webface"
            ).to(self.device)
        
        self.model.eval()
        self.embeddingsPath = embeddingsPath
        if embeddingsPath is not None:
            self.faceEmbeddings, self.faceNames = self.load_faceslist(embeddingsPath)       
        self.power = pow(10, 6)

    def process(self, face, bbox=None):
        if bbox is not None: 
            face = self.extract_face(bbox, face)
            
        face = Image.fromarray(face)
        idx, score = self.inference(self.model, face, self.faceEmbeddings)

        if idx != -1:
            score = torch.Tensor.cpu(score[0]).detach().numpy()*self.power
            name = self.faceNames[idx]
        else:
            name = "Unknown"

        return [name, idx, score] 


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
        names = np.load(DATA_PATH+'/usernames.npy')
        return embeds, names
    
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
        # print(norm_diff)
        norm_score = torch.sum(torch.pow(norm_diff, 2), dim=1) #(1,n), moi cot la tong khoang cach euclide so vs embed moi
        
        min_dist, embed_idx = torch.min(norm_score, dim = 1)
        print(min_dist*self.power, self.faceNames[embed_idx])
        # print(min_dist.shape)
        if min_dist*self.power > threshold:
            return -1, -1
        else:
            return embed_idx, min_dist.double()
        
    def face2vec(self,face):
        face = Image.fromarray(face)
        vec = self.model(self.trans(face).to(self.device).unsqueeze(0))
        #print("vec.shape:",vec.shape)
        return vec

    def register(self, face, name, get_vector= False):
        vec = self.face2vec(face)

        self.faceEmbeddings = torch.cat((self.faceEmbeddings,vec),0)
        self.faceNames = np.append(self.faceNames, name)
        print("faceEmbeddings.shape:",self.faceEmbeddings.shape)
        print("faceNames.shape:",self.faceNames.shape)

        embeddingPath = "outs/data/faceEmbedings"
        device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
        if device == 'cpu':
            torch.save(self.faceEmbeddings, embeddingPath+"/faceslistCPU.pth")
        else:
            torch.save(self.faceEmbeddings, embeddingPath+"/faceslist.pth")
        np.save(embeddingPath+"/usernames", self.faceNames)
        print('Update Completed! There are {0} people in FaceLists'.format(self.faceNames.shape[0]))

        if get_vector:
            return vec
               
    def updateFace(self, face, name, get_vector= False):
        out = np.where(self.faceNames == name)[0]
        if len(out) < 1:
            return
        vec = self.face2vec(face)

        self.faceEmbeddings = torch.cat((self.faceEmbeddings,vec),0)
        self.faceNames = np.append(self.faceNames, name)
        print("faceEmbeddings.shape:",self.faceEmbeddings.shape)
        print("faceNames.shape:",self.faceNames.shape)

        embeddingPath = "outs/data/faceEmbedings"
        device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
        if device == 'cpu':
            torch.save(self.faceEmbeddings, embeddingPath+"/faceslistCPU.pth")
        else:
            torch.save(self.faceEmbeddings, embeddingPath+"/faceslist.pth")
        np.save(embeddingPath+"/usernames", self.faceNames)
        print('Update Completed! There are {0} people in FaceLists'.format(self.faceNames.shape[0]))

        if get_vector:
            return vec
        

        
    def update_faceEmbeddings(self,imagePath, embeddingPath):
        embeddings = []
        names = []
        device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

        for usr in os.listdir(imagePath):
            embeds = []
            for file in glob.glob(os.path.join(imagePath, usr)+'/*.jpg'):
                # print(usr)
                try:
                    img = Image.open(file)
                except:
                    continue
                with torch.no_grad():
                    # print('smt')
                    embeds.append(self.model(self.trans(img).to(device).unsqueeze(0))) #1 anh, kich thuoc [1,512]
            if len(embeds) == 0:
                continue
            embedding = torch.cat(embeds).mean(0, keepdim=True) #dua ra trung binh cua 30 anh, kich thuoc [1,512]
            embeddings.append(embedding) # 1 cai list n cai [1,512]
            # print(embedding)
            names.append(usr)
            
        embeddings = torch.cat(embeddings) #[n,512]
        names = np.array(names)

        if device == 'cpu':
            torch.save(embeddings, embeddingPath+"/faceslistCPU.pth")
        else:
            torch.save(embeddings, embeddingPath+"/faceslist.pth")
        np.save(embeddingPath+"/usernames", names)
        print('Update Completed! There are {0} people in FaceLists'.format(names.shape[0]))
        

def test_recognition():
    prev_frame_time = 0
    new_frame_time = 0
    power = pow(10, 6)
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

    mtcnn = MTCNN(thresholds= [0.7, 0.7, 0.8] ,keep_all=True, device = device)
    faceRecognition = faceNet("outs/data/faceEmbedings")

    video = "video/faceRecognition.mp4"
    cap = cv2.VideoCapture(video)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH,640)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT,480)
    frame_idx = 0
    while cap.isOpened():
        isSuccess, frame = cap.read()
        if isSuccess:
            frame_idx +=1
            boxes, _ = mtcnn.detect(frame)
            if boxes is not None:
                for box in boxes:
                    bbox = list(map(int,box.tolist()))
                    face = faceRecognition.extract_face(bbox, frame)
                    if face is None:
                        continue
                    faceRecognition.updateFace(face, "tuan")
                    continue
                    idx , score, name = faceRecognition.process(face)
                    if idx != -1:
                        frame = cv2.rectangle(frame, (bbox[0],bbox[1]), (bbox[2],bbox[3]), (0,0,255), 6)
                        score = torch.Tensor.cpu(score[0]).detach().numpy()*power
                        frame = cv2.putText(frame, name + '_{:.2f}'.format(score), (bbox[0],bbox[1]), cv2.FONT_HERSHEY_DUPLEX, 2, (0,255,0), 2, cv2.LINE_8)
                    else:
                        frame = cv2.rectangle(frame, (bbox[0],bbox[1]), (bbox[2],bbox[3]), (0,0,255), 6)
                        frame = cv2.putText(frame,'Unknown', (bbox[0],bbox[1]), cv2.FONT_HERSHEY_DUPLEX, 2, (0,255,0), 2, cv2.LINE_8)

            new_frame_time = time.time()
            fps = 1/(new_frame_time-prev_frame_time)
            prev_frame_time = new_frame_time
            fps = str(int(fps))
            cv2.putText(frame, fps, (7, 70), cv2.FONT_HERSHEY_DUPLEX, 3, (100, 255, 0), 3, cv2.LINE_AA)

        image_name = "outs/images/image" + '{:05}'.format(frame_idx) + ".jpg"
        #image_name = "outs/check/check.jpg"
        cv2.imwrite(image_name, frame)
    cap.release()

def updateFaceEmbeddings():
    faceRecognition = faceNet()
    faceimages = "FaceNet-Infer/data/test_images"
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
    name_face = "duy"
    facespath = "outs/faces/000002.jpg"
    facespath = "images/test2.jpg"
    img = cv2.imread(facespath)

    embeddingPath = "outs/data/faceEmbedings"
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    faceRecognition = faceNet(embeddingPath)

    out = faceRecognition.process(img)
    print("out:",out)

    print("faceRecognition.faceNames:", faceRecognition.faceNames)
    print("All index value of 3 is: ", np.where(faceRecognition.faceNames == "duy")[0])

    out = np.where(faceRecognition.faceNames == "duy")[0]
    print("type(out):",type(out))
    print("out:",out[0])

    print("faceRecognition.faceEmbeddings:",faceRecognition.faceEmbeddings[out[0]].size())
    vec = faceRecognition.face2vec(img)
    print("vec.size = ", vec[0].size())
    with torch.no_grad():
        faceRecognition.faceEmbeddings[out[0]] = vec[0]

    out = faceRecognition.process(img)
    print("out:",out)

    #'''
    if name_face in faceRecognition.faceNames:
        print("Name exist")
        return
    faceRecognition.register(img, name_face)


    if device == 'cpu':
        torch.save(faceRecognition.faceEmbeddings, embeddingPath+"/faceslistCPU.pth")
    else:
        torch.save(faceRecognition.faceEmbeddings, embeddingPath+"/faceslist.pth")
    np.save(embeddingPath+"/usernames", faceRecognition.faceNames)
    print('Update Completed! There are {0} people in FaceLists'.format(faceRecognition.faceNames.shape[0]))
    #'''

    
if __name__ == "__main__":
    #test_recognition()
    #updateFaceEmbeddings()
    #test_facedetection()
    test_facerecognition()
