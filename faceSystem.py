from facenet_pytorch import MTCNN
from fastapi import FastAPI,  UploadFile, File, Header
from fastapi.responses import FileResponse
from typing import Annotated
import os
from random import randint
import uuid

from faceRecognition import faceNet
from detector import Detector
from tracker import Tracker
import cv2
import torch
import time
import io
import PIL.Image as Image
import numpy as np
from typing import List
from numpy import random
from collections import Counter
import requests
import datetime

from typing import Union
from pydantic import BaseModel
from multiprocessing import Process
import base64
from kafka_process import kafka_consummer

embeddings_path = "outs/data/faceEmbedings"
faceReg = faceNet(embeddings_path)

device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
power = pow(10, 6)
mtcnn = MTCNN(thresholds= [0.7, 0.7, 0.8] ,keep_all=True, device = device)

list_process = []

def file2image(file):
    image = Image.open(io.BytesIO(file)).convert('RGB') 
    image = np.array(image)
    image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
    return image 

def facerecognize(image):
    boxes, _ = mtcnn.detect(image)
    names = []
    if boxes is not None:
        for box in boxes:
            bbox = list(map(int,box.tolist()))
            name, idx , score = faceReg.process(image, bbox)
            names.append(name)

    return names

def facerecognize_process(video_path):
    prev_frame_time = 0
    new_frame_time = 0
    power = pow(10, 6)

    faceRecognition = faceNet("outs/data/faceEmbedings")
    tracker = Tracker()
    detection_threshold = 0.5
    colors = [(random.randint(0, 255), random.randint(0, 255), random.randint(0, 255)) for j in range(10)]

    detector = Detector(classes = [0])
    #model_path = 'weights/yolov7-face/yolov7-face.pt'
    model_path = 'weights/yolov7-face/yolov7-tiny.pt'  
    detector.load_model(model_path)

    #video_path1 = "video/vlc-record.mp4"
    cap = cv2.VideoCapture(video_path)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH,640)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT,480)
    frame_idx = 0   
    while cap.isOpened():
        isSuccess, frame = cap.read()
        if isSuccess:
            frame_idx +=1
            print("frame_idx:",frame_idx)
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

                    # DeepSORT -> Extracting Bounding boxes and its confidence scores.
                    if score > detection_threshold:
                        detections.append([bbox[0],bbox[1], bbox[2],bbox[3], score])

                tracker.update(frame, detections)

                
                for track in tracker.tracker.tracks:
                    if len(track.faces) > 0:
                        name, idx , score,  = faceRecognition.process(frame,track.faces[0])
                        #track.names.append([name,round(score,3)])
                        track.names.append(name)
                        #counter = Counter(track.names)
                        #most_common = counter.most_common()
                        #print('track ID {} : {}'.format(track.track_id,most_common))
                        if len(track.names) >= 30:
                            counter = Counter(track.names)
                            most_common = counter.most_common()
                            print('track ID {} : {}'.format(track.track_id,most_common))
                            if most_common[0][1] >  30:
                                track.name = most_common[0][0]
                                track.employee_id = idx
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
                    if track.track_id is not None and track.name is not None:
                        txt = 'id:' + str(track.track_id) + "-" + track.name
                    else:
                        txt = 'id:'
                    org = (int(x1), int(y1)- 10)
                    cv2.putText(frame, txt, org, cv2.FONT_HERSHEY_SIMPLEX, 1.1, (0,0,255), 2)

                new_frame_time = time.time()
                fps = 1/(new_frame_time-prev_frame_time)
                prev_frame_time = new_frame_time
                fps = "FPS: " + str(int(fps))
                #cv2.putText(frame, fps, (7, 120), cv2.FONT_HERSHEY_DUPLEX, 2, (100, 255, 0), 3, cv2.LINE_AA)
                #print('FPS {} : number of tracks {}'.format(fps,len(tracker.tracks)))
                #frame = cv2.resize(frame, (960,720), interpolation = cv2.INTER_LINEAR)
                #cv2.imshow("test",frame)
                #if cv2.waitKey(1) & 0xFF == ord('q'):
                #    break
        else:
            break
    cap.release()

class Item(BaseModel):
    name: str
    price: float

app = FastAPI()

@app.get("/")
async def check():
    return {"message": "Hello World"}

@app.post("/items/")
async def create_item(item: Item):
    print("item:",item)
    print("item:",item.name)
    return item

@app.post("/updateinfo")
async def update_info(
                        member_names: List[str] = Header(None),
                        member_ids: List[int] = Header(None),
                        vectors: List[float] = Header(None)
                    ):
    ret = 0

    return {"result": ret}

@app.post("/register")
async def register(
                        file: UploadFile,
                        employee_name: str = Header(None),
                        employee_id: str = Header(None),
                        event: str = Header(None)
                    ):
    #3 event: 
    #1- register     - require: (employee_name, employee_id)
    #2- updateface   - require: (employee_name, employee_id)
    #3- remove       - require: (employee_id)

    print("employee_id_check:",employee_id)
    
    vec = None
    if event == "register":
        contents = await file.read()
        image = file2image(contents)
        result, vec = faceReg.register(image, employee_name, employee_id, get_vector=True)
    elif event == "updateface":
        contents = await file.read()
        image = file2image(contents)
        result, vec = faceReg.updateFace(image, employee_name, employee_id, get_vector=True)
    else:
        result = faceReg.removeId(employee_id)
         
    return {"result": result,
            "employee_id": employee_id,
            "vector": vec}


@app.post("/facerecognize_process")
def register(
                        camera_path: str = Header(None),
                        id_process: str = Header(None),
                        type_camera: str = Header(None),
                        event: str = Header(None)
                    ):  
    ret = 0
    #2 event: 
    #1- run_process     - require: (camera_path)
    #2- shutdown_process   - require: (id_process)
    print("camera_path:",camera_path)
    print("event:",event)

    if event == "run_process":
        p = Process(target=facerecognize_process, args=(camera_path,))
        list_process.append(p)
        p.start()
        id_process = len(list_process) - 1
    else:
        id_process = int(id_process)
        list_process[id_process].kill()
        list_process[id_process].join()
        if not list_process[0].is_alive():
            print("process was killed")
        else:
            print("process still alive")
         
    return {"result": ret,
            "id_process": id_process}


@app.post("/facerecognition")
async def register(
                        file: UploadFile,
                        event: str = Header(None)
                    ):
    #3 event: 
    #1- process     - facedetection + facerecognition , require: image
    #2- facedetection   - require: image
    #3- facerecognition       - require: face_image
    
    ret = 0
    if event == "process":
        contents = await file.read()
        image = file2image(contents)
        names = facerecognize(image)
        print("names:",names)

    elif event == "facedetection":
        contents = await file.read()
        image = file2image(contents)
        
    elif event == "facerecognition":
        contents = await file.read()
        image = file2image(contents)
        names, idx, score = faceReg.process(image)
        print("names:",names)
         
    return {"result": ret,
            "names": names,
            "names": names}

@app.post("/face/")
async def create_upload_file(file: UploadFile = File(...)):
    contents = await file.read()
    image = file2image(contents)

    names, idx, score = faceReg.process(image)
    print("names:",names)
    return {"names": names}


def test_recognition():
    prev_frame_time = 0
    new_frame_time = 0
    power = pow(10, 6)

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
                    face = faceReg.extract_face(bbox, frame)
                    if face is None:
                        continue
                    idx , score, name = faceReg.process(face)
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

if __name__ == "__main__":
    #video = "video/face_video.mp4"
    #facerecognize_process(video)
    now = datetime.datetime.now()
    data = {
        "id": "",
        'employee_id':"0c53feb4-44c3-4f0c-a2b3-31029477eb24",
                    'check_ịn':str(now)}
    headers = {
                    'accept':"application/json"                 
                }

    face = cv2.imread("data/faces_register/bach/00190.jpg")
    retval, buffer = cv2.imencode('.jpg', face)
    jpg_as_text = base64.b64encode(buffer)

    #WEB_SERVER = "http://172.16.50.91:8001/api/v1/attendance/attendance-daily"
    WEB_SERVER = "http://172.16.9.151:8001/api/v1/attendance/attendance-daily"
    ret = requests.post(WEB_SERVER,json={"employer_id":"0c53feb4-44c3-4f0c-a2b3-31029477eb24",
                                         "check_in":str(now),
                                         "data":str(jpg_as_text)})
    print("ret:",ret)