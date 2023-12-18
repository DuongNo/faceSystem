from facenet_pytorch import MTCNN
from fastapi import FastAPI,  UploadFile, File, Header
from fastapi.responses import FileResponse
from typing import Annotated
import os
from random import randint
import uuid

from faceRecognition import faceRecogner
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
from multiprocessing import Process, Value
import base64
from kafka_process import kafka_consummer
from videoCapture import video_capture
from hopenet import headPose
from producer_result import aiProcuder
from config import conf 

embeddings_path = conf.security.embeddings_path
print("embeddings_path:",embeddings_path)
faceReg = faceRecogner(embeddings_path)
processes = {"max":0}

def file2image(file):
    image = Image.open(io.BytesIO(file)).convert('RGB') 
    image = np.array(image)
    image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
    return image

WEB_SERVER = "http://172.16.50.91:8001/api/v1/attendance/attendance-daily"
def facerecognition(video_path, topic, bootstrap_servers, status):
    faceRecognition = faceRecogner("outs/data/faceEmbedings")
    tracker = Tracker()
    detection_threshold = 0.25
    colors = [(random.randint(0, 255), random.randint(0, 255), random.randint(0, 255)) for j in range(10)]

    detector = Detector(classes = [0])
    #model_path = 'weights/yolov7-face/yolov7-face.pt'
    model_path = 'weights/yolov7-face/yolov7-tiny.pt'  
    detector.load_model(model_path)
    
    model_path = "deep-head-pose/hopenet_alpha1.pkl"
    headpose = headPose(model_path)
    
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    writer = cv2.VideoWriter('track.mp4',fourcc, 25.0, (1280,960))

    #video_path1 = "video/vlc-record.mp4"
    #video_path = "rtsp://VDI1:Vdi123456789@172.16.9.254:554/MediaInput/264/Stream1"
    #cap = cv2.VideoCapture(video_path)
    topic = "cam1"
    topic_ai = "cam1_ai"
    bootstrap_servers = ["172.16.50.91:9092"]
    
    #aiprocuder = aiProcuder(topic_ai, bootstrap_servers)
    
    #if video_path is not None:
    #    cap = video_capture(video_path=video_path)
    #else:
    #    cap = video_capture(topic=topic, bootstrap_servers=bootstrap_servers)
    #cap = video_capture(topic=topic, bootstrap_servers=bootstrap_servers)
    cap = video_capture(video_path=video_path)
    frame_idx = 0 
    print("status.value:",status.value)
    isSuccess = True    
    while isSuccess and status.value == 1:
        isSuccess, frame = cap.getFrame()
        start_time = time.time()
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
                        #yaw_predicted, pitch_predicted, roll_predicted, _  = headpose.getHeadPose(track.faces[0], frame)                      
                        #if (yaw_predicted > 20 or yaw_predicted < -20) or (pitch_predicted > 20 or pitch_predicted < -20) or (roll_predicted > 20 or roll_predicted < -20):
                        #    track.faces = []
                        #    continue
                        
                        name, employee_id , score,  = faceRecognition.process(frame,track.faces[0])
                        #track.names.append([name,round(score,3)])
                        track.names.append(name)
                        counter = Counter(track.names)
                        most_common = counter.most_common()
                        print('track ID {} : {}'.format(track.track_id,most_common))
                        if len(track.names) >= 30:
                            counter = Counter(track.names)
                            most_common = counter.most_common()
                            print('track ID {} : {}'.format(track.track_id,most_common))
                            if most_common[0][1] >  30:
                                track.name = most_common[0][0]
                                track.employee_id = employee_id

                                x1, y1, x2, y2 = track.faces[0]
                                scale_x = int((x2 - x1)*0.4)
                                scale_y = int((y2 - y1)*0.4)

                                x1 = max(x1 - scale_x,0)
                                y1 = max(y1 - scale_y,0)
                                x2 = min(x2 + scale_x,frame.shape[1])
                                y2 = min(y2 + scale_y,frame.shape[0])

                                track.face = frame[y1:y2, x1:x2].copy()
                                img_name = track.name + ".jpg"
                                cv2.imwrite(img_name,track.face)

                                
                                now = datetime.datetime.now()
                                retval, buffer = cv2.imencode('.jpg', track.face)
                                jpg_as_text = base64.b64encode(buffer)
                                #employer_id = "0c53feb4-44c3-4f0c-a2b3-31029477eb24"
                                employer_id = str(track.employee_id)
                                ret = requests.post(WEB_SERVER,json={"employer_id":employer_id,
                                                        "check_in":str(now),
                                                        "data":str(jpg_as_text)})
                                print("ret:",ret)
                                print('employee name {} and employee_ID {}'.format(track.name,track.employee_id))
                                
                        track.faces = []

                end_time = time.time()
                time_process = end_time-start_time
                fps = 1/time_process
                fps = "FPS: " + str(int(fps))
                #print("frame_idx: {}  time process: {}".format(frame_idx, time_process))
            
                show_result = False
                show_AIresult = False
                save_video = False
                if show_result or show_AIresult or save_video:
                    for track in tracker.tracks:
                        x1, y1, x2, y2 = track.bbox
                        track_id = track.track_id

                        cv2.rectangle(frame, (int(x1), int(y1)), (int(x2), int(y2)), (colors[track_id % len(colors)]), 3)
                        txt = 'id:' + str(track.track_id) + "-" + track.name
                        org = (int(x1), int(y1)- 10)
                        cv2.putText(frame, txt, org, cv2.FONT_HERSHEY_SIMPLEX, 1.1, (0,0,255), 2)

                    cv2.putText(frame, fps, (7, 120), cv2.FONT_HERSHEY_DUPLEX, 2, (100, 255, 0), 3, cv2.LINE_AA)
                    #print('FPS {} : number of tracks {}'.format(fps,len(tracker.tracks)))
                    
                    #if show_AIresult:
                        #aiprocuder.sendResult(frame, frame_idx)
                    
                    if save_video:
                        frame = cv2.resize(frame, (1280,960), interpolation = cv2.INTER_LINEAR)
                        writer.write(frame)
                        
                    #if show_result:
            #frame = cv2.resize(frame, (960,720), interpolation = cv2.INTER_LINEAR)
            #cv2.imshow("test",frame)
            #if cv2.waitKey(1) & 0xFF == ord('q'):
            #    break
        else:
            break
    #cap.release()
    writer.release()
    print("end process with :",video_path)
    

class Security:
    def __init__(self, conf):
        self.conf = conf
        
        self.faceRecognition = faceRecogner("outs/data/faceEmbedings")
        self.tracker = Tracker()
        self.detection_threshold = 0.25

        self.detector = Detector(classes = [0])
        #model_path = 'weights/yolov7-face/yolov7-face.pt'
        self.detector_path = 'weights/yolov7-face/yolov7-tiny.pt'  
        self.detector.load_model(self.detector_path)

        self.head_pose_path = "deep-head-pose/hopenet_alpha1.pkl"
        self.headpose = headPose(self.head_pose_path)

    def facerecognize(self, frame):
        yolo_dets = self.detector.detect(frame.copy())  
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
                if score > self.detection_threshold:
                    detections.append([bbox[0],bbox[1], bbox[2],bbox[3], score])

            self.tracker.update(frame, detections)

            
            for track in self.tracker.tracker.tracks:
                if len(track.faces) > 0:
                    #yaw_predicted, pitch_predicted, roll_predicted, _  = headpose.getHeadPose(track.faces[0], frame)                      
                    #if (yaw_predicted > 20 or yaw_predicted < -20) or (pitch_predicted > 20 or pitch_predicted < -20) or (roll_predicted > 20 or roll_predicted < -20):
                    #    track.faces = []
                    #    continue
                    
                    name, employee_id , score,  = self.faceRecognition.process(frame,track.faces[0])
                    #track.names.append([name,round(score,3)])
                    track.names.append(name)
                    counter = Counter(track.names)
                    most_common = counter.most_common()
                    print('track ID {} : {}'.format(track.track_id,most_common))
                    if len(track.names) >= 30:
                        counter = Counter(track.names)
                        most_common = counter.most_common()
                        print('track ID {} : {}'.format(track.track_id,most_common))
                        if most_common[0][1] >  30:
                            track.name = most_common[0][0]
                            track.employee_id = employee_id

                            x1, y1, x2, y2 = track.faces[0]
                            scale_x = int((x2 - x1)*0.4)
                            scale_y = int((y2 - y1)*0.4)

                            x1 = max(x1 - scale_x,0)
                            y1 = max(y1 - scale_y,0)
                            x2 = min(x2 + scale_x,frame.shape[1])
                            y2 = min(y2 + scale_y,frame.shape[0])

                            track.face = frame[y1:y2, x1:x2].copy()
                            #img_name = track.name + ".jpg"
                            #cv2.imwrite(img_name,track.face)

                            
                            now = datetime.datetime.now()
                            retval, buffer = cv2.imencode('.jpg', track.face)
                            jpg_as_text = base64.b64encode(buffer)
                            #employer_id = "0c53feb4-44c3-4f0c-a2b3-31029477eb24"
                            employer_id = str(track.employee_id)
                            ret = requests.post(WEB_SERVER,json={"employer_id":employer_id,
                                                    "check_in":str(now),
                                                    "data":str(jpg_as_text)})
                            print("ret:",ret)
                            print('employee name {} and employee_ID {}'.format(track.name,track.employee_id))                     
                    track.faces = []
        
            return self.tracker.tracks
        return None

class Processer:
    def __init__(self, conf):
        self.conf = conf
        self.security = Security(conf)      
 
    def process(self, frame):
        if conf.mode.ai_features.security:
            ret =  self.security.facerecognize(frame)
            
        elif conf.mode.ai_features.traffic:
            ret =  "traffic"
            
        return ret
    
    
def mainProcess(conf, video_path, topic, bootstrap_servers, status):
    print("Start process with status.value:",status.value)
    processer = Processer(conf)
    
    cap = video_capture(video_path=video_path)
    frame_idx = 0 
    isSuccess = True    
    colors = [(random.randint(0, 255), random.randint(0, 255), random.randint(0, 255)) for j in range(10)]
    while isSuccess and status.value == 1:
        isSuccess, frame = cap.getFrame()
        start_time = time.time()
        if isSuccess:
            print("frame_idx:",frame_idx)
            ret = processer.process(frame)
                 
        end_time = time.time()
        time_process = end_time-start_time
        fps = 1/time_process
        fps = "FPS: " + str(int(fps))
        print("frame_idx: {}  time process: {}  FPS: {}".format(frame_idx, time_process, fps))
        
        show_result = True
        show_AIresult = False
        save_video = False
        if show_result or show_AIresult or save_video:
            if ret is not None:
                for track in ret:
                    x1, y1, x2, y2 = track.bbox
                    track_id = track.track_id

                    cv2.rectangle(frame, (int(x1), int(y1)), (int(x2), int(y2)), (colors[track_id % len(colors)]), 3)
                    txt = 'id:' + str(track.track_id) + "-" + track.name
                    org = (int(x1), int(y1)- 10)
                    cv2.putText(frame, txt, org, cv2.FONT_HERSHEY_SIMPLEX, 1.1, (0,0,255), 2)

        #if show_AIresult:
            #aiprocuder.sendResult(frame, frame_idx)
        if show_result:  
            frame = cv2.resize(frame, (960,720), interpolation = cv2.INTER_LINEAR)
            cv2.imshow("test",frame)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
        
        frame_idx +=1
                     
    print("end process with :",video_path)
    

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

@app.post("/facerecognize_process")
def facerecognize_process(
                        camera_path: str = Header(None),
                        topic: str = Header(None),
                        bootstrap_servers: str = Header(None),
                        id_process: str = Header(None),
                        type_camera: str = Header(None),
                        event: str = Header(None)
                    ):  
    ret = 0
    #2 event: 
    #1- run_process  -  run process     - require: (camera_path)
    #2- shutdown     -  shutdown process   - require: (id_process)
    print("register process:")
    print("camera_path:",camera_path)
    print("topic:",topic)
    print("bootstrap_servers:",bootstrap_servers)
    print("event:",event)
    print("id_process:",id_process)

    #-- 1 need module check connect to camera_path
    #-- 2 need mode for camera AI

    if event == "run_process":
        status = Value('i', 0)
        status.value = 1
        p = Process(target=mainProcess, args=(camera_path, topic, bootstrap_servers, status))
        processes["max"] +=1
        id_process = str(processes["max"])
        processes[id_process] = status
        p.start()
    else:
        if id_process in processes.keys():
            processes[id_process].value = 0
            processes.pop(id_process)
        else:
            print("id_process {} were not exist".format(id_process))
            ret = -1
         
    return {"result": ret,
            "id_process": id_process}

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

    print("employee_id:",employee_id)
    print("employee_name:",employee_name)
    
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
        
    print("result:",result)
    return {"result": result,
            "employee_id": employee_id,
            "vector": vec}

    
@app.post("/get_status")
def get_status(
                        id_process: str = Header(None),
                        status: str = Header(None)
                    ): 
    # status: 
    #1- AI_process  -  status of AI process 
    print("get_status:",status) 
    ret = 0
    process_status = "OFF"
    
    if id_process in processes.keys():
        if processes[id_process].value == 1:
            process_status = "ON"

    return {"result": ret,
            "process_status": process_status}

def updateInfoFromdatabase():
    BACKEND_SERVER = 'http://172.16.50.91:8001/api/v1/employer/all/' 
    headers =   {
                    'accept':"application/json"                 
                }
    res = requests.get(BACKEND_SERVER,json={"headers":headers})
    print("res:",res)
    print("type(res:",type(res))
    data = res["data"]
    info = []
    for d in data:
        full_name = d["full_name"]
        id = d["id"]
        face_vec = d["face_vector"]
        info.append([id, full_name, face_vec])
        print("full_name: {} _____ id: {}".format(full_name,id))

 
if __name__ == "__main__":
    video = "video/face_video.mp4"
    #facerecognize_process(video)
    #updateInfoFromdatabase()
    
    status = Value('i', 0)
    status.value = 1
    
    mainProcess(conf, video, None, None, status)
    '''
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
    '''

