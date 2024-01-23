from fastapi import FastAPI,  UploadFile, File, Header, Form
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
from multiprocessing import Process, Value, Array 
import base64
from kafka_process import kafka_consummer
from videoCapture import video_capture
from hopenet import headPose
from producer_result import ResultProcuder
from config import conf 

embeddings_path = conf.embeddings_path

print("embeddings_path:",embeddings_path)
faceReg = faceRecogner(embeddings_path)
processes = {"max":0}

def file2image(file):
    image = Image.open(io.BytesIO(file)).convert('RGB') 
    image = np.array(image)
    image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
    return image

WEB_SERVER = "http://172.16.50.91:8001/api/v1/attendance/attendance-daily"
class faceProcess:
    def __init__(self, conf):
        self.conf = conf
        
        self.faceRecognition = faceRecogner("outs/data/faceEmbedings")
        self.tracker = Tracker()
        self.detection_threshold = 0.25

        self.detector = Detector(classes = [0,1,2,3,4,5])
        #model_path = 'weights/yolov7-face/yolov7-face.pt'
        self.detector_path = '/home/vdc/project/computervision/python/weights/yolov7-face/yolov7-face.pt'  
        self.detector.load_model(self.detector_path)

        self.head_pose_path = "../weights/hopenet_alpha1.pkl"
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
                #x1, y1, x2, y2 = bbox
                #im = cv2.rectangle(frame, (int(x1), int(y1)), (int(x2), int(y2)), (0,255,0), 3)
                #im = cv2.resize(im, (960,720), interpolation = cv2.INTER_LINEAR)
                #cv2.imshow("frame detect",im)
                

                # DeepSORT -> Extracting Bounding boxes and its confidence scores.
                if score > self.detection_threshold:
                    detections.append([bbox[0],bbox[1], bbox[2],bbox[3], score])

            self.tracker.update(frame, detections)

            
            for track in self.tracker.tracker.tracks:
                if len(track.faces) > 0:
                    yaw_predicted, pitch_predicted, roll_predicted, _  = self.headpose.getHeadPose(track.faces[0], frame) 
                    if yaw_predicted is None or pitch_predicted is None or roll_predicted is None:
                        continue                  
                    if (yaw_predicted > 20 or yaw_predicted < -20) or (pitch_predicted > 20 or pitch_predicted < -20) or (roll_predicted > 20 or roll_predicted < -20):
                        track.faces = []
                        continue
                    
                    name, employee_id , score,  = self.faceRecognition.process(frame,track.faces[0])
                    #track.names.append([name,round(score,3)])
                    track.names.append(name)
                    counter = Counter(track.names)
                    most_common = counter.most_common()
                    #print('track ID {} : {}'.format(track.track_id,most_common))
                    if len(track.names) >= 30:
                        counter = Counter(track.names)
                        most_common = counter.most_common()
                        #print('track ID {} : {}'.format(track.track_id,most_common))
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

                            
                            try:
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
                            except Exception as e:
                                print(e)                  
                    track.faces = []
        
            return self.tracker.tracks
        return None

class AI_Process:
    def __init__(self, conf, mode):
        self.conf = conf
        self.face_processer = faceProcess(conf)     
        self.mode = mode
        
    def initialize(self, type_camera, video_path, topic, topic_result, bootstrap_servers):
        self.video_path = video_path
        try:
            #self.resutlSender = ResultProcuder(topic_result, bootstrap_servers)
        
            if type_camera == "camera" or type_camera == "cameraAI":
                self.cap = video_capture(video_path=video_path)
            else:
                self.cap = video_capture(topic=topic, bootstrap_servers=bootstrap_servers)
        except Exception as e:
            print(e)
            status[0] = 0
            print("Can not init processer:")
            
        return
 
    def process(self, frame):
        ret = None
        if self.mode == "security":
            ret =  self.face_processer.facerecognize(frame)
            
        elif self.mode == "traffic":
            ret =  "traffic"
            
        return ret
    
    def image_process(self, status):
        frame_idx = 0 
        isSuccess = True    
        colors = [(random.randint(0, 255), random.randint(0, 255), random.randint(0, 255)) for j in range(10)]
        while isSuccess and status[0] == 1: 
            isSuccess, frame = self.cap.getFrame()
            start_time = time.time()
            if isSuccess:
                print("frame_idx:",frame_idx)
                ret = self.process(frame)
                    
                end_time = time.time()
                time_process = end_time-start_time
                fps = 1/time_process
                fps = "FPS: " + str(int(fps))
                #print("frame_idx: {}  time process: {}  FPS: {}".format(frame_idx, time_process, fps))
                
                show_result = True
                save_video = False
                if show_result or save_video or status[1] == 1: # status.value == 2: send AI result to kafka server
                    if ret is not None:
                        for track in ret:
                            x1, y1, x2, y2 = track.bbox
                            track_id = track.track_id

                            cv2.rectangle(frame, (int(x1), int(y1)), (int(x2), int(y2)), (colors[track_id % len(colors)]), 3)
                            txt = 'id:' + str(track.track_id) + "-" + track.name
                            org = (int(x1), int(y1)- 10)
                            cv2.putText(frame, txt, org, cv2.FONT_HERSHEY_SIMPLEX, 1.1, (0,0,255), 2)

                #if status[1] == 1:
                    #resutlSender.sendResult(frame, frame_idx)
                if show_result:  
                    frame = cv2.resize(frame, (960,720), interpolation = cv2.INTER_LINEAR)
                    cv2.imshow("test",frame)
                    if cv2.waitKey(1) & 0xFF == ord('q'):
                        break
            
            frame_idx +=1
                        
        print("end process with :",self.video_path)
        status[0] = 0

    
#def mainProcess(mode, type_camera, video_path, topic, topic_result, bootstrap_servers, status):
def mainProcess(status, mode, type_camera, camera_path, topic, topic_result, bootstrap_servers):
    print("Start process with status.value:",status[0])

    processer = AI_Process(conf, mode)
    processer.initialize(type_camera, camera_path, topic, topic_result, bootstrap_servers)
    processer.image_process(status)
    '''
    frame_idx = 0 
    isSuccess = True    
    colors = [(random.randint(0, 255), random.randint(0, 255), random.randint(0, 255)) for j in range(10)]
    while isSuccess and status[0] == 1: 
        isSuccess, frame = processer.cap.getFrame()
        start_time = time.time()
        if isSuccess:
            print("frame_idx:",frame_idx)
            ret = processer.process(frame)
                 
            end_time = time.time()
            time_process = end_time-start_time
            fps = 1/time_process
            fps = "FPS: " + str(int(fps))
            #print("frame_idx: {}  time process: {}  FPS: {}".format(frame_idx, time_process, fps))
            
            show_result = True
            save_video = False
            if show_result or save_video or status[1] == 1: # status.value == 2: send AI result to kafka server
                if ret is not None:
                    for track in ret:
                        x1, y1, x2, y2 = track.bbox
                        track_id = track.track_id

                        cv2.rectangle(frame, (int(x1), int(y1)), (int(x2), int(y2)), (colors[track_id % len(colors)]), 3)
                        txt = 'id:' + str(track.track_id) + "-" + track.name
                        org = (int(x1), int(y1)- 10)
                        cv2.putText(frame, txt, org, cv2.FONT_HERSHEY_SIMPLEX, 1.1, (0,0,255), 2)

            #if status[1] == 1:
                #resutlSender.sendResult(frame, frame_idx)
            if show_result:  
                frame = cv2.resize(frame, (960,720), interpolation = cv2.INTER_LINEAR)
                cv2.imshow("test",frame)
                if cv2.waitKey(1) & 0xFF == ord('q'):
                    break
        
        frame_idx +=1
                     
    print("end process with :",processer.video_path)
    status[0] = 0
    '''


class Rectangle:
    def __init__(self, weight, height):
        self.weight = weight
        self.height = height
        print('khởi tạo của Rectangle -> weight: ', weight, ', height: ', weight)

    def cal_perimeter(self):
        # tính chu vi hình chữ nhật
        perimeter = 2 * (self.weight + self.height)
        return perimeter

    def compare_square (self, weight, height):
        # so sánh diện tích hình chữ nhật đưa vào và hình hiện tại
        input_square = weight * height
        current_square = self.weight * self.height
        compare_result = False
        if (input_square > current_square ): compare_result = True
        print('The comparision result is : ', compare_result)
        time.sleep(1)
        return compare_result
    

def function(myRectangle, status, processer):
    while True:
        time.sleep(1)
        print('perimeter: ', myRectangle.cal_perimeter())

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

@app.post("/init_process")
def init_process(
                        mode: str = Form(...),
                        type_camera: str = Form(...),
                        camera_path: str = Form(...),
                        topic: str = Form(...),
                        topic_result: str = Form(...),
                        bootstrap_servers: str = Form(...)                      
                    ):  
    #-- 1 need module check connect to camera_path
    #-- 2 need mode for camera AI
    #mode: 1- security,  2- traffic
    #type_camera: 1- camera, 2- cameraAI, 3- kafka
    #status(int): 0 - turn off process, 1 - run process, 2 - run process vs show results AI
    ret = 0
    message = "init process done"
    print("_______init process_______:")
    print("mode:",mode)
    print("type_camera:",type_camera)
    print("camera_path:",camera_path)
    print("topic:",topic)
    print("topic_result:",topic_result)
    print("bootstrap_servers:",bootstrap_servers)
    
    '''
    if mode is None or type_camera is None or topic_result is None or bootstrap_servers is None:
        message = "mode, type_camera, topic_result, bootstrap_servers was None"
        return 1   
    if type_camera in ["camera", "cameraAI"]:
        if camera_path is None:
            message = "camera_path was None"
            return 1
    elif type_camera == "kafka":
        if topic is None:
            message = "topic was None"
            return 1
    '''
        
    status = Array('i', range(10))
    status[0] = 1
    status[1] = 0
    #processer = AI_Processer(conf, mode)
    #processer.initialize(type_camera, camera_path, topic, topic_result, bootstrap_servers)
    p = Process(name ="process-1", target=mainProcess, args=(status, mode, type_camera, camera_path, topic, topic_result, bootstrap_servers, ))
    p.start()


    '''
    myRectangle = Rectangle(2,3)
    process1 = Process(target=function, args=(myRectangle,status, processer123, ))
    #process2 = Process(target=myRectangle.compare_square, args=(3,5,))

    process1.start()
    
    #process2.start()
    '''
    processes["max"] +=1
    id_process = str(processes["max"])
    processes[id_process] = status
    
         
    return {"result": ret,
            "message":message,
            "id_process": id_process}
    
@app.post("/adjust_process")
def adjust_process(                                    
                        id_process: str = Form(...),
                        status_value: int = Form(...)
                    ):  
    ret = 0
    message = "Adjust process done" 
    if id_process in processes.keys():
        processes[id_process][0] = status_value
        processes.pop(id_process)
    else:
        message = "id_process {} were not exist".format(id_process)
        print(message)
        ret = -1
    
    return {"result": ret,
            "message":message}

@app.post("/updateinfo")
async def update_info(
                        member_names: List[str] = Form(...),
                        member_ids: List[int] = Form(...),
                        vectors: List[float] = Form(...)
                    ):
    ret = 0

    return {"result": ret}

@app.post("/register")
async def register(
                        file: UploadFile,
                        employee_name: str = Form(...),
                        employee_id: str = Form(...),
                        event: str = Form(...)
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
                        id_process: str = Form(...)
                    ): 
    ret = 0
    process_status = "OFF"
    
    if id_process in processes.keys():
        if processes[id_process][0] > 0:
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
    video = "../video/face_video.mp4"
    #facerecognize_process(video)
    #updateInfoFromdatabase()
    mode = "security"
    processer = AI_Processer(conf, mode)
    #processer.initialize(type_camera, camera_path, topic, topic_result, bootstrap_servers)
    status = Array('i', range(10))
    p = Process(name ="process-1", target=mainProcess, args=(processer, status, ))
    p.start()
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

