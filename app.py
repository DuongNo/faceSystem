from fastapi import FastAPI,  UploadFile, File, Header, Form
from fastapi.responses import FileResponse

import cv2
import time
import io
import PIL.Image as Image
import numpy as np
from typing import List
from numpy import random
from collections import Counter
import requests
import datetime
from facerecognition import faceRecogner

from multiprocessing import Process, Value, Array 
import base64
import sys
from faceSystem import faceProcess
from config_face import Config

conf = Config.load_json('/home/vdc/project/computervision/python/VMS/faceprocess/faceSystem/config.json')
embeddings_path = conf.embeddings_path
faceReg = faceRecogner(conf, register=True)
 
processes = {"max":0}

def file2image(file):
    image = Image.open(io.BytesIO(file)).convert('RGB') 
    image = np.array(image)
    image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
    return image

class AI_Process:
    def __init__(self, mode="security"):
        self.conf = conf    
        self.mode = mode
        
    def initialize(self, type_camera, video_path):
        self.video_path = video_path
        self.face_processer = faceProcess(conf)
        self.cap = cv2.VideoCapture(video_path)
        return
 
    def process(self, frame):
        ret = None
        if self.mode == "security":
            ret =  self.face_processer.facerecognize(frame)
            
        return ret
    
    def image_process(self, status):
        frame_idx = 0 
        isSuccess = True    
        colors = [(random.randint(0, 255), random.randint(0, 255), random.randint(0, 255)) for j in range(10)]
        frame_idx = 6600#cong- 3500 #bach - 450, congBA-4800, 6600 - ca 3 nguoi

        self.cap.set(cv2.CAP_PROP_POS_FRAMES, frame_idx)
        while isSuccess and status[0] == 1: 
            isSuccess, frame = self.cap.read()
            start_time = time.time()
            if isSuccess:
                print("frame_idx:",frame_idx)
                print("frame.shape:",frame.shape)
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
                            name = track.name
                            #if track.class_name is None:
                            #    name = track.name
                            #else:
                            #name = track.name + track.class_name + " " + track.lp
                            
                            txt = 'id:' + str(track.track_id) + "-" + name
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

def initProcess(status, type_camera, camera_path):
    print("Start process with status.value:",status[0])
    
    type_camera = conf.default_security.type_camera
    camera_path = conf.default_security.camera_path
    
    #type_camera = "camera"
    #camera_path = "/home/vdc/project/computervision/python/video/face_video.mp4"
    print("running with default_security mode")
        
        
    processer = AI_Process()
    processer.initialize(type_camera, camera_path)
    processer.image_process(status) 

WEB_SERVER = "http://172.16.50.91:8001/api/v1/attendance/attendance-daily"
app = FastAPI()
@app.get("/")
async def check():
    return {"message": "Hello World"}

@app.post("/init_process")
def init_process(
                        type_camera: str = Form(...),
                        camera_path: str = Form(...),                     
                    ):  
    #-- 1 need module check connect to camera_path
    #-- 2 need mode for camera AI
    #mode: 0- defaul security, 1- default traffic, security,  traffic, 
    #type_camera: 1- camera, 2- cameraAI, 3- kafka
    #status(int): 0 - turn off process, 1 - run process, 2 - run process vs show results AI
    ret = 0
    message = "init process done"
    print("_______init process_______:")
    print("type_camera:",type_camera)
    print("camera_path:",camera_path)
    
    '''
    if mode is None or type_camera is None:
        message = "mode, type_camera was None"
        return 1   
    if type_camera in ["camera", "cameraAI"]:
        if camera_path is None:
            message = "camera_path was None"
            return 1 
    '''
        
    status = Array('i', range(10))
    status[0] = 1
    status[1] = 0
    p = Process(name ="process-1", target=initProcess, args=(status, type_camera, camera_path, ))
    p.start()

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
    result = 0
    if event == "register":
        contents = await file.read()
        image = file2image(contents)
        result, vec = faceReg.register(image, employee_name, employee_id, get_vector=True)
    elif event == "updateface":
        contents = await file.read()
        image = file2image(contents)
        result, vec = faceReg.updateFace(image, employee_name, employee_id, get_vector=True)
    elif event == "remove":
        result = faceReg.removeId(employee_id)
    elif event == "facebank":
        img = "/home/vdc/project/computervision/python/VMS/data/facedemo"
        faceReg.createFaceBank(img, embeddings_path)
        
    elif event == "evaluate":
        img = "/home/1.data/computervision/face/face_recognition/VN-celeb"
        #img = "/home/vdc/project/computervision/python/VMS/data/facedemo"
        embeddings_evaluate = "/home/vdc/project/computervision/python/VMS/data/evaluate/embedding"
        faceReg.createFaceBank(img, embeddings_evaluate)
        faceReg.evaluate(img, embeddings_evaluate)
        
    elif event == "clear":
        faceReg.clearInfo()
        faceReg.saveInfo()
        
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
    mode = "security"
   

