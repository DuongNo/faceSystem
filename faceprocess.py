from fastapi import FastAPI,  UploadFile, File, Header, Form
from fastapi.responses import FileResponse
from typing import Annotated
import os
from random import randint
import uuid
import sys
#myFolderPath = '/home/vdc/project/computervision/python/VMS/faceprocess/faceSystem'
#sys.path.append(myFolderPath)

from facerecognition import faceRecogner
from face_tracker import Tracker
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
#from .kafka_process import kafka_consummer
from .videoCapture import video_capture
from hopenet import headPose
#from producer_result import ResultProcuder
from .config_face import Config   
from ultralytics import YOLO

conf = Config.load_json('/home/vdc/project/computervision/python/VMS/faceprocess/faceSystem/config.json')
WEB_SERVER = "http://172.16.50.91:8001/api/v1/attendance/attendance-daily"
class faceProcess:
    def __init__(self, conf):
        self.conf = conf
        
        self.faceRecognition = faceRecogner(self.conf)
        self.tracker = Tracker()
        self.detection_threshold = 0.25

        self.detector = YOLO(self.conf.weights.faceDetection)

        self.head_pose_path = self.conf.weights.headPose
        self.headpose = headPose(self.head_pose_path)

    def facerecognize(self, frame):
        results = self.detector(frame)  # predict on an image
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
                
                if conf > self.detection_threshold:
                    detections.append([bbox[0],bbox[1], bbox[2],bbox[3], conf])
                    
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
                    print('track ID {} : {}'.format(track.track_id,most_common))
                    if len(track.names) >= 20:
                        counter = Counter(track.names)
                        most_common = counter.most_common()
                        #print('track ID {} : {}'.format(track.track_id,most_common))
                        if most_common[0][1] >  20:
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

 
if __name__ == "__main__":
    video = "../video/face_video.mp4"

