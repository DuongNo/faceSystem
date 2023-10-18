from facenet_pytorch import MTCNN
from fastapi import FastAPI,  UploadFile, File, Header
from fastapi.responses import FileResponse
from typing import Annotated
import os
from random import randint
import uuid

from faceRecognition import faceNet
import cv2
import torch
import time
import io
import PIL.Image as Image
import numpy as np
from typing import List

from typing import Union
from pydantic import BaseModel
from multiprocessing import Process


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

    mtcnn = MTCNN(thresholds= [0.7, 0.7, 0.8] ,keep_all=True, device = device)
    faceRecognition = faceNet("outs/data/faceEmbedings")

    #video_path1 = "video/vlc-record.mp4"
    cap = cv2.VideoCapture(video_path)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH,640)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT,480)
    frame_idx = 0   
    while cap.isOpened():
        isSuccess, frame = cap.read()
        #time.sleep(2)
        #print("checking + ",video_path)
        #continue
        if isSuccess:
            frame_idx +=1
            boxes, _ = mtcnn.detect(frame)
            if boxes is not None:
                for box in boxes:
                    bbox = list(map(int,box.tolist()))
                    face = faceReg.extract_face(bbox, frame)
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
                        id_process: int = Header(None),
                        type_camera: str = Header(None),
                        event: str = Header(None)
                    ):  
    ret = 0
    #2 event: 
    #1- run_process     - require: (camera_path)
    #2- shutdown_process   - require: (camera_path)

    if event == "run_process":
        p = Process(target=facerecognize_process, args=(camera_path,))
        list_process.append(p)
        p.start()
        id_process = len(list_process) - 1
    else:
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

@app.post("/image/")
async def create_upload_file(file: UploadFile = File(...)):
    #file.filename = f"{uuid.uuid4()}.jpg"
    contents = await file.read()
    image = file2image(contents)

    names = facerecognize(image)
    return {"names": names}

 
IMAGEDIR = "images/"
@app.get("/show/")
async def read_random_file():
 
    # get random file from the image directory
    files = os.listdir(IMAGEDIR)
    random_index = randint(0, len(files) - 1)
 
    path = f"{IMAGEDIR}{files[random_index]}"
     
    return FileResponse(path)


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
    test_recognition()