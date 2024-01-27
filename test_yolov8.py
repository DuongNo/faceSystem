# Load YOLOv8n, train it on COCO128 for 3 epochs and predict an image with it
from ultralytics import YOLO

model = YOLO('yolov8n.pt')  # load a pretrained YOLOv8n detection model
#model.train(data='coco128.yaml', epochs=3)  # train the model
results = model('https://ultralytics.com/images/bus.jpg')  # predict on an image
boxes = results[0].boxes.xyxy.cpu()
clss = results[0].boxes.cls.cpu().tolist()
confs = results[0].boxes.conf.float().cpu().tolist()
for box, cls, conf in zip(boxes, clss, confs):
    box = list(map(int,box.tolist()))
    cls = int(cls)
    print("box:",box)
    print("cls:",cls)
    print("conf:",conf)


'''
from ultralytics import YOLO
import cv2
from ultralytics.utils.plotting import Annotator  # ultralytics.yolo.utils.plotting is deprecated

model = YOLO('yolov8n.pt')
cap = cv2.VideoCapture(0)
cap.set(3, 640)
cap.set(4, 480)

while True:
    _, img = cap.read()
    
    # BGR to RGB conversion is performed under the hood
    # see: https://github.com/ultralytics/ultralytics/issues/2575
    results = model.predict(img)

    for r in results:
        
        annotator = Annotator(img)
        
        boxes = r.boxes
        for box in boxes:
            
            b = box.xyxy[0]  # get box coordinates in (left, top, right, bottom) format
            c = box.cls
            annotator.box_label(b, model.names[int(c)])
          
    img = annotator.result()  
    cv2.imshow('YOLO V8 Detection', img)     
    if cv2.waitKey(1) & 0xFF == ord(' '):
        break

cap.release()
cv2.destroyAllWindows()
'''