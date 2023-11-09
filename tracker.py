from deep_sort.deep_sort.tracker import Tracker as DeepSortTracker
from deep_sort.tools import generate_detections as gdet
from deep_sort.deep_sort import nn_matching
from deep_sort.deep_sort.detection import Detection
import numpy as np
import os
import cv2
import requests
import datetime
import base64

class Tracker:
    tracker = None
    encoder = None
    tracks = None

    def __init__(self):
        max_cosine_distance = 0.4
        nn_budget = None

        encoder_model_filename = '/home/vdc/project/computervison/python/face/faceSystem/deep_sort/model/mars-small128.pb'
        self.faces_path = "/home/vdc/project/computervison/python/face/faceSystem/outs/tracks"

        metric = nn_matching.NearestNeighborDistanceMetric("cosine", max_cosine_distance, nn_budget)
        self.tracker = DeepSortTracker(metric)
        self.encoder = gdet.create_box_encoder(encoder_model_filename, batch_size=1)

    def update(self, frame, detections):

        if len(detections) == 0:
            self.tracker.predict()
            self.tracker.update([])  
            self.update_tracks(frame)
            return

        bboxes = np.asarray([d[:-1] for d in detections])
        bboxes[:, 2:] = bboxes[:, 2:] - bboxes[:, 0:2]
        scores = [d[-1] for d in detections]

        features = self.encoder(frame, bboxes)

        dets = []
        for bbox_id, bbox in enumerate(bboxes):
            dets.append(Detection(bbox, scores[bbox_id], features[bbox_id]))

        self.tracker.predict()
        self.tracker.update(dets)
        self.update_tracks(frame)

    def update_tracks(self, frame):
        tracks = []
        for track in self.tracker.tracks:
            if not track.is_confirmed() or track.time_since_update > 1:
                continue
            bbox = track.to_tlbr()
            id = track.track_id
            name = track.name

            tracks.append(Track(id, bbox, name))

            if track.name == "" and track.age - track.step_recognize > 3:
                x1, y1, x2, y2 = int(bbox[0]), int(bbox[1]), int(bbox[2]), int(bbox[3])
                x1 = max(x1,0)
                y1 = max(y1,0)
                x2 = min(x2,frame.shape[1])
                y2 = min(y2,frame.shape[0])
                if x2 - x1 > 60 and y2 - y1 > 70:
                    track.faces.append([x1, y1, x2, y2])
                    track.step_recognize = track.age

        self.tracks = tracks

        '''
        for track in self.tracker.deaded_tracks:
            if track.name != '':
                print('track ID {} is dead: {}'.format(track.track_id,track.name))
                img_name = "outs/track_faces/trackID_{}_{}.jpg".format(track.track_id,track.name)
                #cv2.imwrite(img_name,track.face)
                print("img_name:",img_name)
                #print("track.face:",track.face.shape)


                #cv2.imshow("test_face",track.face)
                #if cv2.waitKey(1) & 0xFF == ord('q'):
                #    break
            #if len(track.face) > 0:
            #    path = os.path.join(self.faces_path, str(track.track_id))
            #    if not os.path.isdir(path): 
            #        os.mkdir(path)

            #for idx, face in enumerate(track.face):
            #    image_name = '{:05}'.format(idx) + ".jpg"      
            #    image_path = os.path.join(path, image_name)
            #    cv2.imwrite(image_path, face)   
        '''

class Track:
    track_id = -1
    bbox = None

    def __init__(self, id, bbox, name=''):
        self.track_id = id
        self.bbox = bbox
        self.name = name
