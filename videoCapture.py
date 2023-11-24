from kafka import KafkaConsumer, KafkaError
import cv2
import numpy as np
import datetime
import os
import json
import pickle
import base64

class video_capture:
    def __init__(self,video_path=None, topic=None, bootstrap_servers=None):
        self.video_path = video_path
        self.topic = topic
        self.bootstrap_servers = bootstrap_servers
        
        if self.video_path is not None:
            self.cap = cv2.VideoCapture(self.video_path)
        else:
            self.consumer = KafkaConsumer(
                        topic,
                        bootstrap_servers=bootstrap_servers,
                        auto_offset_reset="latest",
                        #enable_auto_commit=True,
                        group_id='KafkaConsumer',
                        value_deserializer=lambda x: json.loads(x.decode("utf-8")),
                        max_partition_fetch_bytes=5000000,
                        fetch_max_bytes=11534336,
                        receive_buffer_bytes=33554432,
                        send_buffer_bytes=33554432 * 10,
                        api_version=(0, 10)
                    )
            
    def getFrame(self):
        isSuccess = False
        
        if self.video_path is not None:
            isSuccess, frame = self.cap.read()
            
        else:
            msg = self.consumer.poll(timeout=1000)
            if msg is None:
                print("No message received within the timeout.")
            elif msg.error():
                if msg.error().code() == KafkaError._PARTITION_EOF:
                    # End of partition event - not an error
                    print("Reached end of partition.")
                else:
                    print(f"Error: {msg.error()}")
            else:
                # Process the received message
                print(f"Received message: ")
                frame = msg.value
                try:
                    frame_data_str = frame["image"]
                    frame_data_bytes = base64.b64decode(frame_data_str)
                    image_array = np.frombuffer(frame_data_bytes, np.uint8)

                    # Decode the NumPy array into an OpenCV image
                    frame = cv2.imdecode(image_array, cv2.IMREAD_COLOR)
                    isSuccess = True
                except Exception as e:
                    print(e)
                    
        return isSuccess, frame
    
if __name__ == "__main__":
    topic = "cam1"
    bootstrap_servers = ["172.16.50.91:9092"]
    
    cap = video_capture(None, topic,bootstrap_servers)
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    writer = cv2.VideoWriter('track.mp4',fourcc, 25.0, (1280,960))
    isSuccess = True
    while isSuccess:
        isSuccess, frame = cap.getFrame()
        frame = cv2.resize(frame, (1280,960), interpolation = cv2.INTER_LINEAR)
        writer.write(frame)
        #frame = cv2.resize(frame, (960,720), interpolation = cv2.INTER_LINEAR)
        #cv2.imshow("test",frame)
        #if cv2.waitKey(1) & 0xFF == ord('q'):
            #break
    writer.release()
            
    