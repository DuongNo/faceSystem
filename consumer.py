from kafka import KafkaConsumer
import cv2
import numpy as np
import datetime
import os
import json
import pickle
import base64

topic = "CAMERA-VP"
bootstrap_servers = ["172.16.50.91:9092"]

consumer = KafkaConsumer(
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


print("consumer:",consumer)
for msg in consumer:
    frame = msg.value
    print(frame)
    
    try:
        frame_data_str = frame["frame"]
        frame_data_bytes = base64.b64decode(frame_data_str)
        image_array = np.frombuffer(frame_data_bytes, np.uint8)

        # Decode the NumPy array into an OpenCV image
        image = cv2.imdecode(image_array, cv2.IMREAD_COLOR)
        img_name = f"outs/check/frame-{frame['time']}-{frame['index']}.jpg"
        cv2.imwrite(img_name, image)
        print(f"Saved {img_name}")
    except Exception as e:
        print("can not save image")
        print(e)
    