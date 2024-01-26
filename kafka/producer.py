import cv2
import json
from kafka import KafkaProducer
from datetime import datetime

KAFKA_TOPIC = "cam1"
if type(KAFKA_TOPIC) == bytes:
    KAFKA_TOPIC = KAFKA_TOPIC.decode('utf-8')
KAFKA_BOOTSTRAP_SERVERS = "172.16.50.91:9092"
import base64

producer = KafkaProducer(
    bootstrap_servers=[KAFKA_BOOTSTRAP_SERVERS],
    value_serializer=lambda x: json.dumps(x).encode("utf-8"),
    max_request_size=26214400,
    send_buffer_bytes=26214400,
    api_version=(0, 10)
)

rtsp_url = "rtsp://VDI1:Vdi123456789@172.16.9.254:554/MediaInput/264/Stream1"
cap = cv2.VideoCapture(rtsp_url)
frame_count = 0
while cap.isOpened():
    print("frame_count:",frame_count)
    print("producer:",producer)
    ret, frame = cap.read()
    _, jpeg_bytes = cv2.imencode(".jpg", frame)
    base64_image = base64.b64encode(jpeg_bytes).decode("utf-8")
    frame_count += 1
    if ret == True:
        frame_data = {
            "cam_name": "cam_1",
            "index": frame_count,
            "image": base64_image,
            "time": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        }
        # Send to Kafka
        future = producer.send(KAFKA_TOPIC, value=frame_data)
        #producer.flush()
        try:
            record_metadata = future.get(timeout=60)
            print(
                f"Message sent to topic {record_metadata.topic} at partition {record_metadata.partition} and offset {record_metadata.offset}"
            )
        except Exception as e:
            print(f"Error sending message: {str(e)}")
    else:
        break
cap.release()