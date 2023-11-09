import pickle
from kafka import KafkaConsumer, KafkaProducer
from kafka.errors import KafkaError
import cv2
import json



ENCODE_PARAMS = [int(cv2.IMWRITE_JPEG_QUALITY), 90]
class kafka_producer:
    def __init__(self):
        self.producer = KafkaProducer(
            value_serializer=lambda m:pickle.dumps(m),
            bootstrap_servers="172.16.1.137:9092")
    def send_data(self,face ,trackID):
        rs, encode_face = cv2.imencode('.jpg', face, ENCODE_PARAMS)
        p = self.producer.send("cam_push_face",key=bytes(str(trackID), "utf-8") , value=encode_face)
        p.get()


class kafka_consummer:
    def __init__(self, toppic='cam_get_name', server='172.16.1.137:9092'):
        print("toppic:",toppic)
        print("server:",server)
        self.consumer = KafkaConsumer(toppic,
                        bootstrap_servers=[server],
                        api_version=(0, 10),
                        auto_offset_reset="latest",
                        enable_auto_commit=True,
                        value_deserializer=lambda x: json.loads(x.decode("utf-8")),
                        max_partition_fetch_bytes=5000000,
                        fetch_max_bytes=11534336,
                        receive_buffer_bytes=33554432,
                        send_buffer_bytes=33554432 * 10,
                        #auto_offset_reset='earliest'
                        )
                        


    '''
    def thread_consumer(self):
        for message in consumer.consumer:
            print("%s:%d:%d: key=%s value=%s" % (message.topic, message.partition,
                                          message.offset, message.key,
                                          message.value))
            key = message.key.decode("utf-8")
            value = message.value.decode("utf-8")
            self.list_Name_TrackID.append((int(key),value))
            if not self.program_running:
                break
    '''

if __name__ == "__main__":
    toppic = "cam1"
    bootstrap_servers = "172.16.50.91:9092"
    consumer = KafkaConsumer(toppic,
                        bootstrap_servers=bootstrap_servers,
                        auto_offset_reset="latest",                                        
                        enable_auto_commit=True,
                        value_deserializer=lambda x: json.loads(x.decode("utf-8")),
                        max_partition_fetch_bytes=5000000,
                        fetch_max_bytes=11534336,
                        receive_buffer_bytes=33554432,
                        send_buffer_bytes=33554432 * 10,
                        #auto_offset_reset='earliest'
                        api_version=(0, 10)  
                        )

    for msg in consumer:
        frame = msg.value
        print(frame)