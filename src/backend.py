import os, json, time, threading
from datetime import datetime, timezone
from mqtt_base import make_client
from dotenv import load_dotenv
from Predict_models import predict_models
load_dotenv()

HOST = os.environ["HMQ_HOST"]
PORT = int(os.environ.get("HMQ_PORT", 8883))
USER = os.environ["HMQ_USER"]
PASS = os.environ["HMQ_PASS"]

SEND_TOPIC = "plant/predict"
RECEIVE_TOPIC = "backend/sensor"
STATUS_TOPIC = "processor/status"

CLIENT_ID = "processor"

def on_connect(client, userdata, flags, rc):
    print("Processor connected")
    status = {"client": CLIENT_ID, "status": "online", "ts": datetime.now(tz=timezone.utc).isoformat()}
    client.publish(STATUS_TOPIC, json.dumps(status), qos=1, retain=True)
    # subscribe to sensor data + control commands
    # client.subscribe("sensor/send", qos=1)
    client.subscribe(RECEIVE_TOPIC, qos=1)

def on_message(client, userdata, msg):
    if msg.topic == "backend/sensor":
        payload = json.loads(msg.payload.decode())
        # preprocessed = [payload]
        # print("Processor got sensor data:", payload)
        processed = predict_models(payload)
        client.publish(SEND_TOPIC, json.dumps(processed), qos=1)
        print("Processor sent processed data:", processed)
    else:
        print("Processor received control:", msg.payload.decode())
running = True
def start_backend():
    client = make_client(CLIENT_ID, USER, PASS, HOST, PORT, STATUS_TOPIC)
    client.on_connect = on_connect
    client.on_message = on_message
    client.connect(HOST, PORT)
    threading.Thread(target=client.loop_forever, daemon=True).start()

    try:
        while running:
            time.sleep(1)
    except KeyboardInterrupt:
        print("Shutting down...")
        client.disconnect()

def stop_backend():
    global running
    running = False
