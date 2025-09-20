# user_a_sensor.py
import os, json, time, threading
from datetime import datetime, timezone
from mqtt_base import make_client
from dotenv import load_dotenv
import random
load_dotenv()
current_dir = os.path.dirname(__file__)
data_folder = os.path.join(current_dir, r"../Data")
data_file = r"C:\Users\Maan\Desktop\9th Semester\AI Lab\Project\Data\sample_data_for_2label_test.json"
HOST = os.environ["HMQ_HOST"]
PORT = int(os.environ.get("HMQ_PORT", 8883))
USER = os.environ["HMQ_USER"]
PASS = os.environ["HMQ_PASS"]

SEND_TOPIC = "backend/sensor"
RECEIVE_TOPIC = "sensor/receive"
STATUS_TOPIC = "sensor/status"

CLIENT_ID = "sensor-device"

def generate_sensor_data():
    return {
        "Lifecycle_ID": random.randint(1, 14),
        "Repair_Flag": random.randint(0, 1),
        "Cycles_Completed": random.randint(1, 701280),
        "Motor_Current_mean": round(random.uniform(7.5031, 9.478), 4),    # A
        "Vibration_max": round(random.uniform(0.2457, 1.1943), 4),         # G
        "Vibration_std": round(random.uniform(0.0246, 0.1634), 4),         # G
        "Vibration_high_freq_mean": round(random.uniform(0.0627, 0.803), 4), # G
        "Temperature_mean": round(random.uniform(47.02, 51.344), 3),       # °C
        "Pressure_mean": round(random.uniform(148.84, 155.63), 2),         # kPa
        "Acoustic_mean": round(random.uniform(66.698, 71.264), 3),         # dB
        "Humidity_mean": round(random.uniform(47.94, 52.18), 2),           # %
        "Fault_Label": random.randint(0, 2),
        "timestamp": time.time(),
        "unit_id": 1
    }

def on_connect(client, userdata, flags, rc):
    print("Sensor connected")
    status = {"client": CLIENT_ID, "status": "online", "ts": datetime.now(tz=timezone.utc).isoformat()}
    client.publish(STATUS_TOPIC, json.dumps(status), qos=1, retain=True)
    client.subscribe(RECEIVE_TOPIC, qos=1)

def on_message(client, userdata, msg):
    print("Sensor received control:", msg.payload.decode())


# client.loop_start()
def load_json_data(json_path):
    if not os.path.exists(json_path):
        raise FileNotFoundError(f"JSON file not found: {json_path}")
    print(f"[INFO] Loading JSON data from {json_path}")
    with open(json_path, "r") as file:
        return json.load(file)
# simulate sending sensor data
running = True
def start_sensor():
    client = make_client(CLIENT_ID, USER, PASS, HOST, PORT, STATUS_TOPIC)
    client.on_connect = on_connect
    client.on_message = on_message

    client.connect(HOST, PORT)
    threading.Thread(target=client.loop_forever, daemon=True).start()
    while running:
        # data = {"sensor": "temp1", "value": 25.2, "unit": "C", "ts": datetime.now(tz=timezone.utc).isoformat()}
        data = load_json_data(data_file)
        # data = generate_sensor_data()
        client.publish(SEND_TOPIC, json.dumps(data), qos=1)
        # print("Sensor sent:", data)
        time.sleep(10)  # send data every 10 seconds
    time.sleep(5)
def stop_sensor():
    global running
    running = False