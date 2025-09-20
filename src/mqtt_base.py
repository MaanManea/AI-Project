# mqtt_base.py
import ssl, os, json
from datetime import datetime, timezone
import paho.mqtt.client as mqtt

def make_client(client_id, user, password, host, port, status_topic):
    client = mqtt.Client(client_id=client_id)
    client.username_pw_set(user, password)
    # LWT = offline message
    lwt_payload = json.dumps({"client": client_id, "status": "offline", "ts": datetime.now(tz=timezone.utc).isoformat()})
    client.will_set(status_topic, lwt_payload, qos=1, retain=True)

    # TLS setup
    client.tls_set(cert_reqs=ssl.CERT_REQUIRED, tls_version=ssl.PROTOCOL_TLS_CLIENT)
    client.tls_insecure_set(False)

    return client
