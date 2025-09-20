import os
import time
import threading
from backend import start_backend, stop_backend
from sensor import start_sensor, stop_sensor   


if __name__ == "__main__":
    t1 = threading.Thread(target=start_sensor)
    t2 = threading.Thread(target=start_backend)

    t1.start()
    t2.start()

    try:
        while True:
            time.sleep(1)
    except KeyboardInterrupt:
        print("\nStopping...")
        stop_backend()
        stop_sensor()
        t1.join()
        t2.join()
        print("Program stopped.")
