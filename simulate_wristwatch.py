import requests
import time
import random

# Server URL
# URL = "https://neuroband-predict-alzheimers.onrender.com/api/hardware_data"
# For local testing, use:
URL = "http://127.0.0.1:5000/api/hardware_data"

DEVICE_ID = "ESP32_001"

def send_wristwatch_data():
    while True:
        # Simulate sensor readings
        hr = random.uniform(70, 85)
        spo2 = random.uniform(96, 99)
        gait_speed = random.uniform(0.8, 1.4)
        
        payload = {
            "device_id": DEVICE_ID,
            "timestamp": int(time.time() * 1000),
            "imu": {
                "gait_speed": round(gait_speed, 2),
                "stride_variability": round(random.uniform(2.0, 3.0), 2),
                "turning_velocity": round(random.uniform(100, 150), 2),
                "postural_sway": round(random.uniform(1.5, 2.5), 2),
                "step_count": 5000 + random.randint(0, 100)
            },
            "ppg": {
                "heart_rate": round(hr, 2),
                "spo2": round(spo2, 2),
                "lf_hf_ratio": 1.5,
                "desat_events": 0
            }
        }
        
        try:
            response = requests.post(URL, json=payload)
            print(f"[Wristwatch] Sent data. Status: {response.status_code}")
        except Exception as e:
            print(f"[Wristwatch] Error: {e}")
            
        time.sleep(10) # Send every 10 seconds

if __name__ == "__main__":
    print(f"Starting Wristwatch Simulator for {DEVICE_ID}...")
    send_wristwatch_data()
