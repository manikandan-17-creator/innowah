import requests
import time
import random

# Server URL

URL = "http://127.0.0.1:5000/api/hardware_data"

DEVICE_ID = "ESP32_001"

def send_headband_data():
    while True:
        # Simulate EEG readings
        alpha = random.uniform(25, 35)
        theta = random.uniform(10, 15)
        
        payload = {
            "device_id": DEVICE_ID,
            "timestamp": int(time.time() * 1000),
            "eeg": {
                "alpha_power": round(alpha, 2),
                "theta_power": round(theta, 2),
                "delta_power": round(random.uniform(5, 10), 2),
                "theta_alpha_ratio": round(theta / alpha, 2),
                "dominant_frequency": round(random.uniform(8, 12), 2)
            },
            "ppg": {
                "rmssd": round(random.uniform(30, 40), 2),
                "sdnn": round(random.uniform(40, 50), 2)
            }
        }
        
        try:
            response = requests.post(URL, json=payload)
            print(f"[Headband] Sent data. Status: {response.status_code}")
        except Exception as e:
            print(f"[Headband] Error: {e}")
            
        time.sleep(10) # Send every 10 seconds

if __name__ == "__main__":
    print(f"Starting Headband Simulator for {DEVICE_ID}...")
    send_headband_data()
