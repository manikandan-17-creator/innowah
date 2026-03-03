#include <HTTPClient.h>
#include <WiFi.h>

#define EEG_PIN 0

const char *ssid = "Tab A15";
const char *password = "tissnee050";
const char *serverURL = "https://neuroband-predict-alzheimers.onrender.com/api/hardware_data";

unsigned long lastUpload = 0;

void setup() {
  Serial.begin(115200);
  WiFi.begin(ssid, password);
  Serial.print("Connecting WiFi");
  while (WiFi.status() != WL_CONNECTED) {
    delay(500);
    Serial.print(".");
  }
  Serial.println("\nWiFi Connected");
}


void loop() {

  // Simulated EEG computation (replace with FFT later)
  float raw = analogRead(EEG_PIN);
  float alpha = raw * 0.02;
  float theta = raw * 0.012;
  float delta = raw * 0.008;
  float theta_alpha_ratio = theta / (alpha + 0.01);
  float dominant_frequency = 8.0 + (raw / 200.0);


  if (millis() - lastUpload > 10000) {

    lastUpload = millis();

    HTTPClient http;
    http.begin(serverURL);
    http.addHeader("Content-Type", "application/json");

    String json = "{";
    json += "\"device_id\":\"ESP32_INNOWAH_001\",";
    json += "\"timestamp\":" + String(millis()) + ",";

    json += "\"eeg\":{";
    json += "\"alpha_power\":" + String(alpha) + ",";
    json += "\"theta_power\":" + String(theta) + ",";
    json += "\"delta_power\":" + String(delta) + ",";
    json += "\"theta_alpha_ratio\":" + String(theta_alpha_ratio) + ",";
    json += "\"dominant_frequency\":" + String(dominant_frequency);
    json += "}";

    json += "}";

    Serial.println("HEADBAND JSON:");
    Serial.println(json);

    int httpCode = http.POST(json);
    Serial.print("HTTP Response: ");
    Serial.println(httpCode);

    http.end();
  }

  delay(500);
}