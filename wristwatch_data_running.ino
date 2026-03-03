#include "MAX30100_PulseOximeter.h"
#include <Adafruit_SSD1306.h>
#include <HTTPClient.h>
#include <WiFi.h>
#include <Wire.h>

#define SCREEN_WIDTH 128
#define SCREEN_HEIGHT 64
#define OLED_ADDR 0x3C
#define MPU_ADDR 0x68

Adafruit_SSD1306 display(SCREEN_WIDTH, SCREEN_HEIGHT, &Wire, -1);
PulseOximeter pox;

const char *ssid = "Tab A15";
const char *password = "tissnee050";
const char *serverURL ="https://neuroband-predict-alzheimers.onrender.com/api/hardware_data";

unsigned long lastUpload = 0;
int step_count = 0;

void setup() {

  Serial.begin(115200);
  Wire.begin(21, 22);

  display.begin(SSD1306_SWITCHCAPVCC, OLED_ADDR);
  display.clearDisplay();
  display.setTextColor(WHITE);

  // Wake MPU6050
  Wire.beginTransmission(MPU_ADDR);
  Wire.write(0x6B);
  Wire.write(0);
  Wire.endTransmission(true);

  if (!pox.begin()) {
    Serial.println("MAX30100 FAILED");
  } else {
    Serial.println("MAX30100 READY");
  }

  pox.setIRLedCurrent(MAX30100_LED_CURR_27_1MA);

  WiFi.begin(ssid, password);
  Serial.print("Connecting WiFi");
  while (WiFi.status() != WL_CONNECTED) {
    delay(500);
    Serial.print(".");
  }
  Serial.println("\nWiFi Connected");
}

void loop() {

  pox.update();

  float hr = pox.getHeartRate();
  float spo2 = pox.getSpO2();

  // Read MPU6050
  Wire.beginTransmission(MPU_ADDR);
  Wire.write(0x3B);
  Wire.endTransmission(false);
  Wire.requestFrom(MPU_ADDR, 14, true);

  int16_t AcX = Wire.read() << 8 | Wire.read();
  int16_t AcY = Wire.read() << 8 | Wire.read();
  int16_t AcZ = Wire.read() << 8 | Wire.read();
  Wire.read();
  Wire.read();
  int16_t GyX = Wire.read() << 8 | Wire.read();
  int16_t GyY = Wire.read() << 8 | Wire.read();
  int16_t GyZ = Wire.read() << 8 | Wire.read();

  float ax = AcX / 16384.0;
  float ay = AcY / 16384.0;
  float gx = GyX / 131.0;
  float gy = GyY / 131.0;
  float gz = GyZ / 131.0;

  float gait_speed = sqrt(ax * ax + ay * ay);
  float stride_variability = abs(ax - ay) * 10;
  float turning_velocity = abs(gz);
  float postural_sway = sqrt(gx * gx + gy * gy);

  if (gait_speed > 1.2)
    step_count++;

  float rmssd = 35.0;
  float sdnn = 45.0;
  float lf_hf = 1.6;
  int desat = (spo2 < 92) ? 1 : 0;

  // OLED display
  display.clearDisplay();
  display.setCursor(0, 0);
  display.setTextSize(1);
  display.print("HR: ");
  display.println(hr);
  display.print("SpO2: ");
  display.println(spo2);
  display.print("GyZ: ");
  display.println(gz);
  display.display();

  if (millis() - lastUpload > 10000) {

    lastUpload = millis();

    HTTPClient http;
    http.begin(serverURL);
    http.addHeader("Content-Type", "application/json");

    String json = "{";
    json += "\"device_id\":\"ESP32_INNOWAH_001\",";
    json += "\"timestamp\":" + String(millis()) + ",";

    json += "\"imu\":{";
    json += "\"gait_speed\":" + String(gait_speed) + ",";
    json += "\"stride_variability\":" + String(stride_variability) + ",";
    json += "\"turning_velocity\":" + String(turning_velocity) + ",";
    json += "\"postural_sway\":" + String(postural_sway) + ",";
    json += "\"step_count\":" + String(step_count);
    json += "},";

    json += "\"ppg\":{";
    json += "\"heart_rate\":" + String(hr) + ",";
    json += "\"spo2\":" + String(spo2) + ",";
    json += "\"rmssd\":" + String(rmssd) + ",";
    json += "\"sdnn\":" + String(sdnn) + ",";
    json += "\"lf_hf_ratio\":" + String(lf_hf) + ",";
    json += "\"desat_events\":" + String(desat);
    json += "}";

    json += "}";

    Serial.println("WRISTBAND JSON:");
    Serial.println(json);

    int code = http.POST(json);
    Serial.print("HTTP Response: ");
    Serial.println(code);

    http.end();
  }

  delay(200);
}