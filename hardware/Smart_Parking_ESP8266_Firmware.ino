/*
 * Project: AI-Driven Smart Parking System
 * Description: Firmware for ESP8266 & Ultrasonic Sensor HC-SR04
 * Author: Benhein Michael Ruben L
 * Conference: IEEE ICUIS 2025
 * Description: Logic for ultrasonic vehicle detection and real-time occupancy state.
 */

#define TRIG_PIN 12  // D6 -> GPIO12
#define ECHO_PIN 14  // D5 -> GPIO14

void setup() {
  Serial.begin(115200);
  pinMode(TRIG_PIN, OUTPUT);
  pinMode(ECHO_PIN, INPUT);
}

void loop() {
  long duration;
  int distance;
  
  // Send ultrasonic pulse
  digitalWrite(TRIG_PIN, LOW);
  delayMicroseconds(2);
  digitalWrite(TRIG_PIN, HIGH);
  delayMicroseconds(10);
  digitalWrite(TRIG_PIN, LOW);
  
  // Read echo pulse
  duration = pulseIn(ECHO_PIN, HIGH);
  distance = duration * 0.034 / 2; // Convert to cm

  // Print result
  Serial.print("Distance: ");
  Serial.print(distance);
  Serial.println(" cm");

  // Parking logic
  if (distance < 10) {
    Serial.println("🚗 Parking Spot Occupied");
  } else {
    Serial.println("✅ Parking Spot Available");
  }

  delay(1000); // Delay for stability
}
