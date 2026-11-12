/*
  FingerLock - sketch.ino (runs on the UNO Q's STM32 microcontroller)

  Python on the Linux side decides WHEN to open; this sketch does the
  real-time part: drives the DC motor for a set time, sets the status
  LEDs, and stops everything if Linux stops talking (heartbeat failsafe).

  Bridge functions Python can call:
    open_safe()       run the motor forward for OPEN_MS
    close_safe()      run it backward for CLOSE_MS
    set_status(code)  0 locked, 1 reading, 2 open, 3 wrong code / lockout
    heartbeat()       "Linux is alive" - sent every 0.5 s

  WIRING (DRV8833, L9110S, or similar 2-input H-bridge driver):
    UNO Q D5  -> driver IN1        UNO Q GND -> driver GND (shared ground!)
    UNO Q D6  -> driver IN2        motor     -> driver OUT1 / OUT2
    Motor power comes from a SEPARATE supply into the driver's VM pin.
    Never wire the motor straight to UNO Q pins: they are 3.3 V and can't
    supply motor current.
    Optional: D7 -> transistor/MOSFET that switches the LEDs lighting the camera.
*/
#include <Arduino_RouterBridge.h>

// ---------------- pins ----------------
const int MOTOR_IN1 = 5;      // PWM-capable pins (check the UNO Q pinout if you change them)
const int MOTOR_IN2 = 6;
const int LIGHT_PIN = 7;      // camera lights (on while reading a code)

// ---------------- motion tuning ----------------
const int MOTOR_SPEED = 180;             // 0-255 PWM duty
const unsigned long OPEN_MS = 350;       // how long to run = how far the latch moves
const unsigned long CLOSE_MS = 350;
const bool REVERSE_DIRECTION = false;    // flip if "open" turns the wrong way
const unsigned long LINUX_TIMEOUT_MS = 3000;

enum Motion { STOPPED, OPENING, CLOSING };
Motion motion = STOPPED;
unsigned long motionStart = 0;
bool isOpen = false;          // best guess of the latch position (there is no sensor)
int statusCode = 0;
volatile unsigned long lastHeartbeat = 0;   // written by the Bridge thread

void motorStop() {            // both inputs low = coast
  analogWrite(MOTOR_IN1, 0);
  analogWrite(MOTOR_IN2, 0);
}

void motorDrive(bool openDirection) {
  bool forward = openDirection != REVERSE_DIRECTION;
  analogWrite(MOTOR_IN1, forward ? MOTOR_SPEED : 0);
  analogWrite(MOTOR_IN2, forward ? 0 : MOTOR_SPEED);
}

// The onboard RGB LEDs are active-low: LOW = on.
void showStatus() {
  bool red = statusCode == 3, green = statusCode == 2, blue = statusCode == 1;
  digitalWrite(LED3_R, red ? LOW : HIGH);
  digitalWrite(LED3_G, green ? LOW : HIGH);
  digitalWrite(LED3_B, blue ? LOW : HIGH);
  digitalWrite(LIGHT_PIN, statusCode == 1 ? HIGH : LOW);
}

// ---- called from Python (provide_safe: runs between loop() passes) ----
void open_safe() {
  if (motion == STOPPED && !isOpen) {
    motion = OPENING;
    motionStart = millis();
    motorDrive(true);
  }
}

void close_safe() {
  if (motion == STOPPED && isOpen) {
    motion = CLOSING;
    motionStart = millis();
    motorDrive(false);
  }
}

void set_status(int code) {
  statusCode = code;
  showStatus();
}

// provide (not _safe): runs in the Bridge thread, so it only stores a time
void heartbeat() {
  lastHeartbeat = millis();
}

void setup() {
  pinMode(MOTOR_IN1, OUTPUT);
  pinMode(MOTOR_IN2, OUTPUT);
  motorStop();
  pinMode(LIGHT_PIN, OUTPUT);
  pinMode(LED3_R, OUTPUT);
  pinMode(LED3_G, OUTPUT);
  pinMode(LED3_B, OUTPUT);
  pinMode(LED4_R, OUTPUT);
  digitalWrite(LED4_R, HIGH);
  showStatus();

  Bridge.begin();
  Bridge.provide_safe("open_safe", open_safe);
  Bridge.provide_safe("close_safe", close_safe);
  Bridge.provide_safe("set_status", set_status);
  Bridge.provide("heartbeat", heartbeat);
  lastHeartbeat = millis();
}

void loop() {
  unsigned long beat = lastHeartbeat;           // read the shared value BEFORE millis()
  unsigned long now = millis();
  bool linuxLost = now - beat > LINUX_TIMEOUT_MS;

  if (motion != STOPPED) {
    unsigned long runFor = (motion == OPENING) ? OPEN_MS : CLOSE_MS;
    bool done = now - motionStart >= runFor;
    if (done || linuxLost) {
      motorStop();
      if (done) isOpen = (motion == OPENING);
      motion = STOPPED;
    }
  }

  // Failsafe: Linux went quiet -> motor stays off, red LED4 warns, camera lights off
  digitalWrite(LED4_R, linuxLost ? LOW : HIGH);
  if (linuxLost && statusCode == 1) {
    statusCode = 0;
    showStatus();
  }
}
