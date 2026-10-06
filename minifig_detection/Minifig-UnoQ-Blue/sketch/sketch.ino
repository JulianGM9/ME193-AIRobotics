#include "Arduino_RouterBridge.h"

// Cytron Maker Drive control pins.
const int M1A = 6;  // Motor 1
const int M1B = 5;
const int M2A = 9;  // Motor 2
const int M2B = 3;

// Set to true for a motor that spins the wrong way (e.g. mounted mirrored).
const bool M1_REVERSED = true;
const bool M2_REVERSED = true;

void setup() {
    pinMode(M1A, OUTPUT);
    pinMode(M1B, OUTPUT);
    pinMode(M2A, OUTPUT);
    pinMode(M2B, OUTPUT);
    set_motors(0, 0);

    Bridge.begin();
    Bridge.provide("set_motors", set_motors);
}

void loop() {}

// Maker Drive: PWM on A = forward, PWM on B = backward, both LOW = stop.
void drive(int pinA, int pinB, int speed) {
    if (speed > 0) {
        analogWrite(pinA, speed);
        analogWrite(pinB, 0);
    } else {
        analogWrite(pinA, 0);
        analogWrite(pinB, -speed);
    }
}

// Each speed: -255 (full backward) .. 0 (stop) .. 255 (full forward)
void set_motors(int speed1, int speed2) {
    speed1 = constrain(speed1, -255, 255);
    speed2 = constrain(speed2, -255, 255);
    drive(M1A, M1B, M1_REVERSED ? -speed1 : speed1);
    drive(M2A, M2B, M2_REVERSED ? -speed2 : speed2);
}
