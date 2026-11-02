// Step counter - MCU side (PLAN.md D1, D4).
//
// Reads an MPU-6050 on the header SDA/SCL pins (Wire) at 100 Hz and sends samples to
// the Python relay over the Bridge in batches of BATCH_SIZE. Everything else happens
// on the laptop.
//
// Each batch is one string: samples separated by ';', each sample
//   seq,t_ms,ax,ay,az,gx,gy,gz
// with raw signed 16-bit readings (accel: 2048 LSB/g at +/-16 g,
// gyro: 16.4 LSB/(deg/s) at +/-2000 deg/s). seq increments every sample
// period, including ones where the read failed, so the laptop can spot gaps.
//
// Python calls show(code) to put the relay state on the LED matrix:
//   0 '-' relay starting / MQTT not connected, 1 'C' connected and streaming,
//   2 'R' recording (used from M2). The MCU itself shows '!' while the IMU is missing.

#include <Wire.h>
#include "Arduino_RouterBridge.h"
#include <Arduino_LED_Matrix.h>

#define IMU_WIRE Wire  // header SDA/SCL pins next to AREF (D20/D21). Use Wire1 for the Qwiic port.

const unsigned long SAMPLE_PERIOD_US = 10000;  // 100 Hz (PLAN.md D4)
const int BATCH_SIZE = 10;                       // -> 10 Bridge messages per second
const unsigned long REINIT_INTERVAL_MS = 500;    // how often to retry a lost sensor
const unsigned long NOT_FOUND_REPORT_MS = 2000;  // how often to remind Python the sensor is missing

// MPU-6050 registers
const uint8_t REG_SMPLRT_DIV = 0x19;
const uint8_t REG_CONFIG = 0x1A;
const uint8_t REG_GYRO_CONFIG = 0x1B;
const uint8_t REG_ACCEL_CONFIG = 0x1C;
const uint8_t REG_ACCEL_XOUT_H = 0x3B;
const uint8_t REG_PWR_MGMT_1 = 0x6B;
const uint8_t REG_WHO_AM_I = 0x75;

uint8_t imu_addr = 0;  // 0x68 or 0x69 once found, 0 = not connected
unsigned long next_sample_us = 0;
unsigned long last_reinit_ms = 0;
unsigned long last_not_found_ms = 0;
unsigned long seq = 0;
String batch;
int batch_count = 0;
int relay_code = 0;  // last code Python asked for, restored when the IMU comes back

// ---- LED matrix (13 columns x 8 rows) ----
Arduino_LED_Matrix matrix;
const int MATRIX_COLS = 13;
const int MATRIX_ROWS = 8;

// 5x7 glyphs, one string per row, '#' = on. Index = show() code, 3 = IMU missing.
const char *GLYPHS[][7] = {
    {".....", ".....", ".....", "#####", ".....", ".....", "....."},  // 0 -
    {".####", "#....", "#....", "#....", "#....", "#....", ".####"},  // 1 C
    {"####.", "#...#", "#...#", "####.", "#.#..", "#..#.", "#...#"},  // 2 R
    {"..#..", "..#..", "..#..", "..#..", "..#..", ".....", "..#.."},  // 3 !
};

void draw_glyph(int idx) {
    uint8_t frame[MATRIX_COLS * MATRIX_ROWS] = {0};
    for (int r = 0; r < 7; r++)
        for (int c = 0; c < 5; c++)
            if (GLYPHS[idx][r][c] == '#') frame[r * MATRIX_COLS + 4 + c] = 1;
    matrix.draw(frame);
}

void show(int code) {
    relay_code = (code >= 0 && code <= 2) ? code : 0;
    if (imu_addr != 0) draw_glyph(relay_code);
}

bool write_reg(uint8_t reg, uint8_t value) {
    IMU_WIRE.beginTransmission(imu_addr);
    IMU_WIRE.write(reg);
    IMU_WIRE.write(value);
    return IMU_WIRE.endTransmission() == 0;
}

bool read_regs(uint8_t addr, uint8_t reg, uint8_t *buf, uint8_t len) {
    IMU_WIRE.beginTransmission(addr);
    IMU_WIRE.write(reg);
    if (IMU_WIRE.endTransmission(false) != 0) return false;
    if (IMU_WIRE.requestFrom(addr, len) != len) return false;
    for (uint8_t i = 0; i < len; i++) buf[i] = IMU_WIRE.read();
    return true;
}

// Find the MPU-6050 at either address and configure it. Returns true on success.
bool init_imu() {
    imu_addr = 0;
    const uint8_t candidates[] = {0x68, 0x69};
    for (uint8_t addr : candidates) {
        uint8_t who = 0;
        if (read_regs(addr, REG_WHO_AM_I, &who, 1)) {
            imu_addr = addr;
            Monitor.print("MPU found at 0x");
            Monitor.print(addr, HEX);
            Monitor.print(", WHO_AM_I=0x");
            Monitor.println(who, HEX);
            break;
        }
    }
    if (imu_addr == 0) return false;

    // The MPU runs its own 1 kHz internal rate (SMPLRT_DIV = 0) and we read it on the
    // MCU's 100 Hz grid. If both ran at "100 Hz" on separate clocks, the MCU would every
    // so often read the same internal sample twice or skip one; at 1 kHz every read is
    // at most 1 ms old. The ~44 Hz low-pass filter prevents aliasing (PLAN.md D4).
    bool ok = write_reg(REG_PWR_MGMT_1, 0x01)     // wake up, clock from gyro X PLL
              && write_reg(REG_SMPLRT_DIV, 0x00)  // 1 kHz / (1 + 0) = 1 kHz internal rate
              && write_reg(REG_CONFIG, 0x03)      // DLPF ~44 Hz accel / 42 Hz gyro
              && write_reg(REG_GYRO_CONFIG, 0x18) // +/-2000 deg/s
              && write_reg(REG_ACCEL_CONFIG, 0x18); // +/-16 g
    if (!ok) imu_addr = 0;
    return ok;
}

void send_status(const char *msg) {
    Monitor.println(msg);
    Bridge.notify("imu_status", String(msg));
}

void imu_lost(const char *msg) {
    imu_addr = 0;
    draw_glyph(3);
    send_status(msg);
}

void setup() {
    Bridge.begin();
    Monitor.begin();
    matrix.begin();
    matrix.setGrayscaleBits(1);  // plain on/off
    // provide_safe: runs in the main loop, so drawing never interrupts an IMU read.
    Bridge.provide_safe("show", show);
    IMU_WIRE.begin();
    // 400 kHz fast mode: a 14-byte read keeps the bus busy ~0.45 ms of every 10 ms instead
    // of ~1.7 ms, so an App Lab re-flash is ~4x less likely to land mid-read and leave the
    // MPU holding SDA low (PLAN.md change log, M1).
    IMU_WIRE.setClock(400000);

    if (init_imu()) draw_glyph(relay_code);
    else imu_lost("MPU-6050 not found on SDA/SCL - check the jumper wires");
    batch.reserve(BATCH_SIZE * 48);
    next_sample_us = micros();
}

void loop() {
    unsigned long now_us = micros();
    if ((long)(now_us - next_sample_us) < 0) return;

    // Keep a fixed 100 Hz grid. If we fell more than a period behind, skip ahead
    // (the skipped seq numbers show up as gaps on the laptop).
    next_sample_us += SAMPLE_PERIOD_US;
    if ((long)(now_us - next_sample_us) > (long)SAMPLE_PERIOD_US) {
        unsigned long missed = (now_us - next_sample_us) / SAMPLE_PERIOD_US;
        seq += missed;
        next_sample_us += missed * SAMPLE_PERIOD_US;
    }

    unsigned long this_seq = seq++;

    if (imu_addr == 0) {
        if (millis() - last_reinit_ms >= REINIT_INTERVAL_MS) {
            last_reinit_ms = millis();
            if (init_imu()) {
                draw_glyph(relay_code);
                send_status("MPU-6050 connected");
            } else if (millis() - last_not_found_ms >= NOT_FOUND_REPORT_MS) {
                // Repeat this: a message sent before Python registered its handler is lost.
                last_not_found_ms = millis();
                send_status("MPU-6050 not found on SDA/SCL - check the jumper wires");
            }
        }
        return;  // no sample this period -> gap in seq
    }

    uint8_t raw[14];
    if (!read_regs(imu_addr, REG_ACCEL_XOUT_H, raw, 14)) {
        last_reinit_ms = millis();
        imu_lost("MPU-6050 read failed - retrying");
        return;
    }

    // Big-endian 16-bit values: ax, ay, az, temp, gx, gy, gz
    int16_t v[7];
    for (int i = 0; i < 7; i++) v[i] = (int16_t)((raw[2 * i] << 8) | raw[2 * i + 1]);

    if (batch_count > 0) batch += ';';
    batch += String(this_seq); batch += ',';
    batch += String(millis()); batch += ',';
    batch += String(v[0]); batch += ',';
    batch += String(v[1]); batch += ',';
    batch += String(v[2]); batch += ',';
    batch += String(v[4]); batch += ',';
    batch += String(v[5]); batch += ',';
    batch += String(v[6]);
    batch_count++;

    if (batch_count >= BATCH_SIZE) {
        Bridge.notify("imu_batch", batch);
        batch = "";
        batch_count = 0;
    }
}
