// Gait classifier - MCU side.
//
// Reads an MPU-6050 on the header SDA/SCL pins (Wire) at 100 Hz and sends samples to
// the Python side over the Bridge in batches of BATCH_SIZE. All
// classification happens in Python (PLAN.md D1).
//
// Each batch is one string: samples separated by ';', each sample
//   seq,t_ms,ax,ay,az,gx,gy,gz
// with raw signed 16-bit readings (accel: 2048 LSB/g at +/-16 g,
// gyro: 16.4 LSB/(deg/s) at +/-2000 deg/s). seq increments every sample
// period, including ones where the read failed, so Python can spot gaps.
//
// Python calls show(activity, model) to put the detected activity on the LED
// matrix: a letter (S/W/R/B, or '-' for none yet) plus the active model number.

#include <Wire.h>
#include "Arduino_RouterBridge.h"
#include <Arduino_LED_Matrix.h>

#define IMU_WIRE Wire  // header SDA/SCL pins next to AREF (D20/D21). Use Wire1 for the Qwiic port.

const unsigned long SAMPLE_PERIOD_US = 10000;  // 100 Hz (PLAN.md D7)
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
unsigned long read_errors = 0;
unsigned long late_samples = 0;  // sample periods we fell behind on
String batch;
int batch_count = 0;

// ---- LED matrix (13 columns x 8 rows) ----
Arduino_LED_Matrix matrix;
const int MATRIX_COLS = 13;
const int MATRIX_ROWS = 8;

// 5x7 letters, one string per row, '#' = on. Index matches Python's activity index.
const char *LETTERS[][7] = {
    {".####", "#....", "#....", ".###.", "....#", "....#", "####."},  // 0 S standing
    {"#...#", "#...#", "#...#", "#.#.#", "#.#.#", "##.##", "#...#"},  // 1 W walking
    {"####.", "#...#", "#...#", "####.", "#.#..", "#..#.", "#...#"},  // 2 R running
    {"####.", "#...#", "#...#", "####.", "#...#", "#...#", "####."},  // 3 B biking
    {".....", ".....", ".....", "#####", ".....", ".....", "....."},  // 4 - no prediction yet
};
// 3x5 digits for the model number.
const char *DIGITS[][5] = {
    {".#.", "##.", ".#.", ".#.", "###"},  // 1
    {"##.", "..#", ".#.", "#..", "###"},  // 2
};

void blit(uint8_t *frame, const char *const *rows, int n_rows, int width, int col0, int row0) {
    for (int r = 0; r < n_rows; r++)
        for (int c = 0; c < width; c++)
            if (rows[r][c] == '#') frame[(row0 + r) * MATRIX_COLS + col0 + c] = 1;
}

// activity: 0-3 = standing/walking/running/biking, anything else = '-'. model: 1 or 2.
void show(int activity, int model) {
    uint8_t frame[MATRIX_COLS * MATRIX_ROWS] = {0};
    blit(frame, LETTERS[(activity >= 0 && activity <= 3) ? activity : 4], 7, 5, 1, 0);
    if (model == 1 || model == 2) blit(frame, DIGITS[model - 1], 5, 3, 9, 2);
    matrix.draw(frame);
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

    bool ok = write_reg(REG_PWR_MGMT_1, 0x01)     // wake up, clock from gyro X PLL
              && write_reg(REG_SMPLRT_DIV, 0x09)  // 1 kHz / (1 + 9) = 100 Hz internal rate
              && write_reg(REG_CONFIG, 0x03)      // DLPF ~44 Hz accel / 42 Hz gyro (PLAN.md D7)
              && write_reg(REG_GYRO_CONFIG, 0x18) // +/-2000 deg/s
              && write_reg(REG_ACCEL_CONFIG, 0x18); // +/-16 g
    if (!ok) imu_addr = 0;
    return ok;
}

void send_status(const char *msg) {
    Monitor.println(msg);
    Bridge.notify("imu_status", String(msg));
}

void setup() {
    Bridge.begin();
    Monitor.begin();
    matrix.begin();
    matrix.setGrayscaleBits(1);  // plain on/off
    show(-1, 0);
    // provide_safe: runs in the main loop, so drawing never interrupts an IMU read.
    Bridge.provide_safe("show", show);
    IMU_WIRE.begin();
    IMU_WIRE.setClock(100000);  // standard 100 kHz (PLAN.md D4)

    if (!init_imu()) send_status("MPU-6050 not found on SDA/SCL - check the jumper wires");
    batch.reserve(BATCH_SIZE * 48);
    next_sample_us = micros();
}

void loop() {
    unsigned long now_us = micros();
    if ((long)(now_us - next_sample_us) < 0) return;

    // Keep a fixed 100 Hz grid. If we fell more than a period behind, skip ahead
    // and count it (the skipped seq numbers show up as gaps in Python).
    next_sample_us += SAMPLE_PERIOD_US;
    if ((long)(now_us - next_sample_us) > (long)SAMPLE_PERIOD_US) {
        unsigned long missed = (now_us - next_sample_us) / SAMPLE_PERIOD_US;
        late_samples += missed;
        seq += missed;
        next_sample_us += missed * SAMPLE_PERIOD_US;
    }

    unsigned long this_seq = seq++;

    if (imu_addr == 0) {
        if (millis() - last_reinit_ms >= REINIT_INTERVAL_MS) {
            last_reinit_ms = millis();
            if (init_imu()) {
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
        read_errors++;
        imu_addr = 0;  // force a re-init attempt
        last_reinit_ms = millis();
        send_status("MPU-6050 read failed - retrying");
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
