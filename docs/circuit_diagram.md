# Circuit Diagram - Fuel Quality Detection System

## Components Required

| Component | Quantity | Approx Cost (₹) |
|-----------|----------|-----------------|
| ESP32-WROOM-32 DevKit | 1 | 450 |
| HC-SR04 Ultrasonic Sensor | 1 | 60 |
| TDS Meter Sensor Module | 1 | 250 |
| Capacitive Soil Moisture Sensor (used for dielectric) | 1 | 120 |
| 16x2 LCD with I2C Module | 1 | 180 |
| Green LED (5mm) | 1 | 5 |
| Red LED (5mm) | 1 | 5 |
| Push Button | 1 | 5 |
| 220Ω Resistors | 2 | 5 |
| 10kΩ Resistor | 1 | 2 |
| Breadboard | 1 | 120 |
| Jumper Wires | 1 set | 80 |
| USB Cable for ESP32 | 1 | 50 |

**Total Estimated Cost: ₹1,332**

---

## Wiring Diagram

```
                          ESP32-WROOM-32
                    ┌─────────────────────┐
                    │                     │
    ┌───────────────┤ GPIO 5       GPIO 25├───────[220Ω]───LED(Green)───GND
    │               │                     │
    │   ┌───────────┤ GPIO 18      GPIO 27├───────[220Ω]───LED(Red)────GND
    │   │           │                     │
    │   │           │ GPIO 35 ←───────────┼─────── Capacitive Sensor (Analog Out)
    │   │           │                     │
    │   │           │ GPIO 34 ←───────────┼─────── TDS Sensor (Analog Out)
    │   │           │                     │
    │   │           │ GPIO 21 ────────────┼─────── LCD SDA (I2C)
    │   │           │                     │
    │   │           │ GPIO 22 ────────────┼─────── LCD SCL (I2C)
    │   │           │                     │
    │   │   ┌───────┤ GPIO 4       3.3V   ├─────── Sensor VCC
    │   │   │       │                     │
    │   │   │       │ GND                 ├─────── Common GND
    │   │   │       └─────────────────────┘
    │   │   │
    │   │   └──── Button ────[10kΩ]──── GND
    │   │
    │   └──── HC-SR04 ECHO
    │
    └──────── HC-SR04 TRIG


    HC-SR04 Ultrasonic Sensor:
    ┌─────────────────────┐
    │  VCC  TRIG ECHO GND │
    └──┬─────┬────┬────┬──┘
       │     │    │    │
       │     │    │    └── GND
       │     │    └─────── GPIO 18
       │     └──────────── GPIO 5
       └────────────────── 5V (or VIN)

    
    TDS Sensor Module:
    ┌─────────────────┐
    │  VCC  GND  AOUT │
    └───┬────┬────┬───┘
        │    │    │
        │    │    └── GPIO 34
        │    └─────── GND
        └──────────── 3.3V


    Capacitive Sensor:
    ┌─────────────────┐
    │  VCC  GND  AOUT │
    └───┬────┬────┬───┘
        │    │    │
        │    │    └── GPIO 35
        │    └─────── GND
        └──────────── 3.3V


    LCD 16x2 with I2C:
    ┌─────────────────────┐
    │  GND VCC SDA SCL    │
    └───┬───┬───┬───┬─────┘
        │   │   │   │
        │   │   │   └── GPIO 22 (SCL)
        │   │   └────── GPIO 21 (SDA)
        │   └────────── 5V (or VIN)
        └────────────── GND
```

---

## Pin Summary Table

| ESP32 Pin | Connected To | Purpose |
|-----------|--------------|---------|
| GPIO 5 | HC-SR04 TRIG | Ultrasonic trigger |
| GPIO 18 | HC-SR04 ECHO | Ultrasonic echo |
| GPIO 34 | TDS Sensor AOUT | Conductivity reading |
| GPIO 35 | Capacitive Sensor AOUT | Dielectric reading |
| GPIO 21 | LCD SDA | I2C Data |
| GPIO 22 | LCD SCL | I2C Clock |
| GPIO 25 | Green LED (+) | Pure fuel indicator |
| GPIO 27 | Red LED (+) | Adulterated indicator |
| GPIO 4 | Push Button | Trigger analysis |
| 3.3V | Sensor VCC | Power for analog sensors |
| 5V/VIN | LCD VCC, HC-SR04 VCC | Power for 5V components |
| GND | All GND | Common ground |

---

## Assembly Instructions

### Step 1: Set Up the Breadboard
1. Place ESP32 on one end of the breadboard
2. Connect power rails: 3.3V and GND from ESP32

### Step 2: Connect Ultrasonic Sensor
1. Connect VCC to 5V (VIN pin on ESP32)
2. Connect GND to GND
3. Connect TRIG to GPIO 5
4. Connect ECHO to GPIO 18

### Step 3: Connect TDS Sensor
1. Connect VCC to 3.3V
2. Connect GND to GND
3. Connect analog output to GPIO 34

### Step 4: Connect Capacitive Sensor
1. Connect VCC to 3.3V
2. Connect GND to GND
3. Connect analog output to GPIO 35

### Step 5: Connect LCD
1. Connect VCC to 5V
2. Connect GND to GND
3. Connect SDA to GPIO 21
4. Connect SCL to GPIO 22

### Step 6: Connect LEDs
1. Green LED: Anode → 220Ω resistor → GPIO 25, Cathode → GND
2. Red LED: Anode → 220Ω resistor → GPIO 27, Cathode → GND

### Step 7: Connect Button
1. One terminal to GPIO 4
2. Other terminal to GND
3. Add 10kΩ pull-up resistor between GPIO 4 and 3.3V (optional - using internal pull-up)

---

## Sensor Placement for Testing

```
    ┌─────────────────────────────────┐
    │        Test Container           │
    │  ┌───────────────────────────┐  │
    │  │                           │  │
    │  │    FUEL SAMPLE            │  │
    │  │                           │  │
    │  │  ┌───┐     ╔═══╗         │  │
    │  │  │TDS│     ║US ║         │  │
    │  │  └─┬─┘     ╚═╤═╝         │  │
    │  │    │         │           │  │
    │  │  ┌─┴─────────┴─┐         │  │
    │  │  │ Capacitive  │         │  │
    │  │  │   Sensor    │         │  │
    │  │  └─────────────┘         │  │
    │  └───────────────────────────┘  │
    └─────────────────────────────────┘
    
    TDS = TDS Sensor (dipped in liquid)
    US = Ultrasonic Sensor (facing liquid surface)
    Capacitive = Capacitive sensor (touching liquid)
```

---

## Important Notes

> [!WARNING]
> **Safety**: Always handle fuel samples with care. Work in well-ventilated areas. Keep away from ignition sources.

> [!NOTE]
> **LCD Address**: If LCD doesn't work, the I2C address might be 0x3F instead of 0x27. Use I2C scanner sketch to find correct address.

> [!TIP]
> **Calibration**: Sensor readings will need calibration with known fuel samples for accurate results in real-world use.
