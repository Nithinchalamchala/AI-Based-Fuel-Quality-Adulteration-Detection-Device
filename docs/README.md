# AI-Based Fuel Quality & Adulteration Detection Device

A portable device using ESP32-WROOM-32 and machine learning to detect adulterated fuel in real-time.

## 📊 Project Overview

| Feature | Description |
|---------|-------------|
| **Classification** | Binary (Pure vs Adulterated) |
| **Accuracy** | 92.5% on test data |
| **Response Time** | < 2 seconds |
| **Model Size** | 2.95 KB (TFLite) |
| **Cost** | ~₹1,300 |

## 🔧 Hardware Required

| Component | Qty | Cost (₹) |
|-----------|-----|----------|
| ESP32-WROOM-32 | 1 | 450 |
| HC-SR04 Ultrasonic Sensor | 1 | 60 |
| TDS Meter Sensor | 1 | 250 |
| Capacitive Sensor | 1 | 120 |
| 16x2 LCD with I2C | 1 | 180 |
| LEDs, Resistors, Breadboard | - | 150 |
| **Total** | | **~₹1,300** |

## 📁 Project Structure

```
ProtoType&Testing/
├── data/
│   ├── synthetic_data.py      # Generates training data
│   └── fuel_dataset.csv       # 600 samples dataset
├── model/
│   ├── model_training.py      # Train neural network
│   ├── convert_to_tflite.py   # Convert to ESP32 format
│   ├── fuel_model.h5          # Trained Keras model
│   ├── fuel_model.tflite      # TFLite model (2.95KB)
│   └── scaler_params.txt      # Normalization parameters
├── esp32/
│   └── fuel_detector/
│       ├── fuel_detector.ino  # Arduino sketch
│       └── model_data.h       # Model as C array
└── docs/
    ├── circuit_diagram.md     # Wiring guide
    └── README.md              # This file
```

## 🚀 Quick Start

### 1. Generate Dataset & Train Model
```bash
# Activate virtual environment
source venv/bin/activate

# Generate synthetic data
python data/synthetic_data.py

# Train model
python model/model_training.py

# Convert to TFLite
python model/convert_to_tflite.py
```

### 2. Run Interactive Demo (Streamlit)
Demonstrate the project on your laptop with a graphical interface:
```bash
streamlit run app.py
```

### 3. Upload to ESP32
1. Open `esp32/fuel_detector/fuel_detector.ino` in Arduino IDE
2. Install libraries:
   - `LiquidCrystal_I2C`
   - `TensorFlowLite_ESP32`
3. Select board: **ESP32 Dev Module**
4. Upload sketch

### 4. Wire the Circuit
See [circuit_diagram.md](docs/circuit_diagram.md) for detailed wiring.

## 🧠 Model Architecture
For a deep dive into the ML model (math, layers, and logic), read [MODEL_ARCHITECTURE.md](docs/MODEL_ARCHITECTURE.md).

## 📈 Model Performance

```
              precision    recall  f1-score   support

        Pure       0.90      0.95      0.93        60
 Adulterated       0.95      0.90      0.92        60

    accuracy                           0.93       120
```

## 🧪 How It Works

1. **Sensors measure** fuel properties:
   - Dielectric constant (capacitive sensor)
   - Sound velocity (ultrasonic sensor)
   - Conductivity (TDS sensor)

2. **ESP32 normalizes** readings using saved scaler parameters

3. **TFLite model** runs inference (~50ms)

4. **Result displayed** on LCD + LED indication:
   - 🟢 Green = Pure Fuel
   - 🔴 Red = Adulterated

## ⚠️ Important Notes

- Model trained on **synthetic data** based on research values
- For production use, calibrate with real fuel samples
- Handle fuel safely in well-ventilated areas

## 📚 References

- TensorFlow Lite Micro: https://www.tensorflow.org/lite/microcontrollers
- ESP32 Documentation: https://docs.espressif.com/projects/esp-idf/
