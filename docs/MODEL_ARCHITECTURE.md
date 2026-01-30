# AI Model Implementation Details

This document explains the internal working of the machine learning model used in the Fuel Quality Detection Device.

## 1. Problem Formulation
We treat fuel quality detection as a **Binary Classification Problem**:
- **Class 0**: Pure Petrol
- **Class 1**: Adulterated Fuel (Kerosene, Water, Diesel mix)

## 2. Input Features
The model accepts 3 input features derived from sensors:

| Feature | Symbol | Sensor | Physical Significance |
|---------|--------|--------|-----------------------|
| **Dielectric Constant** | $D$ | Capacitive | Measures relative permittivity. Petrol $\approx$ 1.9, Adulterants > 2.0. |
| **Sound Velocity** | $V$ | Ultrasonic | Speed of sound in medium. Petrol $\approx$ 1300 m/s. Changes with density. |
| **Conductivity** | $\sigma$ | TDS | Electrical conductance. Petrol is an insulator ($\approx$ 0). Water has high $\sigma$. |

## 3. Data Preprocessing (Normalization)
Raw sensor values have different scales (e.g., Velocity 1300 vs Dielectric 1.9). We create a "Standard Scaler" to normalize inputs to have Mean = 0 and Variance = 1.

For each feature $x$:
$$ x_{norm} = \frac{x - \mu}{\sigma} $$

Where $\mu$ (mean) and $\sigma$ (std dev) are calculated from the training dataset.

**Stored Parameters:**
- $\mu = [2.37, 1289.7, 4.56]$
- $\sigma = [1.06, 35.67, 10.72]$

## 4. Neural Network Architecture
We use a **Feed-Forward Neural Network (Dense DNN)** optimized for microcontrollers.

### Structure:
1. **Input Layer**: 3 Neurons (Dielectric, Velocity, Conductivity)
2. **Hidden Layer 1**: 8 Neurons (ReLU Activation)
   - Captures non-linear relationships between features.
3. **Hidden Layer 2**: 4 Neurons (ReLU Activation)
   - Compresses features into higher-level abstract patterns.
4. **Output Layer**: 1 Neuron (Sigmoid Activation)
   - Outputs a probability score between 0 and 1.

### Mathematical Operations:
For a neuron $j$, the output $y_j$ is:
$$ y_j = f(\sum (w_i \cdot x_i) + b) $$
Where $f$ is the activation function (ReLU or Sigmoid).

## 5. Decision Logic
The final neuron outputs a probability score $P$:

| Score Range | Interpretation | Quality |
|-------------|----------------|---------|
| $0.0 \le P < 0.5$ | Pure Petrol | ✅ GOOD |
| $0.5 \le P \le 1.0$ | Adulterated | ❌ BAD |

## 6. Deployment Workflow (TinyML)
1. **Training**: Model trained in Python using TensorFlow.
2. **Quantization**: Converted to `float16` TFLite format to reduce size.
3. **C-Array**: Converted logic into a byte array (`model_data.h`) for ESP32.
4. **Inference**: The ESP32 logic:
   - Reads Sensors $\rightarrow$ Normalizes Data $\rightarrow$ Runs TFLite Interpreter $\rightarrow$ Displays Result.

This lightweight architecture allows the complex decision-making to happen in **< 50 milliseconds** on the ESP32 chip.
