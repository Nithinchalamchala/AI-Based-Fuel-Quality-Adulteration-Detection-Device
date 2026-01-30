"""
Fuel Quality Model Tester
Test the trained model on your laptop before deploying to ESP32

Run: python model/test_model.py
"""

import numpy as np
import tensorflow as tf
import os

# Paths
MODEL_DIR = os.path.dirname(__file__)
TFLITE_MODEL_PATH = os.path.join(MODEL_DIR, 'fuel_model.tflite')

# Scaler parameters (from training)
SCALER_MEAN = np.array([2.371911, 1289.732665, 4.567489])
SCALER_STD = np.array([1.063877, 35.678684, 10.728714])

# Detection threshold
THRESHOLD = 0.5

def load_model():
    """Load TFLite model"""
    interpreter = tf.lite.Interpreter(model_path=TFLITE_MODEL_PATH)
    interpreter.allocate_tensors()
    return interpreter

def predict(interpreter, dielectric, velocity, conductivity):
    """Make prediction with given sensor values"""
    # Normalize input
    raw_input = np.array([[dielectric, velocity, conductivity]])
    normalized = (raw_input - SCALER_MEAN) / SCALER_STD
    
    # Get tensor details
    input_details = interpreter.get_input_details()
    output_details = interpreter.get_output_details()
    
    # Run inference
    interpreter.set_tensor(input_details[0]['index'], normalized.astype(np.float32))
    interpreter.invoke()
    prediction = interpreter.get_tensor(output_details[0]['index'])[0][0]
    
    return prediction

def classify(prediction):
    """Classify based on threshold"""
    if prediction < THRESHOLD:
        return "PURE ✅", "green"
    else:
        return "ADULTERATED ❌", "red"

def main():
    print("=" * 70)
    print("     FUEL QUALITY MODEL TESTER")
    print("     Test your model before deploying to ESP32")
    print("=" * 70)
    
    # Load model
    print("\n📦 Loading TFLite model...")
    interpreter = load_model()
    print("   Model loaded successfully!\n")
    
    # ============================================================
    # PERMISSIBLE LIMITS (Reference Values)
    # ============================================================
    print("=" * 70)
    print("📊 PERMISSIBLE LIMITS (Pure Petrol Reference Values)")
    print("=" * 70)
    print("""
    ┌─────────────────────┬─────────────────────┬─────────────────────┐
    │      Parameter      │    Pure Petrol      │    Adulterated      │
    ├─────────────────────┼─────────────────────┼─────────────────────┤
    │ Dielectric Constant │    1.8 - 2.0        │    > 2.0 - 10.0     │
    │                     │    (typical: 1.9)   │    (varies)         │
    ├─────────────────────┼─────────────────────┼─────────────────────┤
    │ Sound Velocity      │    1250 - 1350 m/s  │    < 1200 m/s or    │
    │                     │    (typical: 1300)  │    > 1400 m/s       │
    ├─────────────────────┼─────────────────────┼─────────────────────┤
    │ Conductivity        │    0.1 - 0.5 μS/cm  │    > 1.0 μS/cm      │
    │                     │    (typical: 0.3)   │    (varies widely)  │
    └─────────────────────┴─────────────────────┴─────────────────────┘
    
    Key Indicators of Adulteration:
    ───────────────────────────────
    • Kerosene: ↑ Dielectric, ↓ Velocity, slight ↑ Conductivity
    • Water:    ↑↑ Dielectric, ↑ Velocity, ↑↑↑ Conductivity
    • Diesel:   ↑ Dielectric, ↓ Velocity, ↑ Conductivity
    """)
    
    # ============================================================
    # SAMPLE TEST DATA
    # ============================================================
    print("=" * 70)
    print("🧪 SAMPLE TEST DATA")
    print("=" * 70)
    
    test_samples = [
        # Format: (name, dielectric, velocity, conductivity, expected)
        ("Pure Petrol (Good)", 1.9, 1300, 0.3, "PURE"),
        ("Pure Petrol (Edge)", 2.0, 1280, 0.5, "PURE"),
        ("5% Kerosene Mix", 2.1, 1250, 0.8, "ADULTERATED"),
        ("10% Kerosene Mix", 2.3, 1200, 1.2, "ADULTERATED"),
        ("20% Kerosene Mix", 2.5, 1100, 2.5, "ADULTERATED"),
        ("2% Water Mix", 3.0, 1320, 15.0, "ADULTERATED"),
        ("5% Water Mix", 5.0, 1350, 30.0, "ADULTERATED"),
        ("10% Diesel Mix", 2.2, 1220, 1.5, "ADULTERATED"),
        ("Low Quality Petrol", 2.05, 1270, 0.6, "BORDERLINE"),
    ]
    
    print("\nRunning predictions on sample data...\n")
    print(f"{'Sample Name':<25} {'Die.':<6} {'Vel.':<8} {'Cond.':<8} {'Score':<8} {'Result':<15}")
    print("-" * 70)
    
    for name, die, vel, cond, expected in test_samples:
        pred = predict(interpreter, die, vel, cond)
        result, _ = classify(pred)
        print(f"{name:<25} {die:<6.1f} {vel:<8.0f} {cond:<8.1f} {pred:<8.4f} {result:<15}")
    
    print("-" * 70)
    print("\n📌 Score Interpretation:")
    print("   • Score < 0.3  →  Definitely PURE")
    print("   • Score 0.3-0.5 →  Likely PURE (borderline)")
    print("   • Score 0.5-0.7 →  Likely ADULTERATED (borderline)")
    print("   • Score > 0.7  →  Definitely ADULTERATED")
    
    # ============================================================
    # INTERACTIVE TESTING
    # ============================================================
    print("\n" + "=" * 70)
    print("🔬 INTERACTIVE TESTING MODE")
    print("=" * 70)
    print("Enter your own sensor values to test the model.")
    print("Type 'quit' to exit.\n")
    
    while True:
        try:
            user_input = input("\nEnter values (dielectric, velocity, conductivity) or 'quit': ")
            
            if user_input.lower() == 'quit':
                print("\nGoodbye! 👋")
                break
            
            # Parse input
            values = [float(x.strip()) for x in user_input.split(',')]
            if len(values) != 3:
                print("❌ Please enter exactly 3 values separated by commas")
                continue
            
            die, vel, cond = values
            
            # Validate ranges
            if not (1.0 <= die <= 100):
                print("⚠️  Warning: Dielectric should be 1-100")
            if not (500 <= vel <= 2000):
                print("⚠️  Warning: Velocity should be 500-2000 m/s")
            if not (0 <= cond <= 1000):
                print("⚠️  Warning: Conductivity should be 0-1000 μS/cm")
            
            # Predict
            pred = predict(interpreter, die, vel, cond)
            result, _ = classify(pred)
            
            print(f"\n   📊 Input: Dielectric={die}, Velocity={vel}, Conductivity={cond}")
            print(f"   🎯 Prediction Score: {pred:.4f}")
            print(f"   📝 Classification: {result}")
            
            if pred < 0.3:
                print("   💬 High confidence: This fuel appears to be pure.")
            elif pred < 0.5:
                print("   💬 Low confidence: This fuel might be pure, but borderline.")
            elif pred < 0.7:
                print("   💬 Low confidence: This fuel might be adulterated.")
            else:
                print("   💬 High confidence: This fuel is likely adulterated!")
                
        except ValueError:
            print("❌ Invalid input. Please enter 3 numbers separated by commas.")
        except KeyboardInterrupt:
            print("\n\nExiting...")
            break

if __name__ == "__main__":
    main()
