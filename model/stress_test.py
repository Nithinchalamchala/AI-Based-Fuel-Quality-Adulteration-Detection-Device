"""
Model Stress Test
Evaluates model on difficult "borderline" cases to check robustness.
"""
import numpy as np
import tensorflow as tf
import os
import pandas as pd

# Load model
MODEL_DIR = os.path.dirname(__file__)
TFLITE_MODEL_PATH = os.path.join(MODEL_DIR, 'fuel_model.tflite')

# Scaler Params (Mean and Std Dev)
SCALER_MEAN = np.array([2.371911, 1289.732665, 4.567489])
SCALER_STD = np.array([1.063877, 35.678684, 10.728714])

def stress_test():
    interpreter = tf.lite.Interpreter(model_path=TFLITE_MODEL_PATH)
    interpreter.allocate_tensors()
    input_details = interpreter.get_input_details()
    output_details = interpreter.get_output_details()
    
    # Define tricky test cases
    # Pure limits: Dielectric < 2.0, Velocity > 1250, Cond < 0.5
    test_cases = [
        # Name                        Dielectric  Velocity  Cond    Expected
        ("Perfect Pure",              1.9,        1300,     0.3,    0), # Clear Pure
        ("Slightly Impure (Safe)",    1.98,       1280,     0.45,   0), # Edge Pure
        ("Very Slight Kerosene",      2.05,       1240,     0.6,    1), # Edge Adulterated
        ("High Dielectric Only",      2.2,        1300,     0.3,    1), # Adulterated (Dielectric)
        ("Low Velocity Only",         1.9,        1200,     0.3,    1), # Adulterated (Velocity)
        ("High Conductivity Only",    1.9,        1300,     1.5,    1), # Adulterated (Cond)
        ("Water Drop (Huge change)",  5.0,        1400,     20.0,   1)  # Clear Adulterated
    ]
    
    print(f"{'Test Case':<25} {'Pred Score':<12} {'Result':<10} {'Expected':<10} {'Status'}")
    print("-" * 75)
    
    failures = 0
    for name, d, v, c, expected in test_cases:
        # Normalize
        features = np.array([[d, v, c]])
        norm = (features - SCALER_MEAN) / SCALER_STD
        
        # Run inference
        interpreter.set_tensor(input_details[0]['index'], norm.astype(np.float32))
        interpreter.invoke()
        pred = interpreter.get_tensor(output_details[0]['index'])[0][0]
        
        result_class = 1 if pred > 0.5 else 0
        status = "✅ PASS" if result_class == expected else "❌ FAIL"
        if result_class != expected: failures += 1
        
        result_str = "Adulterated" if result_class == 1 else "Pure"
        expected_str = "Adulterated" if expected == 1 else "Pure"
        
        print(f"{name:<25} {pred:<12.4f} {result_str:<10} {expected_str:<10} {status}")

    print("-" * 75)
    if failures == 0:
        print("🎉 Model Passed All Stress Tests!")
    else:
        print(f"⚠️  Model Failed {failures} Tests. It struggles with borderline cases.")

if __name__ == "__main__":
    stress_test()
