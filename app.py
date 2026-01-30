"""
Fuel Quality Detection - Demo Interface
Run with: streamlit run app.py
"""

import streamlit as st
import numpy as np
import tensorflow as tf
import os
import pandas as pd
import time

# Page Config
st.set_page_config(
    page_title="Fuel Quality AI Detector",
    page_icon="⛽",
    layout="wide"
)

# Constants & Paths
MODEL_DIR = os.path.join(os.path.dirname(__file__), 'model')
TFLITE_MODEL_PATH = os.path.join(MODEL_DIR, 'fuel_model.tflite')

# Scaler Params (Mean and Std Dev from training)
SCALER_MEAN = np.array([2.371911, 1289.732665, 4.567489])
SCALER_STD = np.array([1.063877, 35.678684, 10.728714])

@st.cache_resource
def load_model():
    """Load TFLite model"""
    interpreter = tf.lite.Interpreter(model_path=TFLITE_MODEL_PATH)
    interpreter.allocate_tensors()
    return interpreter

def predict(interpreter, dielectric, velocity, conductivity):
    """Run inference"""
    # Normalize
    raw = np.array([[dielectric, velocity, conductivity]])
    norm = (raw - SCALER_MEAN) / SCALER_STD
    
    # Get io details
    input_details = interpreter.get_input_details()
    output_details = interpreter.get_output_details()
    
    # Inference
    interpreter.set_tensor(input_details[0]['index'], norm.astype(np.float32))
    interpreter.invoke()
    result = interpreter.get_tensor(output_details[0]['index'])[0][0]
    return result

# ==================== UI LAYOUT ====================

st.title("⛽ AI-Based Fuel Quality Detector")
st.markdown("### Interactive Prototype Demo")
st.markdown("Adjust the sliders below to simulate sensor readings and check fuel quality in real-time.")

col1, col2 = st.columns([1, 1])

with col1:
    st.markdown("#### 📡 Sensor Simulations")
    
    # Sliders for sensors with color coding
    dielectric = st.slider(
        "Dielectric Constant (Capacitive Sensor)",
        min_value=1.5,
        max_value=10.0,
        value=1.9,
        step=0.1,
        help="Pure Petrol: 1.8 - 2.0. Higher values indicate adulteration (e.g. water, kerosene)."
    )
    
    velocity = st.slider(
        "Sound Velocity (m/s) (Ultrasonic Sensor)",
        min_value=1000,
        max_value=1500,
        value=1300,
        step=10,
        help="Pure Petrol: ~1300 m/s. Lower -> Kerosene/Diesel. Higher -> Water."
    )
    
    conductivity = st.slider(
        "Conductivity (μS/cm) (TDS Sensor)",
        min_value=0.0,
        max_value=50.0,
        value=0.3,
        step=0.1,
        help="Pure Petrol: < 0.5. Higher values indicate impurities or water."
    )
    
    # Presets for quick demo
    st.markdown("#### ⚡ Quick Scenes")
    b_col1, b_col2, b_col3 = st.columns(3)
    
    if b_col1.button("Pure Petrol"):
        dielectric = 1.9
        velocity = 1300
        conductivity = 0.3
        st.toast("Loaded: Pure Petrol Data")
        
    if b_col2.button("Kerosene Mix"):
        dielectric = 2.4
        velocity = 1150
        conductivity = 2.5
        st.toast("Loaded: Kerosene Adulteration Data")
        
    if b_col3.button("Water Mix"):
        dielectric = 6.0
        velocity = 1380
        conductivity = 35.0
        st.toast("Loaded: Water Adulteration Data")

with col2:
    st.markdown("#### 🧠 AI Analysis")
    
    # Load model
    try:
        model = load_model()
        
        # Adding a small delay to simulate processing
        with st.spinner("Processing sensor data..."):
            time.sleep(0.3)
            prediction = predict(model, dielectric, velocity, conductivity)
            
        st.markdown("---")
        
        # Display Results
        if prediction < 0.5:
            st.success("## ✅ PURE FUEL DETECTED")
            st.metric("Quality Score", f"{(1-prediction)*100:.1f}% Pure")
            st.markdown(f"**Confidence Level:** Low probability of adulterants ({prediction:.4f})")
            st.image("https://img.icons8.com/color/96/gas-station.png", width=100)
        else:
            st.error("## ⚠️ ADULTERATED FUEL DETECTED")
            st.metric("Adulteration Probability", f"{prediction*100:.1f}%")
            st.markdown("**Assessment:** Fuel properties deviate significantly from standard parameters.")
            st.image("https://img.icons8.com/color/96/high-priority.png", width=100)
            
        st.markdown("---")
        
        # Gauge Chart (Progress Bar for Score)
        st.write("Adulteration Risk Gauge:")
        st.progress(float(prediction))
        
        st.info("""
        **Model Logic:**
        - **Input**: Normalized sensor array [D, V, C]
        - **Brain**: 3-layer TensorFlow Neural Network
        - **Output**: Probability Score (0.0 - 1.0)
        """)
        
    except Exception as e:
        st.error(f"Error loading model: {e}")
        st.warning("Make sure to run `python model/convert_to_tflite.py` first to generate the TFLite model.")

# Footer with technical details
st.markdown("---")
st.caption("Project: AI-Based Fuel Quality Adulteration Detection Device | Model Implementation: TensorFlow Lite Micro")
