"""
ML Model Training for Fuel Quality Detection
Binary Classification: Pure (0) vs Adulterated (1)
Target: TensorFlow Lite deployment on ESP32
"""

import numpy as np
import pandas as pd
import os
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
import pickle

# Constants
MODEL_DIR = os.path.dirname(__file__)
DATA_PATH = os.path.join(os.path.dirname(__file__), '..', 'data', 'fuel_dataset.csv')

def load_and_preprocess_data():
    """Load dataset and prepare for training"""
    print("Loading dataset...")
    df = pd.read_csv(DATA_PATH)
    
    # Features and labels
    X = df[['dielectric', 'velocity', 'conductivity']].values
    y = df['label'].values
    
    # Split data
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )
    
    # Normalize features
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    
    # Save scaler for later use
    scaler_path = os.path.join(MODEL_DIR, 'scaler.pkl')
    with open(scaler_path, 'wb') as f:
        pickle.dump(scaler, f)
    print(f"✅ Scaler saved to: {scaler_path}")
    
    # Also save scaler parameters as text for ESP32
    scaler_params_path = os.path.join(MODEL_DIR, 'scaler_params.txt')
    with open(scaler_params_path, 'w') as f:
        f.write("# Scaler parameters for ESP32\n")
        f.write("# Format: mean_dielectric, mean_velocity, mean_conductivity\n")
        f.write(f"MEAN = [{scaler.mean_[0]:.6f}, {scaler.mean_[1]:.6f}, {scaler.mean_[2]:.6f}]\n")
        f.write(f"STD = [{scaler.scale_[0]:.6f}, {scaler.scale_[1]:.6f}, {scaler.scale_[2]:.6f}]\n")
    print(f"✅ Scaler params saved to: {scaler_params_path}")
    
    return X_train_scaled, X_test_scaled, y_train, y_test, scaler

def build_model():
    """
    Build a small neural network suitable for ESP32
    - Small enough to fit in ESP32 memory (~520KB SRAM)
    - Fast inference time
    """
    model = keras.Sequential([
        layers.Input(shape=(3,), name='sensor_input'),
        layers.Dense(8, activation='relu', name='hidden1'),
        layers.Dense(4, activation='relu', name='hidden2'),
        layers.Dense(1, activation='sigmoid', name='output')
    ])
    
    model.compile(
        optimizer=keras.optimizers.Adam(learning_rate=0.001),
        loss='binary_crossentropy',
        metrics=['accuracy']
    )
    
    return model

def train_model(model, X_train, y_train, X_test, y_test):
    """Train the model with early stopping"""
    print("\n Training model...")
    
    early_stopping = keras.callbacks.EarlyStopping(
        monitor='val_accuracy',
        patience=20,
        restore_best_weights=True
    )
    
    history = model.fit(
        X_train, y_train,
        epochs=200,
        batch_size=16,
        validation_data=(X_test, y_test),
        callbacks=[early_stopping],
        verbose=1
    )
    
    return history

def evaluate_model(model, X_test, y_test):
    """Evaluate model performance"""
    print("\n📊 Model Evaluation:")
    
    loss, accuracy = model.evaluate(X_test, y_test, verbose=0)
    print(f"   Test Loss: {loss:.4f}")
    print(f"   Test Accuracy: {accuracy*100:.2f}%")
    
    # Predictions
    y_pred = (model.predict(X_test) > 0.5).astype(int).flatten()
    
    # Confusion matrix
    from sklearn.metrics import confusion_matrix, classification_report
    cm = confusion_matrix(y_test, y_pred)
    print(f"\n   Confusion Matrix:")
    print(f"   [[TN={cm[0][0]:3d}, FP={cm[0][1]:3d}]")
    print(f"    [FN={cm[1][0]:3d}, TP={cm[1][1]:3d}]]")
    
    print(f"\n   Classification Report:")
    print(classification_report(y_test, y_pred, target_names=['Pure', 'Adulterated']))
    
    return accuracy

def save_model(model):
    """Save model in multiple formats"""
    # Save Keras model
    keras_path = os.path.join(MODEL_DIR, 'fuel_model.h5')
    model.save(keras_path)
    print(f"\n✅ Keras model saved to: {keras_path}")
    
    # Save model summary
    model.summary()
    
    return keras_path

def main():
    print("=" * 60)
    print("Fuel Quality Detection - Model Training")
    print("=" * 60)
    
    # Check if dataset exists
    if not os.path.exists(DATA_PATH):
        print(f"❌ Dataset not found at: {DATA_PATH}")
        print("   Please run synthetic_data.py first!")
        return None
    
    # Load data
    X_train, X_test, y_train, y_test, scaler = load_and_preprocess_data()
    print(f"\n📊 Data shapes:")
    print(f"   Training: {X_train.shape}")
    print(f"   Testing: {X_test.shape}")
    
    # Build model
    model = build_model()
    print("\n🏗️ Model Architecture:")
    model.summary()
    
    # Train
    history = train_model(model, X_train, y_train, X_test, y_test)
    
    # Evaluate
    accuracy = evaluate_model(model, X_test, y_test)
    
    if accuracy >= 0.90:
        print(f"\n🎉 Target accuracy (90%) achieved! Accuracy: {accuracy*100:.2f}%")
    else:
        print(f"\n⚠️ Accuracy ({accuracy*100:.2f}%) below target (90%). Consider tuning.")
    
    # Save model
    save_model(model)
    
    print("\n" + "=" * 60)
    print("Training complete! Next step: Run convert_to_tflite.py")
    print("=" * 60)
    
    return model

if __name__ == "__main__":
    model = main()
