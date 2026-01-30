"""
Synthetic Dataset Generator for Fuel Quality Detection
Generates training data based on research paper values for petrol properties
"""

import numpy as np
import pandas as pd
import os

# Seed for reproducibility
np.random.seed(42)

# Number of samples per class
SAMPLES_PER_CLASS = 300

def generate_pure_petrol_samples(n_samples):
    """
    Generate synthetic data for pure petrol
    Reference values from research literature:
    - Dielectric constant: 1.8 - 2.0
    - Sound velocity: 1250-1350 m/s
    - Conductivity: 0.1-0.5 μS/cm
    """
    data = {
        'dielectric': np.random.normal(1.9, 0.05, n_samples),  # Mean 1.9, std 0.05
        'velocity': np.random.normal(1300, 25, n_samples),      # Mean 1300 m/s, std 25
        'conductivity': np.random.normal(0.3, 0.1, n_samples),  # Mean 0.3 μS/cm, std 0.1
        'label': 0  # 0 = Pure
    }
    return pd.DataFrame(data)

def generate_kerosene_adulterated_samples(n_samples):
    """
    Generate synthetic data for kerosene-adulterated petrol (5-25% adulteration)
    Kerosene has higher dielectric (~1.8-2.2) and lower sound velocity
    """
    adulteration_level = np.random.uniform(0.05, 0.25, n_samples)  # 5-25%
    
    data = {
        'dielectric': np.random.normal(1.9, 0.05, n_samples) + adulteration_level * 1.5,
        'velocity': np.random.normal(1300, 25, n_samples) - adulteration_level * 400,
        'conductivity': np.random.normal(0.3, 0.1, n_samples) + adulteration_level * 2,
        'label': 1  # 1 = Adulterated
    }
    return pd.DataFrame(data)

def generate_water_adulterated_samples(n_samples):
    """
    Generate synthetic data for water-adulterated petrol (1-10% adulteration)
    Water has very high dielectric (~80) and different sound velocity
    """
    adulteration_level = np.random.uniform(0.01, 0.10, n_samples)  # 1-10%
    
    data = {
        'dielectric': np.random.normal(1.9, 0.05, n_samples) + adulteration_level * 50,
        'velocity': np.random.normal(1300, 25, n_samples) + adulteration_level * 300,
        'conductivity': np.random.normal(0.3, 0.1, n_samples) + adulteration_level * 500,
        'label': 1  # 1 = Adulterated
    }
    return pd.DataFrame(data)

def generate_diesel_adulterated_samples(n_samples):
    """
    Generate synthetic data for diesel-adulterated petrol (5-20% adulteration)
    Diesel has slightly higher dielectric and different acoustic properties
    """
    adulteration_level = np.random.uniform(0.05, 0.20, n_samples)  # 5-20%
    
    data = {
        'dielectric': np.random.normal(1.9, 0.05, n_samples) + adulteration_level * 0.8,
        'velocity': np.random.normal(1300, 25, n_samples) - adulteration_level * 150,
        'conductivity': np.random.normal(0.3, 0.1, n_samples) + adulteration_level * 3,
        'label': 1  # 1 = Adulterated
    }
    return pd.DataFrame(data)

def clip_values(df):
    """Ensure values are within realistic physical bounds"""
    df['dielectric'] = df['dielectric'].clip(1.5, 15)
    df['velocity'] = df['velocity'].clip(800, 1600)
    df['conductivity'] = df['conductivity'].clip(0.01, 100)
    return df

def main():
    print("=" * 50)
    print("Fuel Quality Synthetic Dataset Generator")
    print("=" * 50)
    
    # Generate samples
    print("\nGenerating pure petrol samples...")
    pure = generate_pure_petrol_samples(SAMPLES_PER_CLASS)
    
    print("Generating kerosene-adulterated samples...")
    kerosene_adulterated = generate_kerosene_adulterated_samples(SAMPLES_PER_CLASS // 3)
    
    print("Generating water-adulterated samples...")
    water_adulterated = generate_water_adulterated_samples(SAMPLES_PER_CLASS // 3)
    
    print("Generating diesel-adulterated samples...")
    diesel_adulterated = generate_diesel_adulterated_samples(SAMPLES_PER_CLASS // 3)
    
    # Combine all data
    df = pd.concat([pure, kerosene_adulterated, water_adulterated, diesel_adulterated], 
                   ignore_index=True)
    
    # Clip values to realistic bounds
    df = clip_values(df)
    
    # Shuffle the dataset
    df = df.sample(frac=1, random_state=42).reset_index(drop=True)
    
    # Save to CSV
    output_path = os.path.join(os.path.dirname(__file__), 'fuel_dataset.csv')
    df.to_csv(output_path, index=False)
    
    print(f"\n✅ Dataset saved to: {output_path}")
    print(f"\n📊 Dataset Statistics:")
    print(f"   Total samples: {len(df)}")
    print(f"   Pure samples (label=0): {len(df[df['label'] == 0])}")
    print(f"   Adulterated samples (label=1): {len(df[df['label'] == 1])}")
    
    print(f"\n📈 Feature Statistics:")
    print(df.describe().round(3))
    
    return df

if __name__ == "__main__":
    df = main()
