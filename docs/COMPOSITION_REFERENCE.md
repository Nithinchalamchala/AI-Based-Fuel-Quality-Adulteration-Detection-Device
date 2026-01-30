# Fuel Composition & Sensor Reference Values

This document provides a reference for the expected sensor readings for different fuel types and common adulteration scenarios. Use this guide to calibrate your sensors and verify your model's detection capabilities.

## 1. Pure Petrol Protocol (Baseline)
The "Gold Standard" values for pure, unadulterated petrol.

| Parameter | Expected Range | Typical Value | Notes |
|-----------|----------------|---------------|-------|
| **Dielectric Constant** | 1.80 – 2.00 | 1.95 | Low polarity hydrocarbon. |
| **Sound Velocity** | 1250 – 1350 m/s | 1290 m/s | varies slightly with temperature. |
| **Conductivity** | 0.00 – 0.50 μS/cm | 0.20 μS/cm | Excellent insulator. |
| **Model Classification** | **< 0.5 (PURE)** | **0.1 - 0.3** | |

---

## 2. Common Adulterants
How sensor values shift when "Foreign Particles" are added.

### A. Kerosene Adulteration (10% - 20% mix)
Often used due to lower cost.
| Parameter | Change vs Pure | Typical Range |
|-----------|----------------|---------------|
| **Dielectric** | Slight Increase (↑) | 2.1 – 2.4 |
| **Velocity** | Decrease (↓) | 1150 – 1250 m/s |
| **Conductivity** | Minimal Change | 0.5 – 1.0 μS/cm |
| **Detection** | **ADULTERATED** | Detected mostly by velocity drop. |

### B. Diesel Mixing (10% - 20% mix)
| Parameter | Change vs Pure | Typical Range |
|-----------|----------------|---------------|
| **Dielectric** | Increase (↑) | 2.2 – 2.5 |
| **Velocity** | Decrease (↓) | 1200 – 1280 m/s |
| **Conductivity** | Slight Increase | 0.5 – 1.5 μS/cm |
| **Detection** | **ADULTERATED** | Detected by dielectric shift. |

### C. Water Contamination (1% - 5% mix)
Accidental or intentional. Very easy to detect.
| Parameter | Change vs Pure | Typical Range |
|-----------|----------------|---------------|
| **Dielectric** | Huge Increase (↑↑↑) | > 5.0 |
| **Velocity** | Increase (↑) | > 1350 m/s |
| **Conductivity** | Huge Increase (↑↑↑) | > 10.0 μS/cm |
| **Detection** | **ADULTERATED** | Instantly detected (100% confidence). |

---

## 3. Ethanol Blends (E10, E20)
Government standard blends (E10 = 10% Ethanol).
*Note: Our current model is trained to flag these as "Adulterated" if they deviate from pure petrol, but you can retrain to accept E10 as "Pure" if needed.*

| Composition | Dielectric | Velocity | Conductivity |
|-------------|------------|----------|--------------|
| **E10 (10% Ethanol)** | ~3.0 – 4.0 | ~1250 m/s | 0.5 – 2.0 μS/cm |
| **E20 (20% Ethanol)** | ~5.0 – 7.0 | ~1200 m/s | 1.0 – 3.0 μS/cm |
| **Pure Ethanol** | ~24.0 | ~1140 m/s | Varies (absorbs water) |

**Sensor Physics Note:** Ethanol is polar, so it drastically increases the Dielectric Constant compared to non-polar petrol.

---

## 4. Summary Chart for Testing

| Scenario | Dielectric | Velocity | Conductivity | Expected Result |
|----------|:----------:|:--------:|:------------:|:---------------:|
| **Pure Petrol** | **1.9** | **1300** | **0.3** | ✅ **PURE** |
| Mix: Kerosene | 2.4 | 1200 | 0.8 | ❌ **ADULTERATED** |
| Mix: Water | 8.5 | 1400 | 25.0 | ❌ **ADULTERATED** |
| Mix: Ethanol (E20) | 6.0 | 1210 | 1.5 | ❌ **ADULTERATED** |

*Use these values in the Streamlit App or `test_model.py` to demonstrate detection.*
