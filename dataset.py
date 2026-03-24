import numpy as np
import pandas as pd

np.random.seed(42)
n = 6000

terrain = np.random.choice(["plains","hills","mountains"], n)

soil_risk_map = {"clay": 0.9, "sand": 0.5, "rock": 0.1}

rainfall, slope, elevation, vegetation, earthquake, soil_type = [], [], [], [], [], []

for t in terrain:
    if t == "plains":
        rainfall.append(np.random.uniform(50,150))
        slope.append(np.random.uniform(2,10))
        elevation.append(np.random.uniform(100,500))
        vegetation.append(np.random.uniform(0.4,0.9))
        earthquake.append(np.random.uniform(0,3))
        soil_type.append(np.random.choice(["sand", "clay"], p=[0.7, 0.3]))

    elif t == "hills":
        rainfall.append(np.random.uniform(100,220))
        slope.append(np.random.uniform(10,25))
        elevation.append(np.random.uniform(500,1500))
        vegetation.append(np.random.uniform(0.3,0.8))
        earthquake.append(np.random.uniform(1,5))
        soil_type.append(np.random.choice(["sand", "clay", "rock"], p=[0.3, 0.4, 0.3]))

    else:  # mountains
        rainfall.append(np.random.uniform(150,300))
        slope.append(np.random.uniform(25,50))
        elevation.append(np.random.uniform(1500,4000))
        vegetation.append(np.random.uniform(0.1,0.6))
        earthquake.append(np.random.uniform(2,7))
        soil_type.append(np.random.choice(["rock", "clay"], p=[0.8, 0.2]))


rainfall = np.array(rainfall)
slope = np.array(slope)
elevation = np.array(elevation)
vegetation = np.array(vegetation)
earthquake = np.array(earthquake)
soil_numeric = np.array([soil_risk_map[s] for s in soil_type]) # Convert text to risk value

moisture_retention = rainfall/300 + np.random.normal(0,0.1,n)
moisture_retention = np.clip(moisture_retention,0,1)

risk_score = (
    0.20*(rainfall/300) +
    0.20*(slope/50) +
    0.15*soil_numeric +          
    0.10*(elevation/4000) +
    0.15*(earthquake/7) +
    0.20*moisture_retention -
    0.20*vegetation
)

unmeasured_noise = np.random.normal(0, 0.08, n) 
risk_score_noisy = risk_score + unmeasured_noise
landslide = (risk_score_noisy > 0.458).astype(int)

#  FINAL DATAFRAME ---
df = pd.DataFrame({
    "terrain": terrain,
    "soil_type": soil_type,      
    "rainfall": rainfall,
    "vegetation": vegetation,
    "slope": slope,
    "elevation": elevation,
    "earthquake_intensity": earthquake,
    "moisture_retention": moisture_retention,
    "landslide": landslide
})


df = pd.get_dummies(df, columns=['terrain', 'soil_type'])
df.to_csv("synthetic_landslide_dataset.csv", index=False)
print(df.describe())
