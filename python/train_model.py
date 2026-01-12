import pandas as pd
import numpy as np
import joblib
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, r2_score

# ===============================
# LOAD DATA
# ===============================
df = pd.read_csv("python/dataset_esi.csv", encoding="latin1")



# filtering noise (ilmiah)
df = df[(df["lux"] > 5) & (df["lux"] < 2000)]

# ===============================
# FEATURE ENGINEERING
# ===============================
df["lux_time"] = df["lux"] * df["waktu"]
df["log_lux"] = np.log(df["lux"] + 1)
df["lux_squared"] = df["lux"] ** 2

features = ["lux", "waktu", "lux_time", "log_lux", "lux_squared"]

X = df[features]
y = df["esi"]

# ===============================
# SCALING
# ===============================
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# ===============================
# SPLIT DATA
# ===============================
X_train, X_test, y_train, y_test = train_test_split(
    X_scaled, y, test_size=0.2, random_state=42
)

# ===============================
# TRAIN MODEL
# ===============================
model = RandomForestRegressor(
    n_estimators=200,
    max_depth=10,
    random_state=42
)
model.fit(X_train, y_train)

# ===============================
# EVALUATION
# ===============================
y_pred = model.predict(X_test)

print("MAE :", round(mean_absolute_error(y_test, y_pred), 2))
print("R²  :", round(r2_score(y_test, y_pred), 3))

cv = cross_val_score(model, X_scaled, y, cv=5, scoring="r2")
print("R² CV Mean:", round(cv.mean(), 3))

# ===============================
# SAVE MODEL ASSET
# ===============================
joblib.dump(model, "python/models/esi_model.pkl")
joblib.dump(scaler, "python/models/esi_scaler.pkl")
joblib.dump(features, "python/models/esi_features.pkl")

print("\n✅ Model, scaler, dan fitur disimpan")

# ===============================
# VISUALISASI
# ===============================
plt.scatter(y_test, y_pred, alpha=0.4)
plt.xlabel("ESI Aktual")
plt.ylabel("ESI Prediksi")
plt.title("Prediksi vs Aktual ESI")
plt.grid(True)
plt.show()
