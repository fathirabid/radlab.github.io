import pandas as pd
from sklearn.tree import DecisionTreeClassifier
from sklearn.preprocessing import LabelEncoder
from sklearn.tree import export_text

# =========================
# LOAD DATASET
# =========================
data = pd.read_csv("eye_strain.csv")

X = data[['lux_avg', 'time_min']]
y = data['label']

# =========================
# ENCODE LABEL
# =========================
encoder = LabelEncoder()
y_encoded = encoder.fit_transform(y)

# =========================
# TRAIN MODEL
# =========================
model = DecisionTreeClassifier(
    max_depth=3,
    criterion="gini",
    random_state=42
)

model.fit(X, y_encoded)

# =========================
# TAMPILKAN ATURAN (RULE)
# =========================
rules = export_text(
    model,
    feature_names=['lux_avg', 'time_min']
)

print("===== RULE HASIL MACHINE LEARNING =====")
print(rules)

# =========================
# TEST PREDIKSI MANUAL
# =========================
test_data = [[250, 20], [400, 30], [650, 50]]
pred = model.predict(test_data)

print("\n===== HASIL UJI COBA =====")
for i, p in enumerate(pred):
    print(f"Lux={test_data[i][0]}, Time={test_data[i][1]} -> {encoder.inverse_transform([p])[0]}")

