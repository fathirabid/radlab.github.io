import serial
import joblib
import numpy as np
from flask import Flask, jsonify

# =============================
# LOAD MODEL
# =============================
model = joblib.load("model_esi.pkl")

# =============================
# SERIAL ARDUINO
# =============================
ser = serial.Serial("COM3", 9600, timeout=1)

# =============================
# FLASK
# =============================
app = Flask(__name__)
latest_data = {
    "lux": 0,
    "waktu": 0,
    "esi": 0
}

def predict_esi(lux, waktu):
    X = np.array([[lux, waktu, lux * waktu]])
    return float(model.predict(X)[0])

@app.route("/data")
def get_data():
    return jsonify(latest_data)

def read_serial():
    global latest_data
    while True:
        line = ser.readline().decode().strip()
        if not line:
            continue

        try:
            parts = dict(p.split(":") for p in line.split("|"))
            lux = float(parts["Lux"])
            waktu = float(parts["Waktu"])
            esi = predict_esi(lux, waktu)

            latest_data = {
                "lux": lux,
                "waktu": waktu,
                "esi": esi
            }
        except:
            pass

# =============================
# MAIN
# =============================
import threading
threading.Thread(target=read_serial, daemon=True).start()
app.run(debug=True)
