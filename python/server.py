import serial
import joblib
import json
import asyncio
import websockets
import numpy as np

# =====================
# LOAD MODEL
# =====================
model = joblib.load("python/models/esi_model.pkl")
scaler = joblib.load("python/models/esi_scaler.pkl")
features = joblib.load("python/models/esi_features.pkl")

# =====================
# SERIAL
# =====================
ser = serial.Serial("COM3", 9600, timeout=1)

# =====================
# PARSER
# =====================
def parse_line(line):
    data = {}
    parts = line.split("|")
    for p in parts:
        if ":" in p:
            k, v = p.split(":", 1)
            data[k.strip()] = v.strip()
    return data

# =====================
# WEBSOCKET
# =====================
async def send_data(ws):
    while True:
        line = ser.readline().decode(errors="ignore").strip()
        if not line or "Lux:" not in line:
            continue

        d = parse_line(line)

        lux = float(d["Lux"])
        waktu = float(d["Waktu"].replace("mnt", ""))

        # FEATURE ENGINEERING
        X = np.array([[lux, waktu]])
        X = scaler.transform(X)

        esi_pred = model.predict(X)[0]

        # STATUS ML
        if esi_pred < 0.5:
            status = "NYAMAN"
        elif esi_pred < 1.0:
            status = "MULAI LELAH"
        elif esi_pred < 2.0:
            status = "LELAH"
        else:
            status = "SANGAT LELAH"

        payload = {
            "lux": round(lux, 1),
            "waktu": waktu,
            "esi": round(esi_pred, 2),
            "status": status,
            "jarak": d.get("Jarak", "-")
        }

        await ws.send(json.dumps(payload))
        await asyncio.sleep(0.3)

async def main():
    async with websockets.serve(send_data, "localhost", 8765):
        print("🧠 ML Server aktif ws://localhost:8765")
        await asyncio.Future()

asyncio.run(main())
