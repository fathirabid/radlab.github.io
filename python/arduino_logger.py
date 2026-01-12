import serial
import time
import csv

PORT = "COM3"      # sesuaikan
BAUDRATE = 9600

# ===============================
# PARSE DATA DARI ARDUINO
# ===============================
def parse_line(line):
    data = {}

    try:
        parts = line.split("|")

        for p in parts:
            if ":" not in p:
                continue

            key, value = p.split(":", 1)
            data[key.strip()] = value.strip()

        # pastikan semua data ada
        if not all(k in data for k in ["Lux", "Waktu", "ESI", "Status", "Jarak"]):
            return None

        lux = float(data["Lux"])
        waktu = float(data["Waktu"].replace("mnt", "").strip())
        esi = float(data["ESI"])
        status = data["Status"]
        jarak = data["Jarak"]

        return lux, waktu, esi, status, jarak

    except Exception:
        print("SKIP:", line)
        return None


# ===============================
# MAIN PROGRAM
# ===============================
ser = serial.Serial(PORT, BAUDRATE, timeout=1)
time.sleep(2)

with open("dataset_esi.csv", mode="w", newline="") as file:
    writer = csv.writer(file)
    writer.writerow(["lux", "waktu", "esi", "status", "jarak"])

    print("📊 Logging data... tekan CTRL+C untuk stop")

    try:
        while True:
            line = ser.readline().decode("utf-8").strip()
            if not line:
                continue

            parsed = parse_line(line)
            if parsed is None:
                continue

            lux, waktu, esi, status, jarak = parsed
            writer.writerow([lux, waktu, esi, status, jarak])
            print(lux, waktu, esi, status, jarak)

    except KeyboardInterrupt:
        print("\n✅ Logging selesai")
        ser.close()
