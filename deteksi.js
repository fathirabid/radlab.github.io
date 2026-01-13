/**
 * RADLAB - Eye Strain Detection System (Final Version for Socialization)
 */

/* ===============================
   1. KONFIGURASI GLOBAL & ML
=============================== */
const SCALER_MEANS = [328.4139841688654, 171.19045334612616, 63982.37949148477, 5.380336927108956, 193933.28950587672]; 
const SCALER_SCALES = [293.38974847156624, 100.04697239330497, 86582.92174117625, 1.0146251243960756, 374267.0206409999];

let session; 
let port, reader, writer;

const MAX_POINTS = 500;
const MAX_LUX = 1000;
let timeCounter = 0;
let luxLineData = [];
let statusBarData = [];

// Inisialisasi Canvas setelah DOM Load
let lineCanvas, lineCtx, barCanvas, barCtx;

window.onload = () => {
    lineCanvas = document.getElementById("lineChart");
    lineCtx = lineCanvas.getContext("2d");
    barCanvas = document.getElementById("barChart");
    barCtx = barCanvas.getContext("2d");
    initModel();
};

/* ===============================
   2. INISIALISASI MODEL ML
=============================== */
async function initModel() {
    try {
        console.log("🧠 Memuat model ONNX...");
        session = await ort.InferenceSession.create('C:\Users\HP\OneDrive\Documents\GitHub\RADLAB\model_esi.onnx');
        console.log("✅ Model ESI berhasil dimuat!");
    } catch (e) {
        console.error("❌ Gagal memuat model:", e);
    }
}

/* ===============================
   3. LOGIKA PREDIKSI ML
=============================== */
async function getMLPrediction(lux, waktu) {
    if (!session) return 0;
    const lux_time = lux * waktu;
    const log_lux = Math.log(lux + 1);
    const lux_squared = Math.pow(lux, 2);
    const rawFeatures = [lux, waktu, lux_time, log_lux, lux_squared];
    const scaledFeatures = rawFeatures.map((val, i) => (val - SCALER_MEANS[i]) / SCALER_SCALES[i]);

    try {
        const inputTensor = new ort.Tensor('float32', new Float32Array(scaledFeatures), [1, 5]);
        const feeds = { float_input: inputTensor };
        const results = await session.run(feeds);
        const output = results[Object.keys(results)[0]]; 
        return Math.max(0, output.data[0]); 
    } catch (e) { return 0; }
}

/* ===============================
   4. HANDLE DATA ARDUINO
=============================== */
async function handleArduinoData(line) {
    const parts = line.split("|").map(p => p.trim());
    if (parts.length < 2) return;

    let serialData = {};
    parts.forEach(part => {
        if (part.includes(":")) {
            let [key, val] = part.split(":");
            serialData[key.trim()] = val.trim();
        }
    });

    const lux = parseFloat(serialData["Lux"]) || 0;
    // Mengambil angka saja dari string waktu
    const waktu = parseFloat(serialData["Waktu"] ? serialData["Waktu"].replace(/[^\d.]/g, '') : 0);
    const jarak = serialData["Jarak"] || "-";

    // HITUNG ESI DENGAN MACHINE LEARNING
    const esiML = await getMLPrediction(lux, waktu);

    // UPDATE UI
    document.getElementById("luxValue").textContent = lux.toFixed(1);
    document.getElementById("distanceInfo").textContent = "Estimasi jarak: " + jarak;
    document.getElementById("exposureDuration").textContent = "Durasi Paparan: " + waktu + " menit";
    document.getElementById("esiNumber").textContent = esiML.toFixed(2);

    updateEsiIndicator(esiML);
    
    // KIRIM PERINTAH KE ARDUINO UNTUK LED
    sendFeedbackToArduino(esiML);

    const category = getEsiCategory(esiML);
    updateCharts(lux, category);
}

// Fungsi kirim perintah ke Arduino
async function sendFeedbackToArduino(esi) {
    if (!writer) return;
    let cmd = "";
    if (esi < 0.5) cmd = "NYAMAN\n";
    else if (esi < 1.0) cmd = "WASPARA\n";
    else if (esi < 2.0) cmd = "LELAH\n";
    else cmd = "BAHAYA\n";

    try {
        await writer.write(new TextEncoder().encode(cmd));
    } catch (e) { console.error("Gagal kirim LED", e); }
}

function getEsiCategory(esi) {
    if (esi < 0.5) return "NYAMAN";
    if (esi < 1.0) return "MULAI LELAH";
    if (esi < 2.0) return "LELAH";
    return "SANGAT LELAH";
}

function updateEsiIndicator(esi) {
    const indicator = document.getElementById("esiIndicator");
    const numberBox = document.getElementById("esiNumber");
    indicator.className = "indikator-string";
    numberBox.className = "angka-esi";

    if (esi < 0.5) {
        indicator.textContent = "NYAMAN";
        indicator.classList.add("aman");
        numberBox.classList.add("aman");
    } else if (esi < 1.0) {
        indicator.textContent = "MULAI LELAH";
        indicator.classList.add("waspada");
        numberBox.classList.add("waspada");
    } else if (esi < 2.0) {
        indicator.textContent = "LELAH";
        indicator.classList.add("lelah");
        numberBox.classList.add("lelah");
    } else {
        indicator.textContent = "SANGAT LELAH";
        indicator.classList.add("sangat-lelah");
        numberBox.classList.add("sangat-lelah");
    }
}

/* ===============================
   5. KONEKSI SERIAL (TOMBOL HUBUNGKAN)
=============================== */
async function connectSerial() {
    if (!("serial" in navigator)) { 
        alert("Browser tidak mendukung Web Serial. Gunakan Chrome di Laptop."); 
        return; 
    }
    try {
        port = await navigator.serial.requestPort();
        await port.open({ baudRate: 9600 });
        writer = port.writable.getWriter();
        reader = port.readable.getReader();
        readSerialLoop();
    } catch (err) { console.log("User cancel/error"); }
}

async function readSerialLoop() {
    let buffer = "";
    while (true) {
        const { value, done } = await reader.read();
        if (done) break;
        buffer += new TextDecoder().decode(value);
        const lines = buffer.split("\n");
        buffer = lines.pop();
        lines.forEach(line => { if (line.trim()) handleArduinoData(line.trim()); });
    }
}

async function sendFeedbackToArduino(esi) {
    if (!writer) return;
    let cmd = esi < 0.5 ? "NYAMAN\n" : esi < 1.0 ? "WASPARA\n" : esi < 2.0 ? "LELAH\n" : "BAHAYA\n";
    try { await writer.write(new TextEncoder().encode(cmd)); } catch (e) {}
}

function updateCharts(luxValue, status) {
    luxLineData.push({ time: timeCounter++, lux: luxValue, status: status });
    if (luxLineData.length > MAX_POINTS) luxLineData.shift();
    statusBarData = [{ value: luxValue, status: status }];
    drawLineChart();
    drawBarChart();
}

function drawLineChart() {
    if (!lineCtx) return;
    const ctx = lineCtx;
    ctx.clearRect(0, 0, lineCanvas.width, lineCanvas.height);
    const p = 50; // Padding sedikit diperbesar untuk ruang teks axis
    const w = lineCanvas.width - p * 2;
    const h = lineCanvas.height - p * 2;

    // 1. Gambar Sumbu (Axis)
    ctx.strokeStyle = "#9ca3af";
    ctx.lineWidth = 1.5;
    ctx.beginPath();
    ctx.moveTo(p, p); 
    ctx.lineTo(p, p + h); // Sumbu Y
    ctx.lineTo(p + w, p + h); // Sumbu X
    ctx.stroke();

    // 2. Tambahkan Nama Axis (Label)
    ctx.fillStyle = "#9ca3af";
    ctx.font = "12px Roboto";
    
    // Label Sumbu Y (Vertikal)
    ctx.save();
    ctx.translate(p - 35, p + h / 2);
    ctx.rotate(-Math.PI / 2);
    ctx.textAlign = "center";
    ctx.fillText("Intensitas (Lux / ESI)", 0, 0);
    ctx.restore();

    // Label Sumbu X (Horizontal)
    ctx.textAlign = "center";
    ctx.fillText("Waktu Paparan (Simulasi)", p + w / 2, p + h + 35);

    // 3. Skala Dinamis
    const currentMaxLux = Math.max(...luxLineData.map(d => d.lux), 100);
    const yMax = currentMaxLux * 1.2; 
    const maxT = Math.max(...luxLineData.map(d => d.time), MAX_POINTS);

    // 4. Gambar Garis Data
    ctx.lineWidth = 2.5;
    ctx.lineJoin = "round";

    if (luxLineData.length > 0) {
        ctx.beginPath();
        luxLineData.forEach((d, i) => {
            const x = p + (d.time / maxT) * w;
            const y = p + h - (d.lux / yMax) * h;

            if (i === 0) ctx.moveTo(x, y);
            else {
                ctx.strokeStyle = d.status === "NYAMAN" ? "#22c55e" : 
                                 d.status === "MULAI LELAH" ? "#3b82f6" : 
                                 d.status === "LELAH" ? "#facc15" : "#ef4444";
                ctx.lineTo(x, y);
                ctx.stroke();
                ctx.beginPath();
                ctx.moveTo(x, y);
            }
        });
    }
}

function drawBarChart() {
    if (!barCtx) return;
    const ctx = barCtx;
    ctx.clearRect(0, 0, barCanvas.width, barCanvas.height);
    
    if (statusBarData.length > 0) {
        const d = statusBarData[0];
        // Skala bar (anggap 1000 lux adalah bar penuh)
        const barW = (Math.min(d.value, MAX_LUX) / MAX_LUX) * (barCanvas.width - 40);
        
        // Gradien Warna
        const gradient = ctx.createLinearGradient(0, 0, barCanvas.width, 0);
        gradient.addColorStop(0, "#22c55e"); // Hijau
        gradient.addColorStop(0.5, "#facc15"); // Kuning
        gradient.addColorStop(1, "#ef4444"); // Merah

        ctx.fillStyle = gradient;
        ctx.roundRect(20, 15, barW, 30, 8);
        ctx.fill();
        
        // Teks nilai lux di atas bar
        ctx.fillStyle = "#ffffff";
        ctx.font = "bold 12px Roboto";
        ctx.fillText(d.value.toFixed(1) + " LUX", 25, 35);
    }
}

function resetSimulation() { location.reload(); }