/* ===============================
   CANVAS & DATA
=============================== */
const lineCanvas = document.getElementById("lineChart");
const lineCtx = lineCanvas.getContext("2d");
const barCanvas = document.getElementById("barChart");
const barCtx = barCanvas.getContext("2d");

const MAX_POINTS = 500;
const MAX_LUX = 1000;
let timeCounter = 0;

let luxLineData = [];
let statusBarData = [];

let port, reader, writer;

/* ===============================
   HANDLE DATA ARDUINO
=============================== */
function handleArduinoData(line) {
  const parts = line.split("|").map(p => p.trim());
  if (parts.length < 5) return;

  const lux = parseFloat(parts[0].split(":")[1]);
  const waktu = parts[1].split(":")[1].replace("mnt", "").trim();
  const esi = parseFloat(parts[2].split(":")[1]);
  const jarak = parts[4].split(":")[1].trim();

  // ===== UPDATE ANGKA =====
  document.getElementById("luxValue").textContent = lux.toFixed(1);
  document.getElementById("esiValue").textContent = waktu;       // waktu simulasi
  document.getElementById("esiNumber").textContent = esi.toFixed(2);

  // ===== UPDATE STATUS STRING =====
  updateEsiIndicator(esi);

  // ===== JARAK =====
  document.getElementById("distanceInfo").textContent =
    "Estimasi jarak: " + jarak;

  // ===== CHART =====
  const category = getEsiCategory(esi);
  updateCharts(lux, category);
}

/* ===============================
   ESI CATEGORY (untuk chart & warna)
=============================== */
function getEsiCategory(esi){
  if(esi < 0.5) return "NYAMAN";
  else if(esi < 1.0) return "MULAI LELAH";
  else if(esi < 2.0) return "LELAH";
  else return "SANGAT LELAH";
}

/* ===============================
   UPDATE ESI INDICATOR (string + warna)
=============================== */
function updateEsiIndicator(esi){
  const el = document.getElementById("esiIndicator");
  el.className = "esi-status"; // reset class
  let statusText = "";
  
  if(esi < 0.5){
    statusText = "NYAMAN";
    el.classList.add("aman");
  } else if(esi < 1.0){
    statusText = "MULAI LELAH";
    el.classList.add("waspada");
  } else if(esi < 2.0){
    statusText = "LELAH";
    el.classList.add("lelah");
  } else {
    statusText = "SANGAT LELAH";
    el.classList.add("sangat-lelah");
  }

  el.textContent = statusText;
}

/* ===============================
   UPDATE CHARTS
=============================== */
function updateCharts(luxValue, status){
  luxLineData.push({time: timeCounter, lux: luxValue, status: status});
  timeCounter++;
  if(luxLineData.length > MAX_POINTS) luxLineData.shift();

  statusBarData = [{value: luxValue, status: status}];

  drawLineChart();
  drawBarChart();
}

/* ===============================
   DRAW LINE CHART (Lux vs Time)
=============================== */
function drawLineChart(){
  const ctx = lineCtx;
  ctx.clearRect(0,0,lineCanvas.width,lineCanvas.height);

  const padding = 40;
  const w = lineCanvas.width - padding*2;
  const h = lineCanvas.height - padding*2;

  // GRID
  ctx.strokeStyle = "#e5e7eb";
  ctx.lineWidth = 1;
  ctx.setLineDash([3,5]);
  for(let i=0;i<=4;i++){
    const y = padding + (i/4)*h;
    ctx.beginPath();
    ctx.moveTo(padding, y);
    ctx.lineTo(padding + w, y);
    ctx.stroke();
  }
  ctx.setLineDash([]);

  // AXIS
  ctx.strokeStyle = "#9ca3af";
  ctx.lineWidth = 1.2;
  ctx.beginPath();
  ctx.moveTo(padding,padding);
  ctx.lineTo(padding,padding+h);
  ctx.lineTo(padding+w,padding+h);
  ctx.stroke();

  // LABELS
  ctx.fillStyle = "#374151";
  ctx.font = "12px Roboto";
  ctx.fillText("Lux", 5, padding-10);
  ctx.fillText("Waktu (detik)", lineCanvas.width/2 - 40, lineCanvas.height-5);

  const maxLux = Math.max(...luxLineData.map(d=>d.lux),1)*1.1;
  const maxTime = Math.max(...luxLineData.map(d=>d.time), MAX_POINTS);

  // AREA
  const gradient = ctx.createLinearGradient(0,padding,0,padding+h);
  gradient.addColorStop(0,"rgba(37,99,235,0.25)");
  gradient.addColorStop(1,"rgba(37,99,235,0.05)");

  ctx.beginPath();
  for(let i=0;i<luxLineData.length;i++){
    const x = padding + (luxLineData[i].time/maxTime)*w;
    const y = padding + h - (luxLineData[i].lux/maxLux)*h;

    if(i===0) ctx.moveTo(x,y);
    else{
      const prev = padding + h - (luxLineData[i-1].lux/maxLux)*h;
      ctx.quadraticCurveTo((padding+(luxLineData[i-1].time/maxTime)*w + x)/2,(prev+y)/2,x,y);
    }
  }
  ctx.lineTo(padding+w,padding+h);
  ctx.lineTo(padding,padding+h);
  ctx.closePath();
  ctx.fillStyle = gradient;
  ctx.fill();

  // LINE
  ctx.beginPath();
  ctx.lineWidth = 2.5;
  for(let i=0;i<luxLineData.length;i++){
    const x = padding + (luxLineData[i].time/maxTime)*w;
    const y = padding + h - (luxLineData[i].lux/maxLux)*h;

    if(i===0) ctx.moveTo(x,y);
    else{
      const prev = padding + h - (luxLineData[i-1].lux/maxLux)*h;
      ctx.quadraticCurveTo((x + padding + (luxLineData[i-1].time/maxTime)*w)/2,(y+prev)/2,x,y);
    }

    // warna garis sesuai ESI
    const status = luxLineData[i].status;
    if(status==="NYAMAN") ctx.strokeStyle="#22c55e";
    else if(status==="MULAI LELAH") ctx.strokeStyle="#3b82f6";
    else if(status==="LELAH") ctx.strokeStyle="#facc15";
    else ctx.strokeStyle="#ef4444";
  }
  ctx.stroke();
}

/* ===============================
   DRAW BAR CHART
=============================== */
function drawBarChart(){
  const ctx = barCtx;
  ctx.clearRect(0,0,barCanvas.width,barCanvas.height);

  const padding = 20;
  const w = barCanvas.width - padding*2;
  const h = barCanvas.height - padding*2;

  statusBarData.forEach((d,i)=>{
    const barHeight = (d.value/MAX_LUX)*h;
    const x = padding + i*(w/statusBarData.length);
    const y = padding + h - barHeight;

    // warna bar chart sesuai ESI
    if(d.status==="NYAMAN") ctx.fillStyle="#22c55e";
    else if(d.status==="MULAI LELAH") ctx.fillStyle="#3b82f6";
    else if(d.status==="LELAH") ctx.fillStyle="#facc15";
    else ctx.fillStyle="#ef4444";

    ctx.fillRect(x, y, (w/statusBarData.length)*0.8, barHeight);
  });
}

/* ===============================
   RESET SIMULASI
=============================== */
function resetSimulation(){
  luxLineData=[];
  statusBarData=[];
  timeCounter=0;
  drawLineChart();
  drawBarChart();

  document.getElementById("luxValue").textContent="0";
  document.getElementById("esiValue").textContent="0.00";
  document.getElementById("esiNumber").textContent="0.00";
  document.getElementById("esiIndicator").textContent="-";
  document.getElementById("distanceInfo").textContent="Estimasi jarak: -";

  document.getElementById("esiIndicator").className="esi-status";
  document.getElementById("esiNumber").className="status aman";

  if(writer) writer.write(new TextEncoder().encode("RESET\n"));
}

/* ===============================
   CONNECT ARDUINO
=============================== */
async function connectSerial(){
  if(!("serial" in navigator)){ alert("Gunakan Chrome/Edge"); return; }
  try{
    port = await navigator.serial.requestPort();
    await port.open({baudRate:9600});

    writer = port.writable.getWriter();
    reader = port.readable.getReader();

    readSerialLoop();
  }catch(err){ console.error("Gagal koneksi:", err);}
}

async function readSerialLoop(){
  let buffer="";
  while(true){
    const {value,done}=await reader.read();
    if(done) break;
    buffer += new TextDecoder().decode(value);
    const lines = buffer.split("\n");
    buffer = lines.pop();
    lines.forEach(line=>{if(line) handleArduinoData(line.trim());});
  }
}

const ws = new WebSocket("ws://localhost:8765");

ws.onmessage = (event) => {
    const d = JSON.parse(event.data);

    document.getElementById("luxValue").innerText = d.lux;
    document.getElementById("esiNumber").innerText = d.esi.toFixed(2);
    document.getElementById("esiIndicator").innerText = d.status;
    document.getElementById("distanceInfo").innerText =
        "Estimasi jarak: " + d.jarak;

    const s = document.getElementById("esiIndicator");
    s.className = "esi-status " + d.status.replace(" ", "-").toLowerCase();
};
