// =========================================================
// CHECKPOINT — Mask Detection Terminal (frontend logic)
// =========================================================

const tabs = document.querySelectorAll(".tab");
const panels = document.querySelectorAll(".panel");
const statusLed = document.getElementById("statusLed");
const statusText = document.getElementById("statusText");

const videoImg = document.getElementById("videoImg");
const feedPlaceholder = document.getElementById("feedPlaceholder");
const startBtn = document.getElementById("startBtn");
const stopBtn = document.getElementById("stopBtn");

const dropzone = document.getElementById("dropzone");
const fileInput = document.getElementById("fileInput");
const scanBtn = document.getElementById("scanBtn");
const uploadForm = document.getElementById("uploadForm");
const resultGrid = document.getElementById("resultGrid");
const previewImg = document.getElementById("previewImg");
const outputImg = document.getElementById("outputImg");
const readout = document.getElementById("readout");
const faceCount = document.getElementById("faceCount");
const detectionList = document.getElementById("detectionList");

// ---------- Tab switching ----------
tabs.forEach((tab) => {
  tab.addEventListener("click", () => {
    tabs.forEach((t) => t.classList.remove("active"));
    panels.forEach((p) => p.classList.remove("active"));

    tab.classList.add("active");
    document.getElementById(`panel-${tab.dataset.tab}`).classList.add("active");

    // agar live tab se hatt rahe hain, stream band kar do
    if (tab.dataset.tab !== "live") {
      stopStream();
    }
  });
});

// ---------- Live webcam stream ----------
function startStream() {
  videoImg.src = "/video_feed?" + new Date().getTime();
  videoImg.style.display = "block";
  feedPlaceholder.style.display = "none";
  stopBtn.style.display = "block";
  setStatus("on-ok", "SCANNING LIVE");
}

function stopStream() {
  videoImg.src = "";
  videoImg.style.display = "none";
  feedPlaceholder.style.display = "flex";
  stopBtn.style.display = "none";
  setStatus("", "STANDBY");
}

startBtn.addEventListener("click", startStream);
stopBtn.addEventListener("click", stopStream);

function setStatus(ledClass, text) {
  statusLed.className = "led " + ledClass;
  statusText.textContent = text;
}

// ---------- Upload flow ----------
dropzone.addEventListener("click", () => fileInput.click());

dropzone.addEventListener("dragover", (e) => {
  e.preventDefault();
  dropzone.classList.add("drag-over");
});
dropzone.addEventListener("dragleave", () => dropzone.classList.remove("drag-over"));
dropzone.addEventListener("drop", (e) => {
  e.preventDefault();
  dropzone.classList.remove("drag-over");
  if (e.dataTransfer.files.length) {
    fileInput.files = e.dataTransfer.files;
    handleFileSelect();
  }
});

fileInput.addEventListener("change", handleFileSelect);

function handleFileSelect() {
  const file = fileInput.files[0];
  if (!file) return;

  scanBtn.disabled = false;
  dropzone.querySelector(".dz-text").textContent = file.name;

  const reader = new FileReader();
  reader.onload = (e) => {
    previewImg.src = e.target.result;
  };
  reader.readAsDataURL(file);
}

uploadForm.addEventListener("submit", async (e) => {
  e.preventDefault();
  const file = fileInput.files[0];
  if (!file) return;

  scanBtn.disabled = true;
  scanBtn.textContent = "SCANNING...";
  setStatus("on-amber", "PROCESSING");

  const formData = new FormData();
  formData.append("image", file);

  try {
    const res = await fetch("/upload", { method: "POST", body: formData });
    const data = await res.json();

    if (data.error) {
      alert(data.error);
      setStatus("on-alert", "ERROR");
      return;
    }

    resultGrid.style.display = "grid";
    readout.style.display = "block";
    outputImg.src = data.image;
    faceCount.textContent = data.faces_found;

    detectionList.innerHTML = "";
    data.detections.forEach((d) => {
      const row = document.createElement("div");
      row.className = "detection-item";
      const badgeClass = d.label === "Mask" ? "mask" : "no-mask";
      row.innerHTML = `
        <span>FACE — ${d.confidence}% confidence</span>
        <span class="badge ${badgeClass}">${d.label.toUpperCase()}</span>
      `;
      detectionList.appendChild(row);
    });

    const anyNoMask = data.detections.some((d) => d.label === "No Mask");
    if (data.faces_found === 0) {
      setStatus("", "NO FACE FOUND");
    } else if (anyNoMask) {
      setStatus("on-alert", "ALERT — NO MASK");
    } else {
      setStatus("on-ok", "CLEAR — MASK OK");
    }
  } catch (err) {
    alert("Kuch galat ho gaya. Server console check karo.");
    setStatus("on-alert", "ERROR");
  } finally {
    scanBtn.disabled = false;
    scanBtn.textContent = "SCAN IMAGE";
  }
});
