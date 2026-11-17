// Scanner3D — cliente móvil (PWA, sin dependencias de build)
const $ = (s) => document.querySelector(s);
const $$ = (s) => [...document.querySelectorAll(s)];

// ---------------------------------------------------------------- token / API
const params = new URLSearchParams(location.search);
if (params.has("token")) {
  try { localStorage.setItem("token", params.get("token")); } catch {}
  history.replaceState(null, "", location.pathname);
}
let TOKEN = "";
try { TOKEN = localStorage.getItem("token") || ""; } catch {}

async function api(path, opts = {}) {
  const r = await fetch(path, { ...opts, headers: { ...(opts.headers || {}), "X-Token": TOKEN } });
  if (!r.ok) {
    let msg = r.statusText;
    try { msg = (await r.json()).detail || msg; } catch {}
    throw new Error(typeof msg === "string" ? msg : JSON.stringify(msg));
  }
  return r.status === 204 ? null : r.json();
}
const fileUrl = (id, name) => `/api/jobs/${id}/files/${name}` + (TOKEN ? `?token=${encodeURIComponent(TOKEN)}` : "");

function toast(msg, ms = 2600) {
  const t = $("#toast");
  t.textContent = msg;
  t.classList.add("on");
  clearTimeout(t._h);
  t._h = setTimeout(() => t.classList.remove("on"), ms);
}

// ------------------------------------------------------------------- pestañas
function showTab(name) {
  $$(".tabs button").forEach((b) => b.classList.toggle("on", b.dataset.tab === name));
  $$(".tab").forEach((t) => t.classList.toggle("active", t.id === `tab-${name}`));
  if (name === "jobs") refreshJobs();
}
$$(".tabs button").forEach((b) => b.addEventListener("click", () => showTab(b.dataset.tab)));

// --------------------------------------------------------- fotos del escaneo
const shots = []; // {blob, url, heading}
function addShot(blob, heading = null) {
  shots.push({ blob, url: URL.createObjectURL(blob), heading });
  renderTray();
}
function renderTray() {
  $("#count").textContent = shots.length;
  $("#btn-send").disabled = shots.length < 3;
  const box = $("#thumbs");
  box.innerHTML = "";
  shots.slice(-30).reverse().forEach((s) => {
    const im = new Image();
    im.src = s.url;
    im.alt = "";
    box.appendChild(im);
  });
}
$("#btn-clear").addEventListener("click", () => {
  if (shots.length && !confirm("¿Vaciar las fotos del escaneo actual?")) return;
  shots.splice(0).forEach((s) => URL.revokeObjectURL(s.url));
  covered.fill(false);
  drawRing();
  renderTray();
});
$("#file").addEventListener("change", (e) => {
  [...e.target.files].forEach((f) => addShot(f));
  toast(`${e.target.files.length} fotos agregadas`);
  e.target.value = "";
});

// ---------------------------------------------------------------- cámara
const video = $("#video");
let stream = null;
let mode = "manual";
let autoOn = false;
let timerId = null;
let lastShot = 0;

async function startCamera() {
  if (!navigator.mediaDevices?.getUserMedia) {
    $("#cam-msg").textContent = window.isSecureContext
      ? "Este navegador no permite usar la cámara."
      : "La cámara en vivo necesita HTTPS (arranca el servidor con --https). Puedes usar la Galería.";
    return;
  }
  try {
    stream = await navigator.mediaDevices.getUserMedia({
      video: { facingMode: { ideal: "environment" }, width: { ideal: 1920 }, height: { ideal: 1440 } },
      audio: false,
    });
    video.srcObject = stream;
    await video.play();
    $("#cam-msg").textContent = "";
    $("#btn-shot").disabled = false;
    $("#btn-auto").disabled = mode === "manual";
    $("#btn-cam").textContent = "Cámara activa";
    $("#btn-cam").disabled = true;
    startCompass();
  } catch (e) {
    $("#cam-msg").textContent = "No se pudo abrir la cámara: " + e.message;
  }
}
$("#btn-cam").addEventListener("click", startCamera);

function capture() {
  if (!stream || !video.videoWidth) return;
  const c = document.createElement("canvas");
  c.width = video.videoWidth;
  c.height = video.videoHeight;
  c.getContext("2d").drawImage(video, 0, 0);
  const h = heading;
  c.toBlob((b) => b && addShot(b, h), "image/jpeg", 0.92);
  if (h !== null) covered[Math.floor(h / 10) % 36] = true;
  drawRing();
  lastShot = performance.now();
  const f = $("#flash");
  f.classList.add("on");
  requestAnimationFrame(() => requestAnimationFrame(() => f.classList.remove("on")));
  navigator.vibrate?.(30);
  if (covered.every(Boolean) && mode === "compass" && autoOn) {
    toast("¡Vuelta completa! Ya puedes enviar.");
    toggleAuto(false);
  }
}
$("#btn-shot").addEventListener("click", capture);

$$(".seg button").forEach((b) => b.addEventListener("click", async () => {
  toggleAuto(false);
  mode = b.dataset.mode;
  $$(".seg button").forEach((x) => x.classList.toggle("on", x === b));
  $("#timer-opts").classList.toggle("hidden", mode !== "timer");
  $("#btn-auto").disabled = !stream || mode === "manual";
  $("#mode-hint").textContent = {
    manual: "Rodea el objeto y toma una foto cada ~10°. Mantén el objeto centrado.",
    compass: "Camina alrededor del objeto: dispara solo al entrar en cada sector de 10° sin cubrir.",
    timer: "Celular fijo y objeto en un plato giratorio: dispara cada N segundos.",
  }[mode];
  if (mode === "compass") await requestCompassPermission();
}));

function toggleAuto(on = !autoOn) {
  autoOn = on;
  $("#btn-auto").textContent = on ? "Detener" : "Iniciar";
  clearInterval(timerId);
  timerId = null;
  if (on && mode === "timer") timerId = setInterval(capture, +$("#interval").value);
  if (on && mode === "compass" && heading === null) toast("Sin brújula: usa el modo manual o temporizador");
}
$("#btn-auto").addEventListener("click", () => toggleAuto());

// ---------------------------------------------------------------- brújula
let heading = null;
const covered = new Array(36).fill(false);
const recent = [];

async function requestCompassPermission() {
  const DOE = window.DeviceOrientationEvent;
  if (DOE && typeof DOE.requestPermission === "function") {
    try { await DOE.requestPermission(); } catch {}
  }
  startCompass();
}
let compassStarted = false;
function startCompass() {
  if (compassStarted) return;
  compassStarted = true;
  const onOri = (e) => {
    let h = null;
    if (typeof e.webkitCompassHeading === "number") h = e.webkitCompassHeading;      // iOS
    else if (e.absolute && typeof e.alpha === "number") h = (360 - e.alpha) % 360;   // Android
    if (h === null) return;
    heading = h;
    const now = performance.now();
    recent.push([now, h]);
    while (recent.length && now - recent[0][0] > 400) recent.shift();
    $("#heading").textContent = `${Math.round(h)}° · ${covered.filter(Boolean).length}/36`;
    drawRing();
    if (autoOn && mode === "compass") maybeAutoShoot(now);
  };
  window.addEventListener("deviceorientationabsolute", onOri);
  window.addEventListener("deviceorientation", onOri);
}
function angDiff(a, b) { return Math.abs(((a - b + 540) % 360) - 180); }
function maybeAutoShoot(now) {
  const bin = Math.floor(heading / 10) % 36;
  if (covered[bin] || now - lastShot < 700) return;
  const steady = recent.length > 2 && recent.every(([, h]) => angDiff(h, heading) < 3);
  if (steady) capture();
}

function drawRing() {
  const svg = $("#ring");
  let s = "";
  for (let i = 0; i < 36; i++) {
    const a0 = ((i * 10 - 90) * Math.PI) / 180, a1 = (((i + 1) * 10 - 91) * Math.PI) / 180;
    const p = (r, a) => `${(r * Math.cos(a)).toFixed(2)} ${(r * Math.sin(a)).toFixed(2)}`;
    s += `<path d="M${p(46, a0)} A46 46 0 0 1 ${p(46, a1)} L${p(34, a1)} A34 34 0 0 0 ${p(34, a0)}Z" fill="${covered[i] ? "#4fc3a1" : "rgba(255,255,255,.18)"}"/>`;
  }
  if (heading !== null) {
    const a = ((heading - 90) * Math.PI) / 180;
    s += `<line x1="0" y1="0" x2="${(30 * Math.cos(a)).toFixed(2)}" y2="${(30 * Math.sin(a)).toFixed(2)}" stroke="#fff" stroke-width="3" stroke-linecap="round"/>`;
  }
  svg.innerHTML = s + `<circle r="4" fill="#fff"/>`;
}
drawRing();

// ---------------------------------------------------------------- enviar
$("#btn-send").addEventListener("click", async () => {
  const btn = $("#btn-send");
  btn.disabled = true;
  const bar = $("#upload-bar");
  bar.classList.remove("hidden");
  try {
    const job = await api("/api/jobs", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ name: $("#opt-name").value, quality: $("#opt-quality").value, engine: $("#opt-engine").value }),
    });
    const B = 4;
    let rejected = 0;
    for (let i = 0; i < shots.length; i += B) {
      const fd = new FormData();
      shots.slice(i, i + B).forEach((s, k) => fd.append("files", s.blob, s.blob.name || `foto_${i + k}.jpg`));
      const r = await api(`/api/jobs/${job.id}/images`, { method: "POST", body: fd });
      rejected += r.rejected.length;
      bar.firstElementChild.style.width = `${Math.min(100, ((i + B) / shots.length) * 100)}%`;
      btn.textContent = `Subiendo ${Math.min(i + B, shots.length)}/${shots.length}…`;
    }
    if (rejected) toast(`${rejected} archivos rechazados (formato no válido)`);
    await api(`/api/jobs/${job.id}/start`, { method: "POST" });
    shots.splice(0).forEach((s) => URL.revokeObjectURL(s.url));
    covered.fill(false);
    drawRing();
    renderTray();
    openJob(job.id);
  } catch (e) {
    toast("Error: " + e.message, 4000);
  } finally {
    btn.textContent = "Enviar y reconstruir";
    btn.disabled = shots.length < 3;
    bar.classList.add("hidden");
    bar.firstElementChild.style.width = "0";
  }
});

// ---------------------------------------------------------------- trabajos
const STATUS = { created: "Sin enviar", queued: "En cola", running: "Procesando", done: "Listo", failed: "Falló", cancelled: "Cancelado" };
const esc = (s) => String(s).replace(/[&<>"']/g, (c) => ({ "&": "&amp;", "<": "&lt;", ">": "&gt;", '"': "&quot;", "'": "&#39;" }[c]));

async function refreshJobs() {
  const box = $("#jobs");
  try {
    const jobs = await api("/api/jobs");
    if (!jobs.length) { box.innerHTML = `<div class="empty">Aún no hay modelos.</div>`; return; }
    box.innerHTML = jobs.map((j) => `
      <div class="job" data-id="${j.id}">
        <div class="t"><strong>${esc(j.name)}</strong><span class="chip ${j.status}">${STATUS[j.status] || j.status}</span></div>
        <div class="m">${j.images} fotos · ${new Date(j.created * 1000).toLocaleString()}${j.status === "running" ? ` · ${Math.round(j.progress * 100)}%` : ""}</div>
        ${j.status === "running" ? `<div class="bar"><i style="width:${j.progress * 100}%"></i></div>` : ""}
      </div>`).join("");
    box.querySelectorAll(".job").forEach((el) => el.addEventListener("click", () => openJob(el.dataset.id)));
  } catch (e) {
    box.innerHTML = `<div class="empty">${esc(e.message)}</div>`;
  }
}

let current = null;
let pollId = null;
let loadedModel = null;

async function openJob(id) {
  current = id;
  loadedModel = null;
  $("#detail").classList.remove("hidden");
  $("#viewer-msg").textContent = "";
  const { clearViewer } = await import("./viewer.js");
  clearViewer($("#viewer"));
  await pollJob();
  clearInterval(pollId);
  pollId = setInterval(pollJob, 1500);
}
function closeJob() {
  clearInterval(pollId);
  current = null;
  $("#detail").classList.add("hidden");
  refreshJobs();
}
$("#btn-back").addEventListener("click", closeJob);

async function pollJob() {
  if (!current) return;
  let j;
  try { j = await api(`/api/jobs/${current}`); } catch (e) { toast(e.message); return; }
  $("#d-name").textContent = j.name;
  $("#d-bar").style.width = `${(j.status === "done" ? 1 : j.progress) * 100}%`;
  $("#d-msg").textContent = `${STATUS[j.status] || j.status} — ${j.message || ""}`;
  const active = j.status === "running" || j.status === "queued";
  $("#btn-cancel").classList.toggle("hidden", !active);
  $("#btn-retry").classList.toggle("hidden", !(j.status === "failed" || j.status === "cancelled"));
  $("#btn-delete").classList.toggle("hidden", active);
  if (!active) clearInterval(pollId);
  if (!active && j.status !== "done") $("#viewer-msg").textContent = "Sin modelo";
  if (active) $("#viewer-msg").textContent = "Reconstruyendo…";

  const r = j.result;
  $("#d-stats").innerHTML = r ? [
    ["Cámaras ubicadas", `${r.registered}/${r.images}`], ["Motor", r.engine],
    ["Puntos densos", r.dense_points.toLocaleString()], ["Triángulos", r.triangles.toLocaleString()],
    ["Tiempo", `${r.seconds} s`],
  ].map(([k, v]) => `<div>${k}<b>${esc(v)}</b></div>`).join("") : "";
  const labels = { "model.glb": "GLB", "model.obj": "OBJ", "model.ply": "PLY (malla)", "points.ply": "PLY (nube)" };
  $("#d-files").innerHTML = (j.files || []).filter((f) => labels[f])
    .map((f) => `<a class="btn" href="${fileUrl(j.id, f)}" download>⬇ ${labels[f]}</a>`).join("");

  if (j.status === "done" && loadedModel !== j.id) {
    loadedModel = j.id;
    $("#viewer-msg").textContent = "Cargando visor…";
    try {
      const { showModel } = await import("./viewer.js");
      await showModel($("#viewer"), fileUrl(j.id, "model.glb"));
      $("#viewer-msg").textContent = "";
    } catch (e) {
      $("#viewer-msg").textContent = "No se pudo cargar el visor 3D: " + e.message;
    }
  }
}
$("#btn-cancel").addEventListener("click", async () => { await api(`/api/jobs/${current}/cancel`, { method: "POST" }); pollJob(); });
$("#btn-retry").addEventListener("click", async () => {
  try { await api(`/api/jobs/${current}/start`, { method: "POST" }); } catch (e) { return toast(e.message); }
  clearInterval(pollId);
  pollId = setInterval(pollJob, 1500);
  pollJob();
});
$("#btn-delete").addEventListener("click", async () => {
  if (!confirm("¿Borrar este modelo y sus fotos?")) return;
  await api(`/api/jobs/${current}`, { method: "DELETE" });
  closeJob();
});

// ---------------------------------------------------------------- arranque
(async () => {
  const c = $("#conn");
  try {
    const info = await api("/api/info");
    c.textContent = "Conectado";
    c.className = "chip ok";
    if (!info.engines.includes("colmap")) $('#opt-engine option[value="colmap"]').disabled = true;
  } catch (e) {
    c.textContent = e.message.includes("Token") ? "Token inválido" : "Sin conexión";
    c.className = "chip bad";
    if (e.message.includes("Token")) toast("Abre el enlace (o QR) que muestra el servidor, incluye el token.", 6000);
  }
  if ("serviceWorker" in navigator && window.isSecureContext) navigator.serviceWorker.register("sw.js").catch(() => {});
})();
