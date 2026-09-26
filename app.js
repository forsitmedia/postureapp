import { PoseLandmarker, FilesetResolver }
  from "https://cdn.jsdelivr.net/npm/@mediapipe/tasks-vision@0.10.14";

/* ---------------------------------------------------------
   Unshrimp Tracker - live slouch scoring from a webcam.
   Scoring is personalised: we capture the user's own best
   and worst posture, then project live pose onto that axis.
--------------------------------------------------------- */

const $ = id => document.getElementById(id);

const video   = $("video");
const overlay = $("overlay");
const ctx     = overlay.getContext("2d");

// landmark indices (MediaPipe Pose)
const NOSE = 0, L_SH = 11, R_SH = 12, L_EAR = 7, R_EAR = 8, L_EL = 13, R_EL = 14;

const SKELETON = [[L_SH,R_SH],[L_SH,L_EL],[R_SH,R_EL],[NOSE,L_SH],[NOSE,R_SH],[L_EAR,NOSE],[R_EAR,NOSE]];

const JEERS = [
  "I blinked and you became seafood. Fix it.",
  "That's a prawn posture. Straighten up.",
  "Your spine just filed a complaint.",
  "Shrimp detected. Deploying shame.",
  "You're folding like a deck chair.",
  "Shoulders back. I can see you curling."
];

let landmarker = null;
let running = false;
let rafId = null;
let lastVideoTime = -1;

// calibration
let calGood = null;     // {neck,size,tilt,off}
let calBad  = null;
let phase = "idle";     // idle | frame | good | bad | live
let holdMs = 0;
let lastT = performance.now();

// live state
let score = 100;
let scoreEMA = null;
let smoothLm = null;

// alerting
let badMs = 0;
let alerting = false;
let cooldownMs = 0;

// stats
let sessionMs = 0, uprightMs = 0, alerts = 0, streakMs = 0, bestStreak = 0;

const HOLD_TARGET = 1800;   // ms to hold a calibration pose
const COOLDOWN    = 12000;  // ms before the video can fire again

/* ------------------------- helpers ------------------------- */
const clamp = (v,a,b) => Math.min(b, Math.max(a,v));
const dist  = (a,b) => Math.hypot(a.x-b.x, a.y-b.y);

function fmt(ms){
  const s = Math.floor(ms/1000);
  return `${Math.floor(s/60)}:${String(s%60).padStart(2,"0")}`;
}

function setStep(name, state){
  document.querySelectorAll(".steps li").forEach(li=>{
    if(li.dataset.step === name){
      li.classList.toggle("active", state === "active");
      li.classList.toggle("done",   state === "done");
    }
  });
}
function markDone(name){ setStep(name,"done"); }

/* --------------------- feature extraction ------------------ */
function features(lm){
  const ls = lm[L_SH], rs = lm[R_SH], nose = lm[NOSE];
  const size = dist(ls, rs);                       // shoulder width ~ proximity
  if(size < 1e-4) return null;
  const midX = (ls.x + rs.x)/2, midY = (ls.y + rs.y)/2;
  return {
    size,
    neck: (midY - nose.y) / size,                  // head height above shoulders
    tilt: (ls.y - rs.y) / size,                    // one-sided collapse
    off : (nose.x - midX) / size                   // head lateral drift
  };
}

/* ------------------------- scoring ------------------------- */
function computeScore(f){
  if(!calGood) return 100;

  // fall back to synthetic "bad" if the user skipped that step
  const bad = calBad || {
    neck: calGood.neck * 0.62,
    size: calGood.size * 1.22,
    tilt: calGood.tilt,
    off : calGood.off
  };

  // relative good->bad deltas (dimensionless), used as per-axis weights
  const relN = (bad.neck - calGood.neck) / (calGood.neck || 1e-6);
  const relS = (bad.size - calGood.size) / (calGood.size || 1e-6);

  const wN = Math.abs(relN), wS = Math.abs(relS);
  let t = 0;

  if(wN + wS > 1e-6){
    const uN = wN > 1e-6 ? ((f.neck - calGood.neck)/(calGood.neck||1e-6)) / relN : 0;
    const uS = wS > 1e-6 ? ((f.size - calGood.size)/(calGood.size||1e-6)) / relS : 0;
    t = (wN*clamp(uN,-0.5,1.6) + wS*clamp(uS,-0.5,1.6)) / (wN + wS);
  }
  t = clamp(t, 0, 1);

  let s = 100 * (1 - t);

  // independent penalties: leaning sideways / head drifting off-centre
  const tiltPen = clamp((Math.abs(f.tilt - calGood.tilt) - 0.04) / 0.22, 0, 1) * 25;
  const offPen  = clamp((Math.abs(f.off  - calGood.off ) - 0.06) / 0.30, 0, 1) * 15;

  return clamp(s - tiltPen - offPen, 0, 100);
}

/* -------------------------- drawing ------------------------ */
function resize(){
  const r = overlay.getBoundingClientRect();
  const dpr = Math.min(window.devicePixelRatio || 1, 2);
  overlay.width  = Math.round(r.width  * dpr);
  overlay.height = Math.round(r.height * dpr);
}
window.addEventListener("resize", resize);

// video uses object-fit:cover, so map normalised coords through the same crop
function mapper(){
  const bw = overlay.width, bh = overlay.height;
  const vw = video.videoWidth || 4, vh = video.videoHeight || 3;
  const scale = Math.max(bw/vw, bh/vh);
  const dw = vw*scale, dh = vh*scale;
  const ox = (bw-dw)/2, oy = (bh-dh)/2;
  return (p) => ({ x: ox + p.x*dw, y: oy + p.y*dh });
}

function draw(lm, col){
  const bw = overlay.width, bh = overlay.height;
  ctx.clearRect(0,0,bw,bh);
  if(!lm) return;

  ctx.save();
  ctx.translate(bw,0); ctx.scale(-1,1);   // match the mirrored <video>
  const M = mapper();
  const dpr = bw / overlay.getBoundingClientRect().width;

  ctx.lineWidth = 3*dpr;
  ctx.strokeStyle = col;
  ctx.lineCap = "round";
  for(const [a,b] of SKELETON){
    if((lm[a].visibility ?? 1) < .35 || (lm[b].visibility ?? 1) < .35) continue;
    const p = M(lm[a]), q = M(lm[b]);
    ctx.beginPath(); ctx.moveTo(p.x,p.y); ctx.lineTo(q.x,q.y); ctx.stroke();
  }

  ctx.fillStyle = col;
  for(const i of [NOSE,L_SH,R_SH,L_EL,R_EL]){
    if((lm[i].visibility ?? 1) < .35) continue;
    const p = M(lm[i]);
    ctx.beginPath(); ctx.ellipse(p.x,p.y,7*dpr,5*dpr,0,0,Math.PI*2); ctx.fill();
    ctx.strokeStyle = "rgba(0,0,0,.55)"; ctx.lineWidth = 1.5*dpr; ctx.stroke();
    ctx.strokeStyle = col; ctx.lineWidth = 3*dpr;
  }
  ctx.restore();
}

/* ------------------------ framing UI ----------------------- */
function framing(lm){
  const vis = i => (lm?.[i]?.visibility ?? 0) > .5;
  const noseOk = vis(NOSE);
  const shOk   = vis(L_SH) && vis(R_SH);
  let distOk = false;
  if(shOk){ const s = dist(lm[L_SH], lm[R_SH]); distOk = s > .14 && s < .55; }

  $("chkNose").classList.toggle("ok", noseOk);
  $("chkShoulders").classList.toggle("ok", shOk);
  $("chkDist").classList.toggle("ok", distOk);
  return noseOk && shOk && distOk;
}

/* --------------------------- toast ------------------------- */
let toastUntil = 0;
function toast(msg, ms=2600){
  const t = $("toast");
  t.textContent = msg; t.hidden = false;
  toastUntil = performance.now() + ms;
}
function tickToast(now){ if(!$("toast").hidden && now > toastUntil) $("toast").hidden = true; }

/* ------------------------ calibration ---------------------- */
function enterPhase(p){
  phase = p; holdMs = 0;
  const panel = $("calibPanel");

  if(p === "frame"){
    panel.hidden = true;
    setStep("frame","active");
    toast("Get your nose and both shoulders in frame.", 3000);
  }
  if(p === "good"){
    panel.hidden = false;
    $("calibTitle").textContent = "Capture: best posture";
    $("calibHint").textContent  = "Sit up tall, shoulders back, chin level. Hold it.";
    markDone("frame"); setStep("good","active");
  }
  if(p === "bad"){
    panel.hidden = false;
    $("calibTitle").textContent = "Capture: worst shrimp";
    $("calibHint").textContent  = "Now collapse. Hunch forward into the screen. Hold it.";
    markDone("good"); setStep("bad","active");
  }
  if(p === "live"){
    panel.hidden = true;
    markDone("good"); markDone("bad"); setStep("live","active");
    $("scorebadge").hidden = false;
    toast("Live. Now try to slouch.", 2600);
  }
  $("holdfill").style.width = "0%";
}

function calibrate(f, dt, ok){
  if(!ok || !f){ holdMs = Math.max(0, holdMs - dt*1.5); }
  else { holdMs += dt; }
  $("holdfill").style.width = `${clamp(holdMs/HOLD_TARGET,0,1)*100}%`;

  if(holdMs >= HOLD_TARGET && f){
    if(phase === "good"){ calGood = {...f}; enterPhase("bad"); }
    else if(phase === "bad"){
      // only accept if it is meaningfully different from the good pose
      const moved = Math.abs(f.neck - calGood.neck)/calGood.neck > .08 ||
                    Math.abs(f.size - calGood.size)/calGood.size > .06;
      if(moved){ calBad = {...f}; enterPhase("live"); }
      else { holdMs = 0; toast("That looks the same as your good pose. Really slouch.", 2600); }
    }
  }
}

/* --------------------------- alerts ------------------------ */
function fireWarning(){
  alerting = true; alerts++;
  $("stAlerts").textContent = alerts;
  $("warntext").textContent = JEERS[Math.floor(Math.random()*JEERS.length)];
  $("warnwrap").hidden = false;
  const v = $("warnvid");
  if($("soundOn").checked){ v.currentTime = 0; v.muted = false; v.play().catch(()=>{ v.muted = true; v.play().catch(()=>{}); }); }
}
function closeWarning(){
  $("warnwrap").hidden = true;
  const v = $("warnvid"); v.pause();
  alerting = false; badMs = 0; cooldownMs = COOLDOWN;
}
$("warnClose").addEventListener("click", closeWarning);
$("warnvid").addEventListener("ended", closeWarning);

/* --------------------------- loop -------------------------- */
function loop(){
  rafId = requestAnimationFrame(loop);
  if(!landmarker || video.readyState < 2) return;

  const now = performance.now();
  const dt  = Math.min(now - lastT, 100);
  lastT = now;
  tickToast(now);

  if(video.currentTime === lastVideoTime) return;
  lastVideoTime = video.currentTime;

  const res = landmarker.detectForVideo(video, now);
  const lm  = res?.landmarks?.[0] || null;

  // smooth landmarks a little so the skeleton does not jitter
  if(lm){
    if(!smoothLm || smoothLm.length !== lm.length) smoothLm = lm.map(p=>({...p}));
    else lm.forEach((p,i)=>{
      smoothLm[i].x += (p.x - smoothLm[i].x)*0.45;
      smoothLm[i].y += (p.y - smoothLm[i].y)*0.45;
      smoothLm[i].visibility = p.visibility;
    });
  } else smoothLm = null;

  const ok = lm ? framing(smoothLm) : (framing([]), false);
  const f  = smoothLm ? features(smoothLm) : null;

  if(phase === "frame" && ok) enterPhase("good");
  else if(phase === "good" || phase === "bad") calibrate(f, dt, ok);

  let col = "#ff7a2f";

  if(phase === "live" && f && ok){
    const raw = computeScore(f);
    scoreEMA = scoreEMA === null ? raw : scoreEMA + (raw - scoreEMA)*0.16;
    score = scoreEMA;

    const th = +$("thresh").value;
    const graceMs = +$("grace").value * 1000;
    const good = score >= th;

    col = good ? "#3ddc97" : (score >= th-18 ? "#ffb020" : "#ff4d4d");

    // stats
    sessionMs += dt;
    if(good){ uprightMs += dt; streakMs += dt; bestStreak = Math.max(bestStreak, streakMs); }
    else { streakMs = 0; }

    // alerting
    if(cooldownMs > 0) cooldownMs -= dt;
    if(!good && !alerting && cooldownMs <= 0){
      badMs += dt;
      if(badMs >= graceMs) fireWarning();
    } else if(good){ badMs = 0; }

    // readouts
    $("scorenum").textContent = Math.round(score);
    $("scorenum").style.color = col;
    $("meterfill").style.width = `${score}%`;
    $("meterfill").style.background = col;
    $("stateLabel").textContent = good ? "upright" : "shrimping";
    $("stUpright").textContent = sessionMs > 0 ? `${Math.round(uprightMs/sessionMs*100)}%` : "0%";
    $("stSession").textContent = fmt(sessionMs);
    $("stStreak").textContent  = fmt(bestStreak);

    if(!good && $("toast").hidden && Math.random() < 0.012) toast(JEERS[Math.floor(Math.random()*JEERS.length)]);
  } else if(phase === "live"){
    $("stateLabel").textContent = "lost you";
    col = "#6f6762";
  }

  draw(smoothLm, col);
}

/* --------------------------- start ------------------------- */
async function start(){
  if(running) return;
  $("startBtn").textContent = "Starting...";
  $("startBtn").disabled = true;

  try{
    const stream = await navigator.mediaDevices.getUserMedia({
      video:{ width:{ideal:1280}, height:{ideal:720}, facingMode:"user" }, audio:false
    });
    video.srcObject = stream;
    await video.play();
  }catch(e){
    $("startBtn").textContent = "Start camera";
    $("startBtn").disabled = false;
    alert("Camera blocked: " + e.message + "\n\nAllow camera access and try again.");
    return;
  }

  try{
    const vision = await FilesetResolver.forVisionTasks(
      "https://cdn.jsdelivr.net/npm/@mediapipe/tasks-vision@0.10.14/wasm");
    landmarker = await PoseLandmarker.createFromOptions(vision, {
      baseOptions:{
        modelAssetPath:"https://storage.googleapis.com/mediapipe-models/pose_landmarker/pose_landmarker_lite/float16/1/pose_landmarker_lite.task",
        delegate:"GPU"
      },
      runningMode:"VIDEO",
      numPoses:1
    });
  }catch(e){
    alert("Could not load the pose model. Check your connection.\n\n" + e.message);
    return;
  }

  running = true;
  $("idle").hidden = true;
  resize();
  markDone("camera");
  enterPhase("frame");
  lastT = performance.now();
  loop();
}

/* ------------------------- controls ------------------------ */
$("startBtn").addEventListener("click", start);

$("skipBtn").addEventListener("click", ()=>{
  if(phase === "good"){ toast("Need a good pose first."); return; }
  if(phase === "bad"){ calBad = null; enterPhase("live"); }
});

$("recalBtn").addEventListener("click", ()=>{
  if(!running){ toast("Start the camera first."); return; }
  calGood = calBad = null; scoreEMA = null;
  ["frame","good","bad","live"].forEach(s=>setStep(s,""));
  $("scorebadge").hidden = true;
  enterPhase("frame");
});

$("resetBtn").addEventListener("click", ()=>{
  sessionMs = uprightMs = alerts = streakMs = bestStreak = badMs = 0;
  cooldownMs = 0; scoreEMA = null;
  $("stUpright").textContent="0%"; $("stSession").textContent="0:00";
  $("stAlerts").textContent="0";   $("stStreak").textContent="0:00";
  toast("Stats cleared.");
});

$("thresh").addEventListener("input", e => $("thVal").textContent = e.target.value);
$("grace").addEventListener("input",  e => $("graceVal").textContent = e.target.value + "s");

setStep("camera","active");
resize();
