import { PoseLandmarker, FilesetResolver }
  from "https://cdn.jsdelivr.net/npm/@mediapipe/tasks-vision@0.10.14";

/* =========================================================
   Unshrimp Tracker
   A guided calibration wizard followed by live posture
   scoring. We capture the user's own normal pose, four
   distinct bad poses, and their ideal pose, then score by
   distance to ideal in a normalised feature space.
   ========================================================= */

const $ = id => document.getElementById(id);
const clamp = (v,a,b) => Math.min(b, Math.max(a,v));
const dist  = (a,b) => Math.hypot(a.x-b.x, a.y-b.y);

const video = $("video"), overlay = $("overlay"), ctx = overlay.getContext("2d");

const NOSE=0, L_EAR=7, R_EAR=8, L_SH=11, R_SH=12, L_EL=13, R_EL=14, L_WR=15, R_WR=16;
const SKELETON=[[L_SH,R_SH],[L_SH,L_EL],[R_SH,R_EL],[L_EL,L_WR],[R_EL,R_WR],
                [NOSE,L_SH],[NOSE,R_SH],[L_EAR,NOSE],[R_EAR,NOSE]];

/* feature dimensions used for scoring */
const DIMS = ["neck","size","tilt","off"];

const BAD_LABELS = {
  close:"Leaning into the screen",
  down :"Slumped down",
  left :"Collapsed to the left",
  right:"Collapsed to the right"
};

const JEERS = [
  "I blinked and you became seafood. Fix it.",
  "That's a prawn posture. Straighten up.",
  "Your spine just filed a complaint.",
  "Shrimp detected. Deploying shame.",
  "You're folding like a deck chair."
];

/* ========================= steps ========================= */
const STEPS = [
  { id:"intro", kind:"intro",
    title:"Unshrimp Tracker",
    body:"First we learn how you normally sit. Then we record your bad habits so we can spot them. Then we teach you the good one. Takes about a minute.",
    btn:"Start camera" },

  { id:"frame", kind:"frame",
    title:"Let's find you",
    body:"Sit back so your <b>head and both shoulders</b> are in the picture. We'll move on by ourselves." },

  { id:"normal", kind:"capture", ref:"normal",
    title:"Sit how you normally sit",
    body:"Don't correct anything. Slouch if you slouch. This is the baseline we compare against.",
    cue:"Be honest. Hold still." },

  { id:"close", kind:"capture", ref:"close", diff:true,
    title:"Now lean into the screen",
    body:"Push your head and chest toward the monitor, like you're reading something tiny.",
    cue:"Closer. Exaggerate it." },

  { id:"down", kind:"capture", ref:"down", diff:true,
    title:"Now slump back and down",
    body:"Let your chin fall toward your chest and sink into the chair.",
    cue:"Collapse. Let gravity win." },

  { id:"left", kind:"capture", ref:"left", diff:true,
    title:"Collapse onto your left",
    body:"Drop your left shoulder and lean your weight onto that side.",
    cue:"Left shoulder down." },

  { id:"right", kind:"capture", ref:"right", diff:true,
    title:"Now the other side",
    body:"Same thing, mirrored. Drop your right shoulder and lean right.",
    cue:"Right shoulder down." },

  { id:"t1", kind:"tutorial", check:"elbowsIn",
    title:"Elbows to your sides",
    body:"Bring both elbows in until they're <b>touching your sides</b>. Thumbs pointing outward.",
    cue:"Elbows in, thumbs out." },

  { id:"t2", kind:"tutorial", check:"shouldersLevel",
    title:"Rotate your thumbs out",
    body:"Keep your elbows pinned to your sides and rotate your thumbs outward <b>as far as they'll go</b>. Your shoulders will pull back and even out on their own.",
    cue:"As far as you can. Elbows stay put." },

  { id:"t3", kind:"tutorial", check:"armsDown",
    title:"Now drop your arms",
    body:"Let your arms fall loose, but <b>keep your chest and shoulders exactly where they are</b>.",
    cue:"Arms down. Chest stays open." },

  { id:"ideal", kind:"capture", ref:"ideal",
    title:"That's your position. Hold it.",
    body:"This is what we'll hold you to for the rest of the session.",
    cue:"Shoulders level, chest open, head stacked." },

  { id:"done", kind:"done",
    title:"Calibrated",
    body:"Go back to typing. We'll tell you the moment you fold.",
    btn:"Start tracking" }
];

/* ========================= state ========================= */
let landmarker=null, stepIdx=0, holdMs=0, lastT=performance.now(), lastVideoTime=-1;
let smoothLm=null, refs={}, spread=null, scoreEMA=null, score=100;
let sessionMs=0, uprightMs=0, alerts=0, streakMs=0, bestStreak=0;
let badMs=0, alerting=false, cooldownMs=0, live=false, tutorialHint=0;
let advancing=false;   // latch: stop the loop re-firing a step while it advances
let goodMs=0, nagMoveMs=0, lastNagPos=null;
let prevFeat=null, motionEMA=0, gateFailMs=0;

const HOLD = 2400, COOLDOWN = 1200;   // short re-arm; the nag clears itself on recovery
const RECOVER_MS = 600;               // how long you must look good before it lets go
const step = () => STEPS[stepIdx];

/* ==================== feature extraction ================= */
function features(lm){
  const ls=lm[L_SH], rs=lm[R_SH], nose=lm[NOSE];
  const size = dist(ls,rs);
  if(size < 1e-4) return null;
  const midX=(ls.x+rs.x)/2, midY=(ls.y+rs.y)/2;

  const tuckL = Math.abs(lm[L_EL].x - ls.x)/size;
  const tuckR = Math.abs(lm[R_EL].x - rs.x)/size;

  return {
    size,
    neck : (midY - nose.y)/size,
    tilt : (ls.y - rs.y)/size,
    off  : (nose.x - midX)/size,
    tuck : (tuckL + tuckR)/2,
    wristDrop: ((lm[L_WR].y - lm[L_EL].y) + (lm[R_WR].y - lm[R_EL].y))/2/size
  };
}

/* ======================== scoring ======================== */
function buildSpread(){
  spread = {};
  for(const d of DIMS){
    const vals = Object.values(refs).map(r=>r[d]).filter(v=>Number.isFinite(v));
    const s = vals.length>1 ? Math.max(...vals)-Math.min(...vals) : 0;
    spread[d] = Math.max(s, 1e-3);   // floor so a dim never explodes
  }
}
const delta = (a,b) => DIMS.map(d => (a[d]-b[d])/spread[d]);
const norm  = v => Math.hypot(...v);
const dot   = (a,b) => a.reduce((s,x,i)=>s+x*b[i],0);

/* how far along the ideal->bad axis the current pose sits (0 = ideal, 1 = that bad pose) */
function faultAmounts(f){
  const out = {};
  if(!refs.ideal) return out;
  const cur = delta(f, refs.ideal);
  for(const k of Object.keys(BAD_LABELS)){
    if(!refs[k]){ out[k]=0; continue; }
    const axis = delta(refs[k], refs.ideal);
    const len2 = dot(axis,axis);
    out[k] = len2 < 1e-6 ? 0 : clamp(dot(cur,axis)/len2, 0, 1.4);
  }
  return out;
}

function computeScore(f){
  if(!refs.ideal || !spread) return 100;
  const d = norm(delta(f, refs.ideal));
  const bads = Object.keys(BAD_LABELS).filter(k=>refs[k]);
  // typical distance from ideal to a bad pose = the scale of "fully wrong"
  const scale = bads.length
    ? bads.reduce((s,k)=>s+norm(delta(refs[k], refs.ideal)),0)/bads.length
    : 1;
  return clamp(100*(1 - d/Math.max(scale,1e-3)), 0, 100);
}

/* ======================== drawing ======================== */
function resize(){
  const r = overlay.getBoundingClientRect();
  const dpr = Math.min(window.devicePixelRatio||1, 2);
  overlay.width = Math.round(r.width*dpr);
  overlay.height= Math.round(r.height*dpr);
}
addEventListener("resize", resize);

function mapper(){
  const bw=overlay.width, bh=overlay.height;
  const vw=video.videoWidth||4, vh=video.videoHeight||3;
  const s=Math.max(bw/vw, bh/vh), dw=vw*s, dh=vh*s;
  const ox=(bw-dw)/2, oy=(bh-dh)/2;
  return p => ({x: ox+p.x*dw, y: oy+p.y*dh});
}

function draw(lm, col){
  ctx.clearRect(0,0,overlay.width,overlay.height);
  if(!lm) return;
  const bw=overlay.width;
  const dpr = bw / Math.max(overlay.getBoundingClientRect().width,1);
  ctx.save();
  ctx.translate(bw,0); ctx.scale(-1,1);
  const M = mapper();

  ctx.lineCap="round"; ctx.lineWidth=3*dpr; ctx.strokeStyle=col;
  for(const [a,b] of SKELETON){
    if((lm[a].visibility??1)<.35 || (lm[b].visibility??1)<.35) continue;
    const p=M(lm[a]), q=M(lm[b]);
    ctx.beginPath(); ctx.moveTo(p.x,p.y); ctx.lineTo(q.x,q.y); ctx.stroke();
  }
  ctx.fillStyle=col;
  for(const i of [NOSE,L_SH,R_SH,L_EL,R_EL,L_WR,R_WR]){
    if((lm[i].visibility??1)<.35) continue;
    const p=M(lm[i]);
    ctx.beginPath(); ctx.ellipse(p.x,p.y,7*dpr,5*dpr,0,0,Math.PI*2); ctx.fill();
  }
  ctx.restore();
}

/* ======================== framing ======================== */
function framing(lm){
  const vis = i => (lm?.[i]?.visibility ?? 0) > .5;
  const nose = vis(NOSE), sh = vis(L_SH)&&vis(R_SH);
  let dOk=false;
  if(sh){ const s=dist(lm[L_SH],lm[R_SH]); dOk = s>.14 && s<.55; }
  $("chkNose").classList.toggle("ok",nose);
  $("chkShoulders").classList.toggle("ok",sh);
  $("chkDist").classList.toggle("ok",dOk);
  return nose && sh && dOk;
}

/* ======================== gates ==========================
   A capture step only starts filling once the user is actually
   doing the pose it asked for. Each gate reports whether the
   pose is reached and, if not, what to change. Gates compare
   against the user's own normal sit, so they scale to the body
   in front of the camera rather than to fixed numbers.
   ========================================================= */
const GATES = {
  normal: (f,n,mo) => mo < 0.055
    ? {ok:true,  hint:"Hold still."}
    : {ok:false, hint:"Settle down and hold still."},

  close: (f,n) => {
    const r = f.size / n.size;                       // wider shoulders = nearer the lens
    return r > 1.10 ? {ok:true,  hint:"Hold it there."}
         : r > 1.04 ? {ok:false, hint:"Closer - keep leaning in."}
                    : {ok:false, hint:"Lean toward the screen."};
  },

  down: (f,n) => {
    const d = (n.neck - f.neck) / Math.max(n.neck, 1e-3);   // head sinking toward shoulders
    return d > 0.14 ? {ok:true,  hint:"Hold it there."}
         : d > 0.06 ? {ok:false, hint:"More - let your head sink."}
                    : {ok:false, hint:"Slump down, chin toward your chest."};
  },

  left: (f,n) => {
    const t = f.tilt - n.tilt;
    return t > 0.085 ? {ok:true,  hint:"Hold it there."}
         : t > 0.040 ? {ok:false, hint:"A little further."}
                     : {ok:false, hint:"Drop your LEFT shoulder."};
  },

  right: (f,n) => {
    const t = n.tilt - f.tilt;
    return t > 0.085 ? {ok:true,  hint:"Hold it there."}
         : t > 0.040 ? {ok:false, hint:"A little further."}
                     : {ok:false, hint:"Drop your RIGHT shoulder."};
  },

  ideal: (f,n) => {
    if(f.neck < n.neck * 0.99) return {ok:false, hint:"Lift your chest - sit taller than your normal slouch."};
    if(Math.abs(f.tilt) > 0.075) return {ok:false, hint:"Level your shoulders."};
    return {ok:true, hint:"That's the one. Hold it."};
  }
};

function gateFor(ref, f, mo){
  const g = GATES[ref];
  if(!g || !f) return {ok:false, hint:"Get your head and shoulders in frame."};
  if(ref !== "normal" && !refs.normal) return {ok:true, hint:"Hold it."};
  return g(f, refs.normal, mo);
}

/* ===================== tutorial checks =================== */
const CHECKS = {
  elbowsIn:      f => f.tuck < 0.42,
  shouldersLevel:f => f.tuck < 0.50 && Math.abs(f.tilt) < 0.07,
  armsDown:      f => f.wristDrop > 0.25
};
const CHECK_HINT = {
  elbowsIn:"Bring your elbows closer to your ribs.",
  shouldersLevel:"Level your shoulders - rotate further.",
  armsDown:"Let your hands hang below your elbows."
};

/* ======================= wizard UI ======================= */
function renderDots(){
  $("wdots").innerHTML = STEPS.map((_,i)=>
    `<i class="${i<stepIdx?"done":i===stepIdx?"now":""}"></i>`).join("");
}

function showStep(){
  const s = step();
  holdMs = 0; tutorialHint = 0; advancing = false; gateFailMs = 0;
  $("wstep").textContent = `Step ${stepIdx+1} of ${STEPS.length}`;
  $("wtitle").textContent = s.title;
  $("wbody").innerHTML = s.body;

  $("wcue").hidden = !s.cue;
  if(s.cue){ $("wcue").textContent = s.cue; $("wcue").classList.remove("ok"); }

  const needsHold = s.kind==="capture" || s.kind==="tutorial";
  $("holdwrap").hidden = !needsHold;
  $("holdfill").style.width = "0%";
  $("wframing").hidden = !(s.kind==="frame" || s.kind==="capture");

  $("wPrimary").hidden = !s.btn;
  if(s.btn) $("wPrimary").textContent = s.btn;
  $("wPrimary").disabled = false;

  // tutorial steps always offer a manual continue so detection can never trap you
  $("wSkip").hidden = !(s.kind==="tutorial" || (s.kind==="capture" && s.diff));
  $("wSkip").textContent = s.kind==="tutorial" ? "Continue" : "Skip this pose";

  renderDots();
}

function nextStep(){
  stepIdx++;
  if(step().kind==="done" && !refs.ideal) refs.ideal = refs.normal;
  showStep();
}

function capture(f, id){
  advancing = true;
  refs[id] = {...f};
  buildSpread();
  $("wcue").classList.add("ok");
  $("wcue").textContent = "Got it.";
  setTimeout(nextStep, 420);
}

/* ==================== wizard tick ======================== */
function wizardTick(f, ok, dt){
  if(advancing) return;
  const s = step();

  if(s.kind==="frame"){
    if(ok){ holdMs += dt; if(holdMs > 700) nextStep(); }
    else holdMs = 0;
    return;
  }

  if(s.kind==="capture"){
    const g = gateFor(s.ref, f, motionEMA);

    // never let a fussy gate strand someone mid-demo
    if(!g.ok && ok && f){ gateFailMs += dt; } else if(g.ok){ gateFailMs = 0; }
    const relaxed = gateFailMs > 9000;

    const pass = ok && !!f && (g.ok || relaxed);

    if(pass) holdMs += dt;
    else holdMs = Math.max(0, holdMs - dt);          // drains, so a wobble costs time not the whole hold

    $("holdfill").style.width = `${clamp(holdMs/HOLD,0,1)*100}%`;
    $("holdfill").classList.toggle("ready", pass);
    $("wcue").classList.toggle("ok", pass);
    $("wcue").textContent =
      !ok      ? "Get your head and both shoulders in frame."
      : relaxed && !g.ok ? "Close enough - just hold still."
      : pass   ? "Hold it..."
               : g.hint;

    if(holdMs >= HOLD && f) capture(f, s.ref);
    return;
  }

  if(s.kind==="tutorial"){
    const pass = f ? CHECKS[s.check](f) : false;
    if(pass) holdMs += dt; else holdMs = Math.max(0, holdMs - dt);
    $("holdfill").style.width = `${clamp(holdMs/HOLD,0,1)*100}%`;
    $("wcue").classList.toggle("ok", pass);
    if(!pass){
      tutorialHint += dt;
      if(tutorialHint > 2500) $("wcue").textContent = CHECK_HINT[s.check];
    } else {
      $("wcue").textContent = "That's it. Hold.";
    }
    if(holdMs >= HOLD) nextStep();
  }
}

/* ======================= live mode ======================= */
function goLive(){
  live = true;
  const box = $("videobox");
  box.classList.remove("fullscreen");
  $("videoslot").appendChild(box);
  $("wizard").hidden = true;
  $("app").hidden = false;
  $("scorebadge").hidden = false;
  document.body.className = "mode-live";
  video.play().catch(()=>{});
  requestAnimationFrame(resize);
}

function backToWizard(fromIdx){
  live = false;
  if(alerting) hideNag();
  const box = $("videobox");
  box.classList.add("fullscreen");
  document.body.appendChild(box);
  $("wizard").hidden = false;
  $("app").hidden = true;
  $("scorebadge").hidden = true;
  document.body.className = "mode-wizard";
  stepIdx = fromIdx; showStep();
  requestAnimationFrame(resize);
}

function renderFaults(am, worst){
  $("faults").innerHTML = Object.keys(BAD_LABELS).map(k=>{
    const pct = Math.round((am[k]||0)*100);
    return `<div class="fault ${k===worst&&pct>25?"hot":""}">
      <span>${BAD_LABELS[k]}</span><b>${pct}%</b></div>`;
  }).join("");
}

/* ========================== nag =========================== */
/* Drops the video somewhere new on screen each time, and pulls it
   back the moment the user returns to a good score. */
function placeNag(){
  const el = $("nag");
  const w = el.offsetWidth  || 260;
  const h = el.offsetHeight || 210;
  const m = 22;
  const maxX = Math.max(m, innerWidth  - w - m);
  const maxY = Math.max(m, innerHeight - h - m);
  const far  = Math.min(innerWidth, innerHeight) * 0.34;

  let x, y, tries = 0;
  do{
    x = m + Math.random()*(maxX-m);
    y = m + Math.random()*(maxY-m);
    tries++;
  } while(tries < 10 && lastNagPos &&
          Math.hypot(x-lastNagPos.x, y-lastNagPos.y) < far);   // never twice in the same spot

  lastNagPos = {x,y};
  el.style.left = x+"px";
  el.style.top  = y+"px";
}

function showNag(){
  alerting = true; alerts++; nagMoveMs = 0; goodMs = 0;
  $("stAlerts").textContent = alerts;
  $("warntext").textContent = JEERS[Math.floor(Math.random()*JEERS.length)];

  const el = $("nag");
  el.classList.remove("out");
  el.hidden = false;
  placeNag();

  const v = $("warnvid");
  v.currentTime = 0;
  v.muted = !$("soundOn").checked;
  v.play().catch(()=>{ v.muted = true; v.play().catch(()=>{}); });
}

function hideNag(){
  const el = $("nag");
  el.classList.add("out");
  setTimeout(()=>{ el.hidden = true; $("warnvid").pause(); }, 250);
  alerting = false; badMs = 0; goodMs = 0; cooldownMs = COOLDOWN;
}

/* ========================= loop ========================== */
function loop(){
  requestAnimationFrame(loop);
  if(!landmarker || video.readyState < 2) return;

  // Only advance the clock on frames we actually process. The camera
  // runs slower than the display, so charging dt on every animation
  // frame made every timer accrue at roughly half real speed.
  if(video.currentTime === lastVideoTime) return;
  lastVideoTime = video.currentTime;

  const now = performance.now();
  const dt = Math.min(now-lastT, 100); lastT = now;

  const lm = landmarker.detectForVideo(video, now)?.landmarks?.[0] || null;

  if(lm){
    if(!smoothLm || smoothLm.length !== lm.length) smoothLm = lm.map(p=>({...p}));
    else lm.forEach((p,i)=>{
      smoothLm[i].x += (p.x-smoothLm[i].x)*0.45;
      smoothLm[i].y += (p.y-smoothLm[i].y)*0.45;
      smoothLm[i].visibility = p.visibility;
    });
  } else smoothLm = null;

  const ok = framing(smoothLm || []);
  const f  = smoothLm ? features(smoothLm) : null;

  // how much the pose is moving right now, used to tell "settled" from "still moving"
  if(f && prevFeat){
    const m = Math.abs(f.neck-prevFeat.neck)
            + Math.abs(f.tilt-prevFeat.tilt)
            + Math.abs(f.off -prevFeat.off)
            + Math.abs(f.size-prevFeat.size)/Math.max(f.size,1e-3);
    motionEMA += (m*(33/Math.max(dt,1)) - motionEMA) * 0.25;
  }
  prevFeat = f;
  let col = "#ff7a2f";

  if(!live){
    wizardTick(f, ok, dt);
    if(step().kind==="tutorial" && f) col = CHECKS[step().check](f) ? "#3ddc97" : "#ff7a2f";
  } else if(f && ok){
    const raw = computeScore(f);
    // Fall fast so the grace period starts the instant you slump; rise
    // gently so recovery stays as steady as it already feels.
    const a = raw < scoreEMA ? 0.45 : 0.16;
    scoreEMA = scoreEMA===null ? raw : scoreEMA + (raw-scoreEMA)*a;
    score = scoreEMA;

    const th = +$("thresh").value, graceMs = +$("grace").value*1000;
    const good = score >= th;
    col = good ? "#3ddc97" : (score >= th-18 ? "#ffb020" : "#ff4d4d");

    const am = faultAmounts(f);
    const worst = Object.keys(BAD_LABELS).reduce((a,b)=>(am[b]||0)>(am[a]||0)?b:a, "close");
    renderFaults(am, worst);
    const dg = $("diagnosis");
    dg.textContent = good ? "Looking good." : BAD_LABELS[worst];
    dg.className = "diagnosis " + (good?"good":"bad");

    sessionMs += dt;
    if(good){ uprightMs+=dt; streakMs+=dt; bestStreak=Math.max(bestStreak,streakMs); }
    else streakMs = 0;

    if(cooldownMs>0) cooldownMs -= dt;

    if(alerting){
      // straighten up and it lets go by itself
      if(good){ goodMs += dt; if(goodMs >= RECOVER_MS) hideNag(); }
      else{
        goodMs = 0;
        nagMoveMs += dt;
        if(nagMoveMs > 7000){ nagMoveMs = 0; placeNag(); }   // keep moving if ignored
      }
    } else if(!good){
      badMs += dt;                                      // counts during re-arm too
      if(badMs >= graceMs && cooldownMs <= 0) showNag();
    } else { badMs = 0; }

    $("scorenum").textContent = Math.round(score);
    $("scorenum").style.color = col;
    $("scorelabel").textContent = good ? "good" : "shrimping";
    $("meterfill").style.width = `${score}%`;
    $("meterfill").style.background = col;
    $("stateLabel").textContent = good ? "upright" : "shrimping";
    $("stUpright").textContent = sessionMs>0 ? `${Math.round(uprightMs/sessionMs*100)}%` : "0%";
    $("stSession").textContent = fmt(sessionMs);
    $("stStreak").textContent  = fmt(bestStreak);
  } else if(live){
    $("stateLabel").textContent = "lost you";
    col = "#6f6762";
  }

  draw(smoothLm, col);
}
function fmt(ms){ const s=Math.floor(ms/1000); return `${Math.floor(s/60)}:${String(s%60).padStart(2,"0")}`; }

/* ========================= start ========================= */
async function startCamera(){
  $("wPrimary").disabled = true;
  $("wPrimary").textContent = "Starting...";
  try{
    video.srcObject = await navigator.mediaDevices.getUserMedia({
      video:{width:{ideal:1280},height:{ideal:720},facingMode:"user"}, audio:false});
    await video.play();
  }catch(e){
    $("wPrimary").disabled=false; $("wPrimary").textContent="Start camera";
    $("wbody").innerHTML = `Camera blocked: <b>${e.message}</b>. Allow access and try again.`;
    return;
  }
  try{
    const vision = await FilesetResolver.forVisionTasks(
      "https://cdn.jsdelivr.net/npm/@mediapipe/tasks-vision@0.10.14/wasm");
    landmarker = await PoseLandmarker.createFromOptions(vision,{
      baseOptions:{
        modelAssetPath:"https://storage.googleapis.com/mediapipe-models/pose_landmarker/pose_landmarker_lite/float16/1/pose_landmarker_lite.task",
        delegate:"GPU"},
      runningMode:"VIDEO", numPoses:1});
  }catch(e){
    $("wbody").innerHTML = `Could not load the pose model: <b>${e.message}</b>`;
    return;
  }
  resize(); lastT = performance.now(); nextStep(); loop();
}

/* ======================== controls ======================= */
$("wPrimary").addEventListener("click", ()=>{
  const s = step();
  if(s.kind==="intro") startCamera();
  else if(s.kind==="done") goLive();
});

$("wSkip").addEventListener("click", ()=>{
  const s = step();
  if(s.kind==="tutorial") nextStep();
  else if(s.kind==="capture") nextStep();   // reference stays undefined; scoring adapts
});

$("tutorialBtn").addEventListener("click", ()=> backToWizard(STEPS.findIndex(s=>s.id==="t1")));
$("recalBtn").addEventListener("click", ()=>{
  refs = {}; spread = null; scoreEMA = null;
  backToWizard(STEPS.findIndex(s=>s.id==="normal"));
});
$("resetBtn").addEventListener("click", ()=>{
  sessionMs=uprightMs=alerts=streakMs=bestStreak=badMs=0; cooldownMs=0; scoreEMA=null;
  $("stUpright").textContent="0%"; $("stSession").textContent="0:00";
  $("stAlerts").textContent="0"; $("stStreak").textContent="0:00";
});
$("warnClose").addEventListener("click", hideNag);
addEventListener("resize", ()=>{ if(alerting) placeNag(); });
$("thresh").addEventListener("input", e=> $("thVal").textContent = e.target.value);
$("grace").addEventListener("input",  e=> $("graceVal").textContent = e.target.value+"s");

$("videobox").classList.add("fullscreen");
showStep();
resize();
