// =============================================================================
//  main.js: shared hand-tracking loop + mode switching
// =============================================================================
//
//  Structure
//  ---------
//  main.js         camera, model, tracking loop, skeleton, particle trails,
//                  mode switching. Shared by every mode.
//  gestures.js     landmark map + finger-extended / pose helpers + filters.
//  particles.js    allocation-free glowing particle pool.
//  modes/draw.js   Draw mode (air drawing).
//  modes/game.js   Game mode (Bubble Pop).
//  modes/sign.js   Sign mode (ASL fingerspelling).
//
//  Each frame:  track hands -> skeleton -> activeMode.update()
//               -> particle trails -> activeMode.render() -> labels
//  Only the ACTIVE mode's update/render run, so Draw poses can't draw while
//  you're in Game or Sign mode.
//
//  Canvas layers (bottom to top): video, #drawCanvas (persistent strokes,
//  only visible in Draw mode), #overlay (cleared and redrawn every frame).
// =============================================================================

// Load the MediaPipe Tasks Vision library as an ES module from jsDelivr.
// The version is pinned so the demo doesn't break when a new release ships.
import {
  FilesetResolver,
  HandLandmarker,
} from "https://cdn.jsdelivr.net/npm/@mediapipe/tasks-vision@0.10.14/vision_bundle.mjs";

import {
  WRIST, INDEX_TIP, HAND_CONNECTIONS, FINGERTIPS,
  toPixels, handSize, readFingers, OneEuroFilter,
} from "./gestures.js";
import { ParticleSystem } from "./particles.js";
import { drawMode } from "./modes/draw.js";
import { gameMode } from "./modes/game.js";
import { signMode } from "./modes/sign.js";

// Where the WebAssembly runtime lives (same package/version as above).
const WASM_URL = "https://cdn.jsdelivr.net/npm/@mediapipe/tasks-vision@0.10.14/wasm";
// Google's official pre-trained hand landmark model.
const MODEL_URL =
  "https://storage.googleapis.com/mediapipe-models/hand_landmarker/hand_landmarker/float16/1/hand_landmarker.task";

// MediaPipe decides Left/Right assuming the image it receives is ALREADY
// mirrored (like a selfie). We feed it the raw, un-mirrored webcam frame, so
// its labels come out swapped. We swap them back so "Right" = your real right
// hand. If your labels look wrong on your setup, set this to false.
const SWAP_HANDEDNESS = true;

// If a hand disappears for up to this many frames (e.g. motion blur), keep
// its state alive. Draw mode keeps the stroke open and interpolates the gap.
const LOST_GRACE_FRAMES = 6;

// One color per hand for the skeleton/labels.
const HAND_COLORS = { Left: "#38bdf8", Right: "#fb7185" };

// Particle trail tuning. Rates are particles per second.
const TRAIL = {
  capacity: 300,     // hard cap on live particles (keeps FPS safe)
  baseRate: 6,       // emitted even when the fingertip is still
  speedRate: 0.35,   // extra particles per (canvas px / second) of speed
  maxPerFrame: 12,   // never more than this per hand per frame
  minLife: 0.5,      // seconds
  maxLife: 1.0,
};

// ---- DOM elements ----
const $ = (id) => document.getElementById(id);
const video = $("video");
const overlay = $("overlay");
const drawCanvas = $("drawCanvas");
const stage = $("stage");
const stageWrap = $("stageWrap");
const octx = overlay.getContext("2d");

const statusChip = $("statusChip");
const statusText = $("statusText");
const modeBadge = $("modeBadge");
const modeBadgeText = $("modeBadgeText");
const handChip = $("handChip");
const emptyState = $("emptyState");
const sidePanel = $("sidePanel");
const stageCol = document.querySelector(".stage-col");
const messageEl = $("message");
const msgTitle = $("msgTitle");
const msgText = $("msgText");
const skeletonToggle = $("showSkeleton");
const trailsToggle = $("trails");
const intensitySlider = $("trailIntensity");

// ---- Shared state ----
let handLandmarker = null;
let currentDelegate = "GPU";
let lastVideoTime = -1;
let lastFrameTime = performance.now();
let fps = 0;
let inferMs = 0;
let trailIntensity = 1;

// ADAPTIVE EFFECTS BUDGET
// fxQuality (0.25..1) scales particle emission and the particle cap. Once a
// second we look at the RENDER rate (how often requestAnimationFrame fires,
// which doesn't depend on the camera's frame rate). If rendering falls
// behind, effects are cut back; when it's smooth again they slowly return.
// So weaker GPUs get fewer particles instead of a stuttering app.
const FX = { lowFps: 45, highFps: 55, min: 0.25, down: 0.7, up: 0.1 };
let fxQuality = 1;
let rafFps = 60;
let lastRaf = 0;
let fxCheckAt = 0;

// One Euro settings shared by every fingertip filter (sliders edit these).
const filterConfig = { minCutoff: 1.0, beta: 0.008 };

const particles = new ParticleSystem(TRAIL.capacity);

// =============================================================================
//  Tracked hands (shared by all modes)
// =============================================================================
// Keyed by "Left"/"Right". Everything is allocated once per hand and reused
// each frame, so the hot loop doesn't create garbage.
const hands = {};
const visible = [];      // hands seen THIS frame (reused array)
const seen = new Set();  // labels seen this frame (reused)

function getHand(label) {
  if (!hands[label]) {
    hands[label] = {
      label,
      pts: Array.from({ length: 21 }, () => ({ x: 0, y: 0, z: 0 })), // canvas pixels
      tip: { x: 0, y: 0 },       // One Euro filtered index fingertip
      prevTip: { x: 0, y: 0 },   // filtered tip on the previous frame
      vx: 0, vy: 0, speed: 0,    // fingertip velocity, canvas px / second
      fx: new OneEuroFilter(filterConfig),
      fy: new OneEuroFilter(filterConfig),
      size: 1,                   // wrist -> middle knuckle, canvas px
      fingers: { thumb: false, index: false, middle: false, ring: false, pinky: false, count: 0 },
      missing: 0,                // frames since last seen
      fresh: true,               // first frame of this hand
      emitAcc: 0,                // fractional particles carried to the next frame
    };
  }
  return hands[label];
}

// =============================================================================
//  Modes
// =============================================================================

const MODES = { draw: drawMode, game: gameMode, sign: signMode };
const MODE_KEYS = { Digit1: "draw", Digit2: "game", Digit3: "sign" };
let activeMode = drawMode;
let shownBadge = "";

function setMode(id) {
  const next = MODES[id];
  if (!next || next === activeMode) return;
  activeMode.exit();
  activeMode = next;
  activeMode.enter();
  document.body.dataset.mode = id;
  document.querySelectorAll(".mode-tab").forEach((b) => {
    const on = b.dataset.mode === id;
    b.classList.toggle("active", on);
    b.setAttribute("aria-selected", on);
  });
  // Show only the controls/hints that belong to this mode.
  document.querySelectorAll("[data-for]").forEach((el) => (el.hidden = el.dataset.for !== id));
  shownHands = null; // re-evaluate the empty state for this mode
  updateStageChips();
  updateFullscreenAllowed();
  fitStage();
  console.info(`[HandLab] Mode: ${next.name}`);
}

function setBadge({ text, tone }) {
  const key = text + tone;
  if (key === shownBadge) return; // only touch the DOM when it changes
  shownBadge = key;
  modeBadgeText.textContent = text;
  modeBadge.dataset.mode = tone;
}

// On-video chips + empty state. A mode's badge can set keepWhenNoHand (e.g.
// a running game round); otherwise no visible hand shows "NO HAND".
let shownHands = null;
let cameraRunning = false;
function updateStageChips() {
  const b = activeMode.badge();
  setBadge(visible.length || b.keepWhenNoHand ? b : { text: "NO HAND", tone: "NONE" });

  const labels = visible.length === 2 ? "Both hands" : visible.length ? `${visible[0].label} hand` : "";
  const showEmpty = cameraRunning && !visible.length && activeMode.id !== "game";
  const key = labels + showEmpty;
  if (key === shownHands) return;
  shownHands = key;
  handChip.textContent = labels;
  handChip.hidden = !labels;
  emptyState.hidden = !showEmpty;
}

// =============================================================================
//  UI wiring
// =============================================================================

function showMessage(title, html, { error = false } = {}) {
  msgTitle.textContent = title;
  msgText.innerHTML = html;
  messageEl.classList.toggle("error", error);
  messageEl.classList.remove("hidden");
}

function hideMessage() {
  messageEl.classList.add("hidden");
}

function bindSlider(slider, outputId, onChange, format = (v) => v) {
  const out = $(outputId);
  const update = () => {
    out.textContent = format(slider.value);
    onChange(Number(slider.value));
  };
  slider.addEventListener("input", update);
  update();
}

bindSlider($("minCutoff"), "minCutoffValue", (v) => (filterConfig.minCutoff = v));
bindSlider($("beta"), "betaValue", (v) => (filterConfig.beta = v));
bindSlider(intensitySlider, "trailIntensityValue", (v) => (trailIntensity = v), (v) => `${Math.round(v * 100)}%`);

document.querySelectorAll(".mode-tab").forEach((b) =>
  b.addEventListener("click", () => setMode(b.dataset.mode))
);

window.addEventListener("keydown", (e) => {
  // Don't steal keys while typing in a form field (text, number, select…),
  // or while a modal window (e.g. Sign > Evaluate) is open.
  const t = e.target;
  if (t instanceof Element && t.closest("input, select, textarea") &&
    !(t instanceof HTMLInputElement && (t.type === "checkbox" || t.type === "range"))) return;
  if (document.querySelector(".modal:not([hidden])")) return;
  // 1 / 2 / 3 switch modes (no modifiers: Shift+digits belong to the mode).
  if (!e.shiftKey && !e.ctrlKey && !e.metaKey && !e.altKey && MODE_KEYS[e.code]) {
    setMode(MODE_KEYS[e.code]);
    return;
  }
  if (activeMode.onKey(e)) return;
  if (e.ctrlKey || e.metaKey || e.altKey) return;
  const key = e.key.toLowerCase();
  if (e.key === "?") openHelp();
  else if (key === "f") toggleFullscreen();
  else if (key === "s") skeletonToggle.checked = !skeletonToggle.checked;
  else if (key === "t") trailsToggle.checked = !trailsToggle.checked;
});

// Full screen: a CSS layout (body.fullscreen) plus the browser's Fullscreen API
// where available. Browsers without it (e.g. iPhone Safari) still get the
// layout, which fills the window. Esc / browser exit also turns the layout off.
const fullscreenBtn = $("fullscreenBtn");
function setFullscreenLayout(on) {
  document.body.classList.toggle("fullscreen", on);
  fullscreenBtn.setAttribute("aria-pressed", on);
  fullscreenBtn.querySelector("use").setAttribute("href", on ? "#i-shrink" : "#i-expand");
  fullscreenBtn.title = on ? "Exit full screen (F)" : "Full screen (F)";
  fitStage();
}
function toggleFullscreen() {
  const on = !document.body.classList.contains("fullscreen");
  if (on && fullscreenBtn.disabled) return;
  setFullscreenLayout(on);
  if (on) document.documentElement.requestFullscreen?.().catch(() => {});
  else if (document.fullscreenElement) document.exitFullscreen();
}
// A mode can turn full screen off (Sign only allows it in Spell). Called on
// mode and sub-mode changes; leaves full screen if it's no longer allowed.
function updateFullscreenAllowed() {
  const allowed = activeMode.allowsFullscreen?.() ?? true;
  if (!allowed && document.body.classList.contains("fullscreen")) toggleFullscreen();
  fullscreenBtn.disabled = !allowed;
  if (!allowed) fullscreenBtn.title = "Full screen is only available in Spell";
  else if (!document.body.classList.contains("fullscreen")) fullscreenBtn.title = "Full screen (F)";
}
fullscreenBtn.addEventListener("click", toggleFullscreen);
document.addEventListener("fullscreenchange", () => {
  if (!document.fullscreenElement) setFullscreenLayout(false);
});

// Help window (? button / ? key). Opens on the current mode's instructions.
const helpModal = $("helpModal");
function showHelpTopic(id) {
  helpModal.querySelectorAll("[data-help]").forEach((b) => b.classList.toggle("active", b.dataset.help === id));
  helpModal.querySelectorAll("[data-help-for]").forEach((el) => (el.hidden = el.dataset.helpFor !== id));
}
function openHelp() {
  showHelpTopic(activeMode.id);
  helpModal.hidden = false;
  $("helpClose").focus();
}
function closeHelp() {
  helpModal.hidden = true;
}
$("helpBtn").addEventListener("click", openHelp);
$("helpClose").addEventListener("click", closeHelp);
helpModal.addEventListener("click", (e) => {
  if (e.target === helpModal) closeHelp();
  const tab = e.target.closest("[data-help]");
  if (tab) showHelpTopic(tab.dataset.help);
});
window.addEventListener("keydown", (e) => {
  if (e.key === "Escape" && !helpModal.hidden) closeHelp();
});

// Key releases matter for "hold a key" actions (e.g. hold R to record in Sign mode).
window.addEventListener("keyup", (e) => activeMode.onKeyUp?.(e));

// Size the stage so it keeps the camera's aspect ratio and fits the window.
// This only changes CSS size; the canvases' pixel buffers are untouched,
// so resizing the window never erases the drawing.
function fitStage() {
  const aspect = video.videoWidth && video.videoHeight
    ? video.videoWidth / video.videoHeight
    : 4 / 3;
  const availW = stageWrap.clientWidth;
  // Side-by-side layout: the stage fills the height its column gives it.
  // Stacked layout (narrow screens, side panel wrapped below the video): the
  // page scrolls, so cap the video at ~65% of the window height instead.
  const stacked = sidePanel.offsetTop > stageCol.offsetTop + 8;
  stageWrap.style.height = stacked ? "auto" : "";
  const availH = stacked ? window.innerHeight * 0.65 : stageWrap.clientHeight;
  let w = availW;
  let h = w / aspect;
  if (h > availH) {
    h = availH;
    w = h * aspect;
  }
  stage.style.width = `${Math.floor(w)}px`;
  stage.style.height = `${Math.floor(h)}px`;
}
window.addEventListener("resize", fitStage);

// How many canvas pixels make one on-screen (CSS) pixel. The canvases use the
// camera's resolution, which can differ from what's displayed, so we scale
// line widths/radii to look the same size at any window size.
function pxScale() {
  return overlay.width / (overlay.clientWidth || overlay.width);
}

// =============================================================================
//  Shared overlay drawing
// =============================================================================

function drawSkeleton(pts, color) {
  const s = pxScale();

  // Bones
  octx.strokeStyle = color;
  octx.lineWidth = 3 * s;
  octx.lineCap = "round";
  octx.globalAlpha = 0.85;
  octx.beginPath();
  for (const [a, b] of HAND_CONNECTIONS) {
    octx.moveTo(pts[a].x, pts[a].y);
    octx.lineTo(pts[b].x, pts[b].y);
  }
  octx.stroke();
  octx.globalAlpha = 1;

  // Joints: white dots with a colored ring, fingertips slightly bigger
  octx.lineWidth = 2 * s;
  octx.strokeStyle = color;
  octx.fillStyle = "#fff";
  for (let i = 0; i < 21; i++) {
    octx.beginPath();
    octx.arc(pts[i].x, pts[i].y, (FINGERTIPS.includes(i) ? 5 : 3.5) * s, 0, Math.PI * 2);
    octx.fill();
    octx.stroke();
  }
}

function drawLabel(pts, text, color) {
  const s = pxScale();
  const { x, y } = pts[WRIST];
  octx.font = `600 ${14 * s}px Sora, system-ui, sans-serif`;
  const w = octx.measureText(text).width + 16 * s;
  const h = 24 * s;
  const bx = x - w / 2;
  const by = y + 14 * s;

  octx.fillStyle = color;
  octx.beginPath();
  octx.roundRect(bx, by, w, h, 6 * s);
  octx.fill();
  octx.fillStyle = "#0b0d12";
  octx.textAlign = "center";
  octx.textBaseline = "middle";
  octx.fillText(text, x, by + h / 2);
}

// Circular "hold to confirm" ring, used by any mode (clear, start game...).
function drawHoldRing(center, progress, title, subtitle, color) {
  const s = pxScale();
  const r = 48 * s;
  octx.fillStyle = "rgba(11,13,18,0.55)";
  octx.beginPath();
  octx.arc(center.x, center.y, r, 0, Math.PI * 2);
  octx.fill();
  octx.lineWidth = 7 * s;
  octx.lineCap = "round";
  octx.strokeStyle = "rgba(255,255,255,0.2)";
  octx.stroke();
  octx.strokeStyle = color;
  octx.beginPath();
  octx.arc(center.x, center.y, r, -Math.PI / 2, -Math.PI / 2 + progress * Math.PI * 2);
  octx.stroke();
  octx.fillStyle = "#fff";
  octx.textAlign = "center";
  octx.textBaseline = "middle";
  octx.font = `700 ${14 * s}px Sora, system-ui, sans-serif`;
  octx.fillText(title, center.x, center.y - 7 * s);
  octx.font = `500 ${11 * s}px Sora, system-ui, sans-serif`;
  octx.fillText(subtitle, center.x, center.y + 9 * s);
}

// Big centered title + subtitle (placeholders, game screens...).
function drawCenterText(title, subtitle) {
  const s = pxScale();
  const cx = overlay.width / 2, cy = overlay.height / 2;
  octx.textAlign = "center";
  octx.textBaseline = "middle";
  octx.fillStyle = "rgba(255,255,255,0.92)";
  octx.font = `800 ${34 * s}px Sora, system-ui, sans-serif`;
  octx.fillText(title, cx, cy - 14 * s);
  octx.fillStyle = "rgba(200,206,220,0.75)";
  octx.font = `500 ${16 * s}px Sora, system-ui, sans-serif`;
  octx.fillText(subtitle, cx, cy + 20 * s);
}

// =============================================================================
//  Particle trails
// =============================================================================
// Emission rate grows with fingertip speed: a still finger gives a faint
// shimmer, a fast swipe a dense comet tail. The fractional part of
// "rate * dt" is carried over (emitAcc) so slow rates still emit smoothly.
// New particles are spread along the path the fingertip took this frame,
// so fast moves leave a continuous trail instead of clumps.
function emitTrail(hand, color, dt) {
  const rate = (TRAIL.baseRate + hand.speed * TRAIL.speedRate) * trailIntensity * fxQuality;
  hand.emitAcc += rate * dt;
  // Trails may only fill part of the pool (scaled by fxQuality), which also
  // leaves room for bursts such as bubble pops.
  const room = Math.floor(TRAIL.capacity * 0.85 * fxQuality) - particles.count;
  const n = Math.max(0, Math.min(Math.floor(hand.emitAcc), TRAIL.maxPerFrame, room));
  hand.emitAcc -= Math.floor(hand.emitAcc);
  const s = pxScale();
  const { tip, prevTip } = hand;
  for (let i = 0; i < n; i++) {
    const t = Math.random();
    const angle = Math.random() * Math.PI * 2;
    const drift = (10 + Math.random() * 35) * s; // gentle random drift
    particles.spawn(
      prevTip.x + (tip.x - prevTip.x) * t + (Math.random() - 0.5) * 3,
      prevTip.y + (tip.y - prevTip.y) * t + (Math.random() - 0.5) * 3,
      Math.cos(angle) * drift - hand.vx * 0.08, // slight push opposite to motion
      Math.sin(angle) * drift - hand.vy * 0.08,
      (2.5 + Math.random() * 4) * s,
      TRAIL.minLife + Math.random() * (TRAIL.maxLife - TRAIL.minLife),
      color
    );
  }
}

// =============================================================================
//  Per-frame processing
// =============================================================================

function trackHands(results, now) {
  // Newer versions call it `handedness`, older ones `handednesses`.
  const handedness = results.handedness || results.handednesses || [];
  const tSec = now / 1000;
  visible.length = 0;
  seen.clear();

  for (let i = 0; i < results.landmarks.length; i++) {
    // --- Which hand is this? ---
    let label = handedness[i]?.[0]?.categoryName || "Right";
    if (SWAP_HANDEDNESS) label = label === "Left" ? "Right" : "Left";
    // If both hands get the same label (can happen briefly), keep them apart.
    if (seen.has(label)) label = label === "Left" ? "Right" : "Left";
    seen.add(label);

    const hand = getHand(label);
    hand.missing = 0;
    toPixels(results.landmarks[i], hand.pts, overlay.width, overlay.height);
    hand.size = handSize(hand.pts);
    readFingers(hand.pts, hand.fingers);

    // --- One Euro filter on the index fingertip + velocity ---
    hand.prevTip.x = hand.tip.x;
    hand.prevTip.y = hand.tip.y;
    hand.tip.x = hand.fx.filter(hand.pts[INDEX_TIP].x, tSec);
    hand.tip.y = hand.fy.filter(hand.pts[INDEX_TIP].y, tSec);
    if (hand.fresh) {
      hand.prevTip.x = hand.tip.x;
      hand.prevTip.y = hand.tip.y;
      hand.fresh = false;
    }
    visible.push(hand);
  }

  // Hands not seen this frame: short grace period, then forget them.
  for (const label in hands) {
    if (seen.has(label)) continue;
    if (++hands[label].missing > LOST_GRACE_FRAMES) {
      activeMode.handLost(label);
      delete hands[label];
    }
  }
}

function processFrame(results, now, dt) {
  trackHands(results, now);
  for (const hand of visible) {
    hand.vx = (hand.tip.x - hand.prevTip.x) / dt;
    hand.vy = (hand.tip.y - hand.prevTip.y) / dt;
    hand.speed = Math.hypot(hand.vx, hand.vy);
  }

  octx.clearRect(0, 0, overlay.width, overlay.height); // overlay ONLY

  if (skeletonToggle.checked) {
    for (const hand of visible) drawSkeleton(hand.pts, HAND_COLORS[hand.label]);
  }

  activeMode.update(visible, now);

  // Particle trails: the active mode picks the color (or null = no trail).
  for (const hand of visible) {
    const color = trailsToggle.checked ? activeMode.trailColor(hand) : null;
    if (color) emitTrail(hand, color, dt);
    else hand.emitAcc = 0;
  }
  particles.update(dt);
  particles.render(octx);

  activeMode.render(octx, visible, now);

  for (const hand of visible) {
    const suffix = activeMode.labelSuffix(hand);
    drawLabel(hand.pts, suffix ? `${hand.label} · ${suffix}` : hand.label, HAND_COLORS[hand.label]);
  }
  updateStageChips();
}

// Header status chip: "34 fps · GPU · 640×480" (+ "FX nn%" while effects are
// reduced). Green dot at 25+ fps, amber below. Inference time and the WebGL
// renderer are in its tooltip.
let delegateLabel = "—";
let resLabel = "—";
let rendererLabel = "";
let shownStatus = "";
function renderStatus() {
  const f = Math.round(fps);
  const text = `${f} fps · ${delegateLabel} · ${resLabel}` + (fxQuality < 1 ? ` · FX ${Math.round(fxQuality * 100)}%` : "");
  if (text === shownStatus) return; // only touch the DOM when it changes
  shownStatus = text;
  statusText.textContent = text;
  statusChip.dataset.state = !fps ? "" : f >= 25 ? "good" : "warn";
  statusChip.title = `Model inference ${inferMs.toFixed(1)} ms · render ${Math.round(rafFps)} fps` +
    (rendererLabel ? ` · WebGL renderer: ${rendererLabel}` : "");
}

function updateStats(now, ms) {
  const dt = now - lastFrameTime;
  if (dt > 0) {
    // Exponential moving averages so the numbers don't jump around.
    fps = fps ? fps * 0.9 + (1000 / dt) * 0.1 : 1000 / dt;
    inferMs = inferMs ? inferMs * 0.9 + ms * 0.1 : ms;
    renderStatus();
  }
  lastFrameTime = now;
}

// Called on every requestAnimationFrame (see loop()).
function updateFxBudget(now) {
  if (lastRaf) {
    const d = now - lastRaf;
    if (d > 0 && d < 500) rafFps = rafFps * 0.9 + (1000 / d) * 0.1;
  }
  lastRaf = now;
  if (now - fxCheckAt < 1000) return;
  fxCheckAt = now;
  const prev = fxQuality;
  if (rafFps < FX.lowFps) fxQuality = Math.max(FX.min, fxQuality * FX.down);
  else if (rafFps > FX.highFps) fxQuality = Math.min(1, fxQuality + FX.up);
  if (fxQuality !== prev) {
    renderStatus();
    console.info(`[HandLab] Render ${Math.round(rafFps)} fps -> effects quality ${Math.round(fxQuality * 100)}%`);
  }
}

// The main loop. requestAnimationFrame fires before every screen repaint
// (often 60-144 Hz), but webcams deliver ~30 frames/s. Checking
// video.currentTime means we only run the model on a NEW camera frame, and
// skip repaints where nothing changed.
async function loop() {
  updateFxBudget(performance.now());
  if (video.readyState >= 2 && video.currentTime !== lastVideoTime) {
    lastVideoTime = video.currentTime;
    const now = performance.now();
    let results;
    try {
      // VIDEO mode needs a timestamp that always increases.
      results = handLandmarker.detectForVideo(video, now);
    } catch (err) {
      // Some GPUs pass creation but fail at inference. Fall back once.
      if (currentDelegate === "GPU") {
        console.warn("[HandLab] GPU inference failed, switching to CPU:", err);
        handLandmarker.close();
        handLandmarker = await createLandmarker("CPU");
        requestAnimationFrame(loop);
        return;
      }
      throw err;
    }
    const ms = performance.now() - now;
    const dt = Math.min(Math.max((now - lastFrameTime) / 1000, 1 / 240), 0.1);
    processFrame(results, now, dt);
    updateStats(now, ms);
  }
  requestAnimationFrame(loop);
}

// =============================================================================
//  Setup: model + camera
// =============================================================================

// Which GPU does WebGL actually see? If this says "SwiftShader" or
// "Microsoft Basic Render Driver", the "GPU" delegate is really running on a
// software renderer (hardware acceleration is off in the browser).
function getWebGLRenderer() {
  try {
    const gl = document.createElement("canvas").getContext("webgl2");
    if (!gl) return "WebGL2 unavailable";
    const ext = gl.getExtension("WEBGL_debug_renderer_info");
    return gl.getParameter(ext ? ext.UNMASKED_RENDERER_WEBGL : gl.RENDERER);
  } catch {
    return "unknown";
  }
}

async function createLandmarker(delegate) {
  // Downloads + initializes the WebAssembly runtime (cached after the first time).
  const vision = await FilesetResolver.forVisionTasks(WASM_URL);
  const landmarker = await HandLandmarker.createFromOptions(vision, {
    baseOptions: { modelAssetPath: MODEL_URL, delegate },
    runningMode: "VIDEO", // optimized for a stream of frames, uses tracking between frames
    numHands: 2,
    minHandDetectionConfidence: 0.5,
    minHandPresenceConfidence: 0.5,
    minTrackingConfidence: 0.5,
  });
  currentDelegate = delegate;
  const renderer = getWebGLRenderer();
  const software = /swiftshader|basic render|llvmpipe|software/i.test(renderer);
  delegateLabel = delegate === "GPU" && software ? "GPU (software)" : delegate;
  rendererLabel = renderer;
  renderStatus();
  console.info(`[HandLab] HandLandmarker active delegate: ${delegate}`);
  if (delegate === "GPU") {
    console.info(`[HandLab] WebGL renderer: ${renderer}`);
    if (software) {
      console.warn("[HandLab] WebGL is using a software renderer. Enable hardware acceleration in browser settings.");
    }
  }
  return landmarker;
}

async function loadModel() {
  try {
    return await createLandmarker("GPU");
  } catch (err) {
    console.warn("[HandLab] GPU delegate failed, falling back to CPU:", err);
    showMessage("Loading hand model…", "GPU not available, using the CPU instead.");
    return await createLandmarker("CPU");
  }
}

async function startCamera() {
  // 640x480 is plenty for hand tracking (the model works on a 224px crop)
  // and much cheaper to upload to the GPU every frame than 720p/1080p.
  const stream = await navigator.mediaDevices.getUserMedia({
    video: { facingMode: "user", width: { ideal: 640 }, height: { ideal: 480 } },
    audio: false,
  });
  video.srcObject = stream;
  await new Promise((resolve) => (video.onloadedmetadata = resolve));
  await video.play();

  // Match the canvases' internal resolution to the camera's, so landmark
  // coordinates (0..1) map 1:1 onto video pixels. Done ONCE: resizing a
  // canvas wipes it, so this must never run per frame.
  for (const c of [overlay, drawCanvas]) {
    c.width = video.videoWidth;
    c.height = video.videoHeight;
  }
  resLabel = `${video.videoWidth}×${video.videoHeight}`;
  renderStatus();
  console.info(`[HandLab] Camera resolution: ${video.videoWidth}x${video.videoHeight}`);
  fitStage();
}

function cameraErrorMessage(err) {
  switch (err.name) {
    case "NotAllowedError":
    case "SecurityError":
      return ["Camera permission denied",
        "Allow camera access using the icon in your browser's address bar, then reload the page."];
    case "NotFoundError":
    case "OverconstrainedError":
      return ["No camera found",
        "Plug in a webcam (or enable your built-in one) and reload the page."];
    case "NotReadableError":
      return ["Camera is busy",
        "Another app (Zoom, Teams, OBS…) may be using the camera. Close it and reload."];
    default:
      return ["Couldn't start the camera", `${err.name}: ${err.message}`];
  }
}

// Shared helpers handed to every mode.
const app = {
  overlay, octx, drawCanvas, particles, pxScale,
  fxQuality: () => fxQuality, // 0.25..1, modes scale their effects by it
  refit: fitStage,            // call after showing/hiding something next to the video
  updateFullscreen: updateFullscreenAllowed, // call when allowsFullscreen() may have changed
  handColors: HAND_COLORS,
  drawHoldRing, drawCenterText,
};

async function init() {
  for (const mode of Object.values(MODES)) mode.init(app);
  document.body.dataset.mode = "draw";
  document.querySelectorAll("[data-for]").forEach((el) => (el.hidden = el.dataset.for !== "draw"));
  fitStage();

  // Webcams only work in a "secure context": https:// or http://localhost.
  // Opening index.html directly (file://) is NOT one.
  if (!window.isSecureContext || !navigator.mediaDevices?.getUserMedia) {
    showMessage(
      "Needs a local server",
      "The webcam only works over <code>http://localhost</code> or HTTPS. " +
        "Run <code>npx serve</code> or <code>python -m http.server</code> in this folder " +
        "and open the localhost URL it prints.",
      { error: true }
    );
    return;
  }

  try {
    showMessage("Loading hand model…", "Downloading MediaPipe HandLandmarker. This takes a few seconds the first time.");
    handLandmarker = await loadModel();
  } catch (err) {
    console.error(err);
    showMessage("Couldn't load the model",
      "Check your internet connection (the model and WASM load from Google/jsDelivr) and reload.<br><br>" +
        `<small>${err.message || err}</small>`,
      { error: true });
    return;
  }

  try {
    showMessage("Starting camera…", "Click <b>Allow</b> if your browser asks for camera access.");
    await startCamera();
  } catch (err) {
    console.error(err);
    const [title, text] = cameraErrorMessage(err);
    showMessage(title, text, { error: true });
    return;
  }

  hideMessage();
  cameraRunning = true;
  updateStageChips();
  requestAnimationFrame(loop);
}

init();
