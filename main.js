// =============================================================================
//  Air Draw: real-time hand tracking with MediaPipe HandLandmarker
// =============================================================================
//
//  HOW THE 21 HAND LANDMARKS ARE NUMBERED
//  --------------------------------------
//  For every detected hand, MediaPipe returns 21 points. Each one is an object
//  { x, y, z }, where x and y are NORMALIZED to the image: 0..1 across the
//  width and height (0,0 = top-left of the camera frame). z is relative depth
//  (smaller = closer to the camera), roughly in the same units as x.
//
//                 8   12  16  20        <- fingertips (TIP)
//                 |   |   |   |
//                 7   11  15  19        <- DIP joints
//            4    |   |   |   |
//            |    6   10  14  18        <- PIP joints
//            3    |   |   |   |
//             \   5---9---13--17        <- knuckles (MCP joints)
//              2   \          /
//               \   \        /
//                1---\------/
//                     \    /
//                      \  /
//                       0               <- WRIST
//
//   0       wrist
//   1 - 4   thumb : 1 CMC, 2 MCP, 3 IP,  4 TIP
//   5 - 8   index : 5 MCP, 6 PIP, 7 DIP, 8 TIP
//   9 - 12  middle: 9 MCP, 10 PIP, 11 DIP, 12 TIP
//  13 - 16  ring  : 13 MCP, 14 PIP, 15 DIP, 16 TIP
//  17 - 20  pinky : 17 MCP, 18 PIP, 19 DIP, 20 TIP
//
//  Pattern: each finger has 4 points going from the palm outwards, and the
//  tip is always the last one: 4, 8, 12, 16, 20.
//
//  FINGER-COUNT CONTROLS
//  ---------------------
//  ☝  index only            -> DRAW  (pen down, fingertip draws)
//  ✌  index + middle        -> HOVER (cursor follows, pen up)
//     anything else         -> IDLE  (fist, open palm, ...)
//
//  TWO CANVAS LAYERS
//  -----------------
//  #drawCanvas : your strokes. NEVER cleared per frame. It's only wiped by
//                Clear, or redrawn from the `strokes` array after Undo.
//  #overlay    : skeleton, labels, cursor, progress ring. Cleared and
//                redrawn every frame.
// =============================================================================

// Load the MediaPipe Tasks Vision library as an ES module from jsDelivr.
// The version is pinned so the demo doesn't break when a new release ships.
import {
  FilesetResolver,
  HandLandmarker,
} from "https://cdn.jsdelivr.net/npm/@mediapipe/tasks-vision@0.10.14/vision_bundle.mjs";

// Where the WebAssembly runtime lives (same package/version as above).
const WASM_URL = "https://cdn.jsdelivr.net/npm/@mediapipe/tasks-vision@0.10.14/wasm";
// Google's official pre-trained hand landmark model.
const MODEL_URL =
  "https://storage.googleapis.com/mediapipe-models/hand_landmarker/hand_landmarker/float16/1/hand_landmarker.task";

// ---- Landmark indices we use by name (see the diagram above) ----
const WRIST = 0;
const THUMB_MCP = 2;
const THUMB_IP = 3;
const THUMB_TIP = 4;
const INDEX_MCP = 5;
const INDEX_TIP = 8;
const MIDDLE_MCP = 9;
const PINKY_MCP = 17;
const PINKY_TIP = 20;

// [MCP, PIP, DIP, TIP] for the four fingers.
const INDEX = [5, 6, 7, 8];
const MIDDLE = [9, 10, 11, 12];
const RING = [13, 14, 15, 16];
const PINKY = [17, 18, 19, 20];
const FINGERS = [INDEX, MIDDLE, RING, PINKY];

// Pairs of landmark indices that form the "bones" we draw as lines.
const HAND_CONNECTIONS = [
  [0, 1], [1, 2], [2, 3], [3, 4],          // thumb
  [0, 5], [5, 6], [6, 7], [7, 8],          // index
  [5, 9], [9, 10], [10, 11], [11, 12],     // middle
  [9, 13], [13, 14], [14, 15], [15, 16],   // ring
  [13, 17], [17, 18], [18, 19], [19, 20],  // pinky
  [0, 17],                                 // base of the palm
];

// ---- Tuning knobs ----

// FINGER EXTENSION (see isFingerExtended()): a finger counts as extended
// when its tip is at least EXTENDED_RATIO times farther from the wrist than
// its PIP joint is. Curled fingers fold the tip back toward the palm, so the
// ratio drops below 1.
const EXTENDED_RATIO = 1.15;

// Debounce: a new pose must be seen this many frames in a row before the
// mode switches. One noisy frame can't start or end a stroke.
const POSE_FRAMES = 3;

// If a hand disappears for up to this many frames (e.g. motion blur), keep
// its stroke alive. When it comes back, the gap is filled by interpolation.
const LOST_GRACE_FRAMES = 6;

// INTERPOLATION: if the fingertip jumps more than JUMP of the canvas width in
// one frame, insert extra points every STEP so the curve stays continuous.
const JUMP = 0.04;
const STEP = 0.015;

// OPEN-PALM CLEAR (off by default): all 5 fingers straight and spread,
// held still for PALM_HOLD_MS.
const PALM_HOLD_MS = 1500;
const STRAIGHT = 0.9;        // finger straightness, 1.0 = perfectly straight
const PALM_STILL = 0.25;     // max palm drift during the hold, in hand sizes

// MediaPipe decides Left/Right assuming the image it receives is ALREADY
// mirrored (like a selfie). We feed it the raw, un-mirrored webcam frame, so
// its labels come out swapped. We swap them back so "Right" = your real right
// hand. If your labels look wrong on your setup, set this to false.
const SWAP_HANDEDNESS = true;

// One color per hand for the skeleton/labels.
const HAND_COLORS = { Left: "#38bdf8", Right: "#fb7185" };

// ---- DOM elements ----
const $ = (id) => document.getElementById(id);
const video = $("video");
const overlay = $("overlay");
const drawCanvas = $("drawCanvas");
const stage = $("stage");
const stageWrap = $("stageWrap");
const octx = overlay.getContext("2d");
const dctx = drawCanvas.getContext("2d");

const fpsEl = $("fps");
const inferEl = $("infer");
const delegateEl = $("delegate");
const resEl = $("resolution");
const modeBadge = $("modeBadge");
const messageEl = $("message");
const msgTitle = $("msgTitle");
const msgText = $("msgText");
const brushSlider = $("brushSize");
const brushValue = $("brushValue");
const skeletonToggle = $("showSkeleton");
const palmToggle = $("palmClear");
const minCutoffSlider = $("minCutoff");
const betaSlider = $("beta");
const swatches = [...document.querySelectorAll(".swatch")];

// ---- App state ----
let handLandmarker = null;
let currentDelegate = "GPU";
let lastVideoTime = -1;
let lastFrameTime = performance.now();
let fps = 0;
let inferMs = 0;

let brushColor = swatches[0].dataset.color;
let brushSize = Number(brushSlider.value);
let minCutoff = Number(minCutoffSlider.value);
let beta = Number(betaSlider.value);

// Every finished or in-progress stroke lives here, each with its own points,
// color and size. The drawing canvas is just a picture of this array.
// Shape: { color: "#22d3ee", size: 8, points: [{x, y}, ...] }
const strokes = [];

// =============================================================================
//  One Euro filter
// =============================================================================
//  A low-pass filter whose cutoff frequency adapts to speed:
//    - moving slowly -> low cutoff  -> heavy smoothing -> steady line, no jitter
//    - moving fast   -> high cutoff -> light smoothing -> no lag
//  cutoff = minCutoff + beta * |speed|
//  minCutoff: lower = steadier when slow (but more lag).
//  beta:      higher = less lag when fast.
//  Positions are in canvas pixels, so speed is in pixels/second.
//  Paper: Casiez et al., "1€ Filter", CHI 2012.

function smoothingAlpha(cutoff, dt) {
  const tau = 1 / (2 * Math.PI * cutoff);
  return 1 / (1 + tau / dt);
}

class OneEuroFilter {
  constructor(dCutoff = 1.0) {
    this.dCutoff = dCutoff;   // cutoff used to smooth the speed estimate itself
    this.reset();
  }
  reset() {
    this.x = 0;               // last filtered value
    this.dx = 0;              // last filtered speed
    this.t = -1;              // last timestamp (seconds), -1 = no data yet
  }
  filter(value, t) {
    if (this.t < 0) {
      this.x = value;
      this.t = t;
      return value;
    }
    const dt = Math.max(t - this.t, 1e-3);
    this.t = t;
    // 1. Estimate speed, and smooth it a little.
    const rawDx = (value - this.x) / dt;
    this.dx += smoothingAlpha(this.dCutoff, dt) * (rawDx - this.dx);
    // 2. Pick a cutoff from the speed: faster = higher cutoff = less smoothing.
    const cutoff = minCutoff + beta * Math.abs(this.dx);
    // 3. Standard exponential smoothing with that cutoff.
    this.x += smoothingAlpha(cutoff, dt) * (value - this.x);
    return this.x;
  }
}

// =============================================================================
//  Per-hand state
// =============================================================================
// Keyed by "Left"/"Right" so each hand has its own pen. Everything a hand
// needs per frame is allocated once here and reused, so the hot loop
// doesn't create garbage.
const hands = {};
function getHandState(label) {
  if (!hands[label]) {
    hands[label] = {
      pts: Array.from({ length: 21 }, () => ({ x: 0, y: 0 })), // reused every frame
      fx: new OneEuroFilter(),
      fy: new OneEuroFilter(),
      tip: { x: 0, y: 0 },   // filtered index fingertip, canvas pixels
      mode: "IDLE",          // confirmed mode: "DRAW" | "HOVER" | "IDLE"
      candidate: "IDLE",     // the pose seen on recent frames
      candidateFrames: 0,    // how many frames in a row `candidate` was seen
      blocked: false,        // true after Undo, until the hand leaves DRAW
      stroke: null,          // the stroke this hand is currently drawing
      missing: 0,            // frames since this hand was last seen
      palmStart: -1,         // when the open-palm hold began (-1 = not holding)
      palmAnchor: { x: 0, y: 0 },
      palmArmed: true,       // false after a clear, until the palm closes
    };
  }
  return hands[label];
}

// =============================================================================
//  Stroke rendering: quadratic Bezier curves through midpoints
// =============================================================================
//  Joining raw points with straight lines gives visible corners. Instead,
//  each curve runs from the midpoint of (p[i-1], p[i]) to the midpoint of
//  (p[i], p[i+1]), using p[i] as the control point. Consecutive curves meet
//  at midpoints with the same direction, so the line is smooth everywhere.
//
//     p0 ---- m1 ~~~~ m2 ~~~~ m3 ---- p_last
//              (ctrl p1) (ctrl p2)

function setStrokeStyle(ctx, s) {
  ctx.strokeStyle = s.color;
  ctx.fillStyle = s.color;
  ctx.lineWidth = s.size;
  ctx.lineCap = "round";
  ctx.lineJoin = "round";
}

// A dot at the first point, so a quick tap still leaves a mark.
function drawStrokeStart(ctx, s) {
  const p = s.points[0];
  setStrokeStyle(ctx, s);
  ctx.beginPath();
  ctx.arc(p.x, p.y, s.size / 2, 0, Math.PI * 2);
  ctx.fill();
}

// Draw the piece of curve that became known once point n was added.
function drawStrokeSegment(ctx, s, n) {
  const p = s.points;
  setStrokeStyle(ctx, s);
  ctx.beginPath();
  if (n === 1) {
    // First piece: straight from the start point to the first midpoint.
    ctx.moveTo(p[0].x, p[0].y);
    ctx.lineTo((p[0].x + p[1].x) / 2, (p[0].y + p[1].y) / 2);
  } else {
    const a = p[n - 2], b = p[n - 1], c = p[n];
    ctx.moveTo((a.x + b.x) / 2, (a.y + b.y) / 2);
    ctx.quadraticCurveTo(b.x, b.y, (b.x + c.x) / 2, (b.y + c.y) / 2);
  }
  ctx.stroke();
}

// Last piece: from the final midpoint to the actual last point.
function drawStrokeEnd(ctx, s) {
  const p = s.points;
  const n = p.length - 1;
  if (n < 1) return;
  setStrokeStyle(ctx, s);
  ctx.beginPath();
  ctx.moveTo((p[n - 1].x + p[n].x) / 2, (p[n - 1].y + p[n].y) / 2);
  ctx.lineTo(p[n].x, p[n].y);
  ctx.stroke();
}

function renderStroke(ctx, s) {
  drawStrokeStart(ctx, s);
  for (let i = 1; i < s.points.length; i++) drawStrokeSegment(ctx, s, i);
  drawStrokeEnd(ctx, s);
}

// Rebuild the drawing canvas from the strokes array (used after Undo).
function redrawAll() {
  dctx.clearRect(0, 0, drawCanvas.width, drawCanvas.height);
  for (const s of strokes) renderStroke(dctx, s);
}

// Append a point to a stroke and draw just the new piece (incremental, cheap).
// If the fingertip jumped far since the last point (fast move or a few lost
// frames), insert evenly spaced points in between so the line isn't broken.
function addPoint(s, x, y) {
  const last = s.points[s.points.length - 1];
  const d = Math.hypot(x - last.x, y - last.y);
  if (d < 0.75) return; // ignore sub-pixel noise

  const jump = JUMP * drawCanvas.width;
  const step = STEP * drawCanvas.width;
  if (d > jump) {
    const extra = Math.floor(d / step);
    for (let i = 1; i < extra; i++) {
      const t = i / extra;
      s.points.push({ x: last.x + (x - last.x) * t, y: last.y + (y - last.y) * t });
      drawStrokeSegment(dctx, s, s.points.length - 1);
    }
  }
  s.points.push({ x, y });
  drawStrokeSegment(dctx, s, s.points.length - 1);
}

function beginStroke(state) {
  const s = {
    color: brushColor,
    size: brushSize * pxScale(), // stored in canvas pixels
    points: [{ x: state.tip.x, y: state.tip.y }],
  };
  strokes.push(s);
  state.stroke = s;
  drawStrokeStart(dctx, s);
}

function endStroke(state) {
  if (state.stroke) drawStrokeEnd(dctx, state.stroke);
  state.stroke = null;
}

// =============================================================================
//  Clear / Undo
// =============================================================================

function clearDrawing() {
  strokes.length = 0;
  for (const label in hands) hands[label].stroke = null;
  // A hand that's still in DRAW mode carries on with a fresh stroke.
  for (const label in hands) if (hands[label].mode === "DRAW") beginStroke(hands[label]);
  dctx.clearRect(0, 0, drawCanvas.width, drawCanvas.height);
}

function undo() {
  const removed = strokes.pop();
  if (!removed) return;
  // If a hand was drawing the stroke we just removed, lift its pen so it
  // doesn't keep drawing into a deleted stroke. It has to leave DRAW pose
  // (e.g. go to HOVER) and come back to start drawing again.
  for (const label in hands) {
    if (hands[label].stroke === removed) {
      hands[label].stroke = null;
      hands[label].blocked = true;
    }
  }
  redrawAll();
}

// =============================================================================
//  UI
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

function setColor(color) {
  brushColor = color;
  swatches.forEach((s) => s.classList.toggle("active", s.dataset.color === color));
}

function setBrushSize(size) {
  brushSize = Math.max(Number(brushSlider.min), Math.min(Number(brushSlider.max), size));
  brushSlider.value = brushSize;
  brushValue.textContent = brushSize;
}

function bindSlider(slider, outputId, onChange) {
  const out = $(outputId);
  const update = () => {
    out.textContent = slider.value;
    onChange(Number(slider.value));
  };
  slider.addEventListener("input", update);
  update();
}

swatches.forEach((s) => s.addEventListener("click", () => setColor(s.dataset.color)));
brushSlider.addEventListener("input", () => setBrushSize(Number(brushSlider.value)));
bindSlider(minCutoffSlider, "minCutoffValue", (v) => (minCutoff = v));
bindSlider(betaSlider, "betaValue", (v) => (beta = v));
$("clearBtn").addEventListener("click", clearDrawing);
$("undoBtn").addEventListener("click", undo);

window.addEventListener("keydown", (e) => {
  const key = e.key.toLowerCase();
  if ((e.ctrlKey || e.metaKey) && key === "z") {
    e.preventDefault();
    undo();
    return;
  }
  if (e.ctrlKey || e.metaKey || e.altKey) return;
  if (key === "c") clearDrawing();
  else if (key === "u") undo();
  else if (key === "s") skeletonToggle.checked = !skeletonToggle.checked;
  else if (key === "[") setBrushSize(brushSize - 2);
  else if (key === "]") setBrushSize(brushSize + 2);
  else if (key >= "1" && key <= String(swatches.length)) {
    setColor(swatches[Number(key) - 1].dataset.color);
  }
});

// Size the stage so it keeps the camera's aspect ratio and fits the window.
// This only changes CSS size; the canvases' pixel buffers are untouched,
// so resizing the window never erases the drawing.
function fitStage() {
  const aspect = video.videoWidth && video.videoHeight
    ? video.videoWidth / video.videoHeight
    : 4 / 3;
  const styles = getComputedStyle(stageWrap);
  const availW = stageWrap.clientWidth - parseFloat(styles.paddingLeft) - parseFloat(styles.paddingRight);
  const availH = stageWrap.clientHeight - parseFloat(styles.paddingTop) - parseFloat(styles.paddingBottom);
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

// =============================================================================
//  Geometry helpers
// =============================================================================

// Write normalized landmarks (0..1) into a reused array of canvas-pixel points.
// The video is mirrored on screen, so we mirror x here too (x -> 1 - x).
// That keeps the dots on top of your hand in the mirrored video, while
// the canvas itself stays un-mirrored so text is readable.
function toPixels(landmarks, out) {
  const w = overlay.width, h = overlay.height;
  for (let i = 0; i < 21; i++) {
    out[i].x = (1 - landmarks[i].x) * w;
    out[i].y = landmarks[i].y * h;
  }
}

function dist(a, b) {
  return Math.hypot(a.x - b.x, a.y - b.y);
}

// "Hand size" = distance from the wrist (0) to the middle-finger knuckle (9).
// This stays roughly constant for a given hand regardless of finger pose, but
// it gets bigger when the hand is close to the camera and smaller when far.
// Dividing other distances by it makes them distance-independent.
function getHandSize(pts) {
  return dist(pts[WRIST], pts[MIDDLE_MCP]) || 1;
}

// IS A FINGER EXTENDED?
// A naive check like "tip.y < pip.y" (tip above the joint) only works with the
// hand held upright, and breaks as soon as you tilt it. Distances from the
// wrist don't care about rotation:
//
//   extended:  wrist ------ PIP ------ TIP     tip much farther than PIP
//   curled:    wrist ------ PIP                tip folds back toward the
//                       TIP /                  palm, so it's CLOSER than PIP
//
// We also require the tip to be beyond the knuckle (MCP), which rejects a
// finger bent sharply at the knuckle.
// (A finger pointing straight at the camera looks short in 2D, so point it
// up/sideways rather than at the lens.)
function isFingerExtended(pts, [mcp, pip, , tip]) {
  const w = pts[WRIST];
  const dTip = dist(w, pts[tip]);
  return dTip > dist(w, pts[pip]) * EXTENDED_RATIO && dTip > dist(w, pts[mcp]) * 1.3;
}

// THE THUMB IS DIFFERENT: it bends sideways across the palm rather than
// curling toward the wrist, so its wrist distance barely changes. Instead we
// check whether the thumb tip sticks out away from the palm: it's farther
// from the pinky knuckle (17) than the thumb's own IP joint (3) is.
// (Used only by the open-palm check. DRAW/HOVER ignore the thumb, so it's
// fine whether yours is tucked in or sticking out.)
function isThumbExtended(pts) {
  return dist(pts[THUMB_TIP], pts[PINKY_MCP]) > dist(pts[THUMB_IP], pts[PINKY_MCP]) * 1.1;
}

// Map the four fingers to a mode.
function classifyPose(pts) {
  const index = isFingerExtended(pts, INDEX);
  const middle = isFingerExtended(pts, MIDDLE);
  const ring = isFingerExtended(pts, RING);
  const pinky = isFingerExtended(pts, PINKY);
  if (ring || pinky) return "IDLE";       // open palm, rock sign, etc.
  if (index && !middle) return "DRAW";    // ☝
  if (index && middle) return "HOVER";    // ✌
  return "IDLE";                          // fist, or only the middle finger
}

// Straightness of a chain of joints: straight-line length divided by the
// length along the bones. 1.0 = perfectly straight, curled fingers drop to ~0.5.
function straightness(pts, a, b, c, d) {
  const along = dist(pts[a], pts[b]) + dist(pts[b], pts[c]) + dist(pts[c], pts[d]);
  return dist(pts[a], pts[d]) / (along || 1);
}

// STRICT open palm, so a relaxed hand between letters won't trigger it:
//  - all four fingers straight, and pointing away from the wrist
//  - thumb straight and spread away from the index knuckle
//  - fingers fanned out (index tip far from pinky tip)
function isOpenPalm(pts) {
  const size = getHandSize(pts);
  for (const [mcp, pip, dip, tip] of FINGERS) {
    if (straightness(pts, mcp, pip, dip, tip) < STRAIGHT) return false;
    if (dist(pts[WRIST], pts[tip]) < dist(pts[WRIST], pts[pip]) * 1.25) return false;
  }
  const thumbStraight =
    dist(pts[THUMB_MCP], pts[THUMB_TIP]) /
    (dist(pts[THUMB_MCP], pts[THUMB_IP]) + dist(pts[THUMB_IP], pts[THUMB_TIP]) || 1);
  if (thumbStraight < STRAIGHT || !isThumbExtended(pts)) return false;
  if (dist(pts[THUMB_TIP], pts[INDEX_MCP]) < 0.6 * size) return false;
  if (dist(pts[INDEX_TIP], pts[PINKY_TIP]) < 0.85 * size) return false;
  return true;
}

// How many canvas pixels make one on-screen (CSS) pixel. The canvases use the
// camera's resolution, which can differ from what's displayed, so we scale
// line widths/radii to look the same size at any window size.
function pxScale() {
  return overlay.width / (overlay.clientWidth || overlay.width);
}

// =============================================================================
//  Overlay drawing (cleared every frame)
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
    const isTip = i === 4 || i === 8 || i === 12 || i === 16 || i === 20;
    octx.beginPath();
    octx.arc(pts[i].x, pts[i].y, (isTip ? 5 : 3.5) * s, 0, Math.PI * 2);
    octx.fill();
    octx.stroke();
  }
}

function drawLabel(pts, label, color) {
  const s = pxScale();
  const { x, y } = pts[WRIST];
  octx.font = `600 ${14 * s}px system-ui, sans-serif`;
  const w = octx.measureText(label).width + 16 * s;
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
  octx.fillText(label, x, by + h / 2);
}

// Cursor at the index fingertip:
//   DRAW  -> filled dot in the brush color
//   HOVER -> hollow ring in the brush color
//   IDLE  -> small faint grey ring (just so you can see tracking is alive)
function drawCursor(p, mode) {
  const s = pxScale();
  const r = Math.max(brushSize / 2, 6) * s;
  octx.beginPath();
  octx.lineWidth = 2 * s;
  if (mode === "DRAW") {
    octx.arc(p.x, p.y, r, 0, Math.PI * 2);
    octx.fillStyle = brushColor;
    octx.fill();
    octx.strokeStyle = "#fff";
  } else if (mode === "HOVER") {
    octx.arc(p.x, p.y, r, 0, Math.PI * 2);
    octx.strokeStyle = brushColor;
  } else {
    octx.arc(p.x, p.y, 5 * s, 0, Math.PI * 2);
    octx.strokeStyle = "rgba(160,167,185,0.5)";
  }
  octx.stroke();
}

// Big circular progress ring shown while an open palm is being held.
function drawPalmProgress(center, progress) {
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
  octx.strokeStyle = "#f87171";
  octx.beginPath();
  octx.arc(center.x, center.y, r, -Math.PI / 2, -Math.PI / 2 + progress * Math.PI * 2);
  octx.stroke();
  octx.fillStyle = "#fff";
  octx.textAlign = "center";
  octx.textBaseline = "middle";
  octx.font = `700 ${14 * s}px system-ui, sans-serif`;
  octx.fillText("CLEAR", center.x, center.y - 7 * s);
  octx.font = `500 ${11 * s}px system-ui, sans-serif`;
  octx.fillText("hold still", center.x, center.y + 9 * s);
}

// =============================================================================
//  Per-frame processing
// =============================================================================

// Pose state machine with debounce.
// The raw pose of each frame becomes the "candidate". Only once the same
// candidate has been seen POSE_FRAMES frames in a row does the mode change.
//   -> DRAW         : start a new stroke (one stroke per DRAW session, so
//                     Undo removes exactly one "pen down ... pen up")
//   DRAW -> other   : end the stroke (pen up)
function updatePose(state, pose) {
  if (pose === state.candidate) state.candidateFrames++;
  else {
    state.candidate = pose;
    state.candidateFrames = 1;
  }
  if (state.candidateFrames < POSE_FRAMES || pose === state.mode) return;

  if (state.mode === "DRAW") endStroke(state);
  state.mode = pose;
  if (pose === "DRAW") {
    if (!state.blocked) beginStroke(state);
  } else {
    state.blocked = false; // left DRAW, so Undo's block is lifted
  }
}

function updatePalmClear(state, pts, now) {
  if (!palmToggle.checked || state.mode === "DRAW" || !isOpenPalm(pts)) {
    state.palmStart = -1;
    state.palmArmed = true;
    return;
  }
  if (!state.palmArmed) return;

  const center = pts[MIDDLE_MCP];
  // Restart the timer if the palm drifts: it has to be held STILL.
  const drift = dist(center, state.palmAnchor) / getHandSize(pts);
  if (state.palmStart < 0 || drift > PALM_STILL) {
    state.palmStart = now;
    state.palmAnchor.x = center.x;
    state.palmAnchor.y = center.y;
  }
  const progress = (now - state.palmStart) / PALM_HOLD_MS;
  if (progress >= 1) {
    clearDrawing();
    state.palmArmed = false; // no repeat clears until the palm closes
    state.palmStart = -1;
  } else {
    drawPalmProgress(center, progress);
  }
}

const seen = new Set(); // reused every frame

// The on-screen badge shows the "busiest" hand: DRAW beats HOVER beats IDLE.
const MODE_RANK = { NONE: 0, IDLE: 1, HOVER: 2, DRAW: 3 };
let shownMode = "";
function setModeBadge(mode) {
  if (mode === shownMode) return; // only touch the DOM when it changes
  shownMode = mode;
  modeBadge.textContent = mode === "NONE" ? "NO HAND" : mode;
  modeBadge.dataset.mode = mode;
}

function processResults(results, now) {
  octx.clearRect(0, 0, overlay.width, overlay.height); // overlay ONLY
  let topMode = "NONE";

  // Newer versions call it `handedness`, older ones `handednesses`.
  const handedness = results.handedness || results.handednesses || [];
  const tSec = now / 1000;
  seen.clear();

  for (let i = 0; i < results.landmarks.length; i++) {
    // --- Which hand is this? ---
    let label = handedness[i]?.[0]?.categoryName || "Right";
    if (SWAP_HANDEDNESS) label = label === "Left" ? "Right" : "Left";
    // If both hands get the same label (can happen briefly), keep them apart.
    if (seen.has(label)) label = label === "Left" ? "Right" : "Left";
    seen.add(label);

    const state = getHandState(label);
    state.missing = 0;
    const pts = state.pts;
    toPixels(results.landmarks[i], pts);

    // --- One Euro filter on the index fingertip ---
    state.tip.x = state.fx.filter(pts[INDEX_TIP].x, tSec);
    state.tip.y = state.fy.filter(pts[INDEX_TIP].y, tSec);

    // --- Finger count -> DRAW / HOVER / IDLE ---
    updatePose(state, classifyPose(pts));
    if (state.mode === "DRAW" && state.stroke) addPoint(state.stroke, state.tip.x, state.tip.y);

    // --- Optional open-palm clear ---
    updatePalmClear(state, pts, now);

    // --- Overlay ---
    const color = HAND_COLORS[label];
    if (skeletonToggle.checked) drawSkeleton(pts, color);
    drawLabel(pts, `${label} · ${state.mode}`, color);
    drawCursor(state.tip, state.mode);
    if (MODE_RANK[state.mode] > MODE_RANK[topMode]) topMode = state.mode;
  }
  setModeBadge(topMode);

  // Hands not seen this frame: give them a short grace period (the stroke
  // stays open and the gap gets interpolated), then lift the pen and forget.
  for (const label in hands) {
    if (seen.has(label)) continue;
    const state = hands[label];
    if (++state.missing > LOST_GRACE_FRAMES) {
      endStroke(state);
      delete hands[label];
    }
  }
}

function updateStats(now, ms) {
  const dt = now - lastFrameTime;
  lastFrameTime = now;
  if (dt > 0) {
    // Exponential moving averages so the numbers don't jump around.
    fps = fps ? fps * 0.9 + (1000 / dt) * 0.1 : 1000 / dt;
    inferMs = inferMs ? inferMs * 0.9 + ms * 0.1 : ms;
    fpsEl.textContent = `${Math.round(fps)} FPS`;
    inferEl.textContent = `${inferMs.toFixed(1)} ms`;
  }
}

// The main loop. requestAnimationFrame fires before every screen repaint
// (often 60-144 Hz), but webcams deliver ~30 frames/s. Checking
// video.currentTime means we only run the model on a NEW camera frame, and
// skip repaints where nothing changed.
async function loop() {
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
        console.warn("[AirDraw] GPU inference failed, switching to CPU:", err);
        handLandmarker.close();
        handLandmarker = await createLandmarker("CPU");
        requestAnimationFrame(loop);
        return;
      }
      throw err;
    }
    const ms = performance.now() - now;
    processResults(results, now);
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
  delegateEl.textContent = delegate === "GPU" && software ? "GPU (software!)" : delegate;
  delegateEl.title = `WebGL renderer: ${renderer}`;
  console.info(`[AirDraw] HandLandmarker active delegate: ${delegate}`);
  if (delegate === "GPU") {
    console.info(`[AirDraw] WebGL renderer: ${renderer}`);
    if (software) {
      console.warn("[AirDraw] WebGL is using a software renderer. Enable hardware acceleration in browser settings.");
    }
  }
  return landmarker;
}

async function loadModel() {
  try {
    return await createLandmarker("GPU");
  } catch (err) {
    console.warn("[AirDraw] GPU delegate failed, falling back to CPU:", err);
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
  resEl.textContent = `${video.videoWidth}×${video.videoHeight}`;
  console.info(`[AirDraw] Camera resolution: ${video.videoWidth}x${video.videoHeight}`);
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

async function init() {
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
  requestAnimationFrame(loop);
}

init();
