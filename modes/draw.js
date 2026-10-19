// =============================================================================
//  modes/draw.js: DRAW mode, air drawing with the index finger
// =============================================================================
//
//  ☝  index only      -> DRAW  (pen down, fingertip draws)
//  ✌  index + middle  -> HOVER (cursor follows, pen up)
//     anything else   -> IDLE  (fist, open palm, ...)
//
//  Strokes live on #drawCanvas, a separate layer that is NEVER cleared per
//  frame. It's only wiped by Clear, or redrawn from `strokes` after Undo.
//
//  Every mode exports the same interface, called by main.js:
//    init(app)            once at startup (app = shared canvases/helpers)
//    enter() / exit()     when the mode is switched on/off
//    update(hands, now)   logic for this frame (hands = visible hands)
//    render(ctx, hands)   draw this mode's overlay (cursor, rings...)
//    handLost(label)      a hand left the frame for good
//    trailColor(hand)     particle color for this hand, or null = no trail
//    labelSuffix(hand)    text shown after "Left"/"Right"
//    badge()              { text, tone } for the top-left status badge
//    onKey(e)             return true if the key was handled
//    onKeyUp(e)           (optional) a key was released
// =============================================================================

import { MIDDLE_MCP, dist, handSize, isOpenPalm, classifyDrawPose, Debouncer } from "../gestures.js";

const CONFIG = {
  poseFrames: 3,       // a pose must be seen this many frames in a row
  jump: 0.04,          // fingertip jump (fraction of width) that triggers interpolation
  step: 0.015,         // spacing of interpolated points (fraction of width)
  palmHoldMs: 1500,    // optional open-palm clear: hold time
  palmStill: 0.25,     // ...and max palm drift during the hold, in hand sizes
};

const RANK = { NONE: 0, IDLE: 1, HOVER: 2, DRAW: 3 };

let app = null;
let dctx = null;

// Every stroke: { color: "#22d3ee", size: 8, points: [{x, y}, ...] }
const strokes = [];
// Per-hand drawing state, keyed by "Left"/"Right".
const states = {};

let brushColor = "#22d3ee";
let brushSize = 8;
let topMode = "NONE";

let brushSlider, brushValue, palmToggle, swatches;

function getState(label) {
  if (!states[label]) {
    states[label] = {
      hand: null,                                   // latest tracked hand object
      pose: new Debouncer(CONFIG.poseFrames, "IDLE"),
      mode: "IDLE",                                 // confirmed DRAW/HOVER/IDLE
      stroke: null,                                 // stroke being drawn
      blocked: false,                               // after Undo, until leaving DRAW
      palmStart: -1,
      palmAnchor: { x: 0, y: 0 },
      palmArmed: true,
      palmProgress: -1,                             // 0..1 while holding, -1 = no ring
    };
  }
  return states[label];
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
  dctx.clearRect(0, 0, app.drawCanvas.width, app.drawCanvas.height);
  for (const s of strokes) renderStroke(dctx, s);
}

// Append a point to a stroke and draw just the new piece (incremental, cheap).
// If the fingertip jumped far since the last point (fast move or a few lost
// frames), insert evenly spaced points in between so the line isn't broken.
function addPoint(s, x, y) {
  const last = s.points[s.points.length - 1];
  const d = Math.hypot(x - last.x, y - last.y);
  if (d < 0.75) return; // ignore sub-pixel noise

  const w = app.drawCanvas.width;
  if (d > CONFIG.jump * w) {
    const extra = Math.floor(d / (CONFIG.step * w));
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
  const tip = state.hand.tip;
  const s = {
    color: brushColor,
    size: brushSize * app.pxScale(), // stored in canvas pixels
    points: [{ x: tip.x, y: tip.y }],
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
  dctx.clearRect(0, 0, app.drawCanvas.width, app.drawCanvas.height);
  for (const label in states) {
    const s = states[label];
    s.stroke = null;
    // A hand that's still in DRAW mode carries on with a fresh stroke.
    if (s.mode === "DRAW" && !s.blocked && s.hand) beginStroke(s);
  }
}

function undo() {
  const removed = strokes.pop();
  if (!removed) return;
  // If a hand was drawing the stroke we just removed, lift its pen so it
  // doesn't keep drawing into a deleted stroke. It has to leave DRAW pose
  // (e.g. go to HOVER) and come back to start drawing again.
  for (const label in states) {
    if (states[label].stroke === removed) {
      states[label].stroke = null;
      states[label].blocked = true;
    }
  }
  redrawAll();
}

// =============================================================================
//  Gestures
// =============================================================================

// Pose state machine. The Debouncer only reports a new pose once it has been
// seen CONFIG.poseFrames frames in a row, then:
//   -> DRAW         : start a new stroke (one stroke per DRAW session, so
//                     Undo removes exactly one "pen down ... pen up")
//   DRAW -> other   : end the stroke (pen up)
function updatePose(state, hand) {
  const pose = state.pose.update(classifyDrawPose(hand.fingers));
  if (pose === state.mode) return;
  if (state.mode === "DRAW") endStroke(state);
  state.mode = pose;
  if (pose === "DRAW") {
    if (!state.blocked) beginStroke(state);
  } else {
    state.blocked = false; // left DRAW, so Undo's block is lifted
  }
}

// Optional (off by default): strict open palm held STILL clears the canvas.
function updatePalmClear(state, hand, now) {
  state.palmProgress = -1;
  if (!palmToggle.checked || state.mode === "DRAW" || !isOpenPalm(hand.pts)) {
    state.palmStart = -1;
    state.palmArmed = true;
    return;
  }
  if (!state.palmArmed) return;

  const center = hand.pts[MIDDLE_MCP];
  // Restart the timer if the palm drifts: it has to be held still.
  const drift = dist(center, state.palmAnchor) / handSize(hand.pts);
  if (state.palmStart < 0 || drift > CONFIG.palmStill) {
    state.palmStart = now;
    state.palmAnchor.x = center.x;
    state.palmAnchor.y = center.y;
  }
  const progress = (now - state.palmStart) / CONFIG.palmHoldMs;
  if (progress >= 1) {
    clearDrawing();
    state.palmArmed = false; // no repeat clears until the palm closes
    state.palmStart = -1;
  } else {
    state.palmProgress = progress;
  }
}

// =============================================================================
//  Overlay drawing
// =============================================================================

// Cursor at the index fingertip:
//   DRAW  -> filled dot in the brush color
//   HOVER -> hollow ring in the brush color
//   IDLE  -> small faint grey ring (just so you can see tracking is alive)
function drawCursor(ctx, p, mode) {
  const s = app.pxScale();
  const r = Math.max(brushSize / 2, 6) * s;
  ctx.beginPath();
  ctx.lineWidth = 2 * s;
  if (mode === "DRAW") {
    ctx.arc(p.x, p.y, r, 0, Math.PI * 2);
    ctx.fillStyle = brushColor;
    ctx.fill();
    ctx.strokeStyle = "#fff";
  } else if (mode === "HOVER") {
    ctx.arc(p.x, p.y, r, 0, Math.PI * 2);
    ctx.strokeStyle = brushColor;
  } else {
    ctx.arc(p.x, p.y, 5 * s, 0, Math.PI * 2);
    ctx.strokeStyle = "rgba(160,167,185,0.5)";
  }
  ctx.stroke();
}

// =============================================================================
//  UI
// =============================================================================

function setColor(color) {
  brushColor = color;
  swatches.forEach((s) => s.classList.toggle("active", s.dataset.color === color));
}

function setBrushSize(size) {
  brushSize = Math.max(Number(brushSlider.min), Math.min(Number(brushSlider.max), size));
  brushSlider.value = brushSize;
  brushValue.textContent = brushSize;
}

// =============================================================================
//  Mode interface
// =============================================================================

export const drawMode = {
  id: "draw",
  name: "Draw",

  init(ctx) {
    app = ctx;
    dctx = app.drawCanvas.getContext("2d");
    brushSlider = document.getElementById("brushSize");
    brushValue = document.getElementById("brushValue");
    palmToggle = document.getElementById("palmClear");
    swatches = [...document.querySelectorAll(".swatch")];

    brushColor = swatches[0].dataset.color;
    brushSize = Number(brushSlider.value);
    swatches.forEach((s) => s.addEventListener("click", () => setColor(s.dataset.color)));
    brushSlider.addEventListener("input", () => setBrushSize(Number(brushSlider.value)));
    document.getElementById("clearBtn").addEventListener("click", clearDrawing);
    document.getElementById("undoBtn").addEventListener("click", undo);
  },

  enter() {},

  // Leaving Draw mode lifts every pen, so switching back requires a fresh pose.
  exit() {
    for (const label in states) {
      const s = states[label];
      endStroke(s);
      s.mode = "IDLE";
      s.pose.reset("IDLE");
      s.blocked = false;
      s.palmStart = -1;
      s.palmProgress = -1;
    }
    topMode = "NONE";
  },

  update(hands, now) {
    topMode = "NONE";
    for (const hand of hands) {
      const state = getState(hand.label);
      state.hand = hand;
      updatePose(state, hand);
      if (state.mode === "DRAW" && state.stroke) addPoint(state.stroke, hand.tip.x, hand.tip.y);
      updatePalmClear(state, hand, now);
      if (RANK[state.mode] > RANK[topMode]) topMode = state.mode;
    }
  },

  render(ctx, hands) {
    for (const hand of hands) {
      const state = states[hand.label];
      if (!state) continue;
      if (state.palmProgress >= 0) app.drawHoldRing(hand.pts[MIDDLE_MCP], state.palmProgress, "CLEAR", "hold still", "#f87171");
      drawCursor(ctx, hand.tip, state.mode);
    }
  },

  // A hand left the frame for good: lift its pen and forget it.
  handLost(label) {
    if (states[label]) endStroke(states[label]);
    delete states[label];
  },

  // Trails only while actually drawing, in the brush color.
  trailColor(hand) {
    return states[hand.label]?.mode === "DRAW" ? brushColor : null;
  },

  labelSuffix(hand) {
    return states[hand.label]?.mode || "IDLE";
  },

  badge() {
    return { text: topMode === "NONE" ? "NO HAND" : topMode, tone: topMode };
  },

  onKey(e) {
    const key = e.key.toLowerCase();
    if ((e.ctrlKey || e.metaKey) && key === "z") {
      e.preventDefault();
      undo();
      return true;
    }
    if (e.ctrlKey || e.metaKey || e.altKey) return false;
    // Shift+1..6 picks a color (plain 1/2/3 switch modes). e.code is the
    // physical key, so it works even though Shift+1 types "!".
    if (e.shiftKey && /^Digit[1-9]$/.test(e.code)) {
      const n = Number(e.code.slice(5));
      if (n <= swatches.length) setColor(swatches[n - 1].dataset.color);
      return true;
    }
    if (key === "c") clearDrawing();
    else if (key === "u") undo();
    else if (key === "[") setBrushSize(brushSize - 2);
    else if (key === "]") setBrushSize(brushSize + 2);
    else return false;
    return true;
  },
};
