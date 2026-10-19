// =============================================================================
//  sign/motion.js: the two MOVING letters, J and Z
// =============================================================================
//
//    J = the I handshape (little finger up), and the LITTLE FINGERTIP draws a
//        hook: down, then curving sideways.
//    Z = pointing (index finger only), and the INDEX FINGERTIP draws a Z:
//        across, diagonally down and back, across again.
//
//  HOW: a rolling buffer keeps the last ~2 s of fingertip positions while
//  the matching handshape is held. Every frame we simplify that path into a
//  few straight strokes and check the stroke pattern.
//
//  1. SIMPLIFY with Ramer-Douglas-Peucker (RDP): keep only the points where
//     the path really changes direction (anything closer than `rdpEpsilon`
//     hand sizes to a straight line is dropped). Hand jitter disappears, and
//     a Z becomes ~4 corner points no matter how fast or slow it was drawn.
//  2. MERGE neighbouring strokes that point the same way (within
//     `mergeAngle`), so a wobbly stroke still counts as one.
//  3. MATCH the most recent strokes against the letter's pattern (see
//     matchZ / matchJ). Everything is measured in HAND SIZES, so it works
//     near or far from the camera.
//
//  WHY PATH CHECKS AND NOT DTW (dynamic time warping)?
//  DTW compares the motion to recorded templates. It's flexible, but only as
//  good as its templates (ideally per person), and it needs the start/end of
//  the gesture cut out. J and Z have very distinctive shapes, so rules
//  written from the letter's definition work for anyone without recording
//  templates, ignore speed (RDP drops timing), and every threshold can be
//  read and tuned. Trade-off: very sloppy or unusual versions may fail a
//  strict check, where DTW could learn someone's personal style.
//
//  BOTH HANDS / MIRRORING: Z is accepted in either horizontal direction
//  (first and last strokes the same way, middle stroke back the other way),
//  and the J hook in either sideways direction. So it works for left and
//  right hands without trusting the Left/Right label. Up/down is never
//  mirrored, so "down" stays strict.
//
//  WHILE MOVING: `inProgress` is true while the handshape is held and the
//  fingertip is moving (and briefly after). Sign mode uses it to stop the
//  STATIC letters I (same handshape as J) and pointing/D (same as Z) from
//  being typed halfway through a J or Z.
// =============================================================================

import { INDEX_TIP, PINKY_TIP, WRIST, MIDDLE_MCP, dist } from "../gestures.js";

export const MOTION_CFG = {
  windowMs: 2000,         // how much fingertip history to keep
  shapeGraceMs: 200,      // the handshape may flicker this long without resetting the path
  rdpEpsilon: 0.1,        // RDP tolerance, hand sizes
  mergeAngle: 35,         // merge consecutive strokes whose directions differ less than this (deg)
  movingSpeed: 0.8,       // hand sizes / second: faster than this = "drawing"
  movingWindowMs: 250,    // speed is measured over this window
  movingHoldMs: 450,      // stay "in progress" this long after the motion stops
  cooldownMs: 800,        // after recognising J/Z, ignore new motion for this long

  // Z (index fingertip)
  zMinStroke: 0.45,       // each of the three strokes at least this long
  zFlatRatio: 1.2,        // strokes 1 and 3 are "across": |dx| > ratio x |dy|
  zMinDrop: 0.25,         // stroke 2 must go DOWN at least this much

  // J (little fingertip)
  jMinDown: 0.3,          // downward stroke before the hook
  jMinHook: 0.18,         // sideways travel of the hook
  jMinTurn: 40,           // direction change between the down stroke and the hook (deg)
};

// Handshape checks use the fuzzy finger states from sign/rules.js
// (f.ext[i]: 0 index, 1 middle, 2 ring, 3 pinky).
export function isIShape(f) {
  return f.ext[3] > 0.5 && f.ext[0] < 0.5 && f.ext[1] < 0.5 && f.ext[2] < 0.5;
}
export function isPointShape(f) {
  return f.ext[0] > 0.5 && f.ext[1] < 0.5 && f.ext[2] < 0.5 && f.ext[3] < 0.5;
}

// ---- Geometry helpers ----

// Perpendicular distance from p to the line a-b.
function lineDist(p, a, b) {
  const dx = b.x - a.x, dy = b.y - a.y;
  const len = Math.hypot(dx, dy);
  if (!len) return Math.hypot(p.x - a.x, p.y - a.y);
  return Math.abs(dy * p.x - dx * p.y + b.x * a.y - b.y * a.x) / len;
}

// Ramer-Douglas-Peucker: keep the point farthest from the a-b line if it's
// more than eps away, and recurse on both halves.
export function rdp(points, eps) {
  if (points.length < 3) return points.slice();
  let maxD = 0, idx = 0;
  const a = points[0], b = points[points.length - 1];
  for (let i = 1; i < points.length - 1; i++) {
    const d = lineDist(points[i], a, b);
    if (d > maxD) { maxD = d; idx = i; }
  }
  if (maxD <= eps) return [a, b];
  const left = rdp(points.slice(0, idx + 1), eps);
  const right = rdp(points.slice(idx), eps);
  return left.slice(0, -1).concat(right);
}

const angleOf = (s) => Math.atan2(s.dy, s.dx) * 180 / Math.PI;
function angleDiff(a, b) {
  let d = Math.abs(a - b) % 360;
  return d > 180 ? 360 - d : d;
}

// Simplified points -> strokes { dx, dy, len }, merging same-direction neighbours.
export function strokes(points) {
  const out = [];
  for (let i = 1; i < points.length; i++) {
    const s = { dx: points[i].x - points[i - 1].x, dy: points[i].y - points[i - 1].y };
    s.len = Math.hypot(s.dx, s.dy);
    if (s.len < 1e-6) continue;
    const prev = out[out.length - 1];
    if (prev && angleDiff(angleOf(prev), angleOf(s)) < MOTION_CFG.mergeAngle) {
      prev.dx += s.dx; prev.dy += s.dy; prev.len = Math.hypot(prev.dx, prev.dy);
    } else out.push(s);
  }
  return out;
}

// ---- Pattern matchers (screen coordinates: +x right, +y DOWN; units: hand sizes) ----

// Z: the LAST three strokes are across / diagonally down-and-back / across.
export function matchZ(st) {
  const c = MOTION_CFG;
  if (st.length < 3) return { ok: false, why: "Z needs 3 strokes" };
  const [s1, s2, s3] = st.slice(-3);
  const across = (s) => Math.abs(s.dx) > c.zFlatRatio * Math.abs(s.dy);
  if (!(across(s1) && s1.len >= c.zMinStroke)) return { ok: false, why: "1st stroke not a long sideways line" };
  if (!(Math.sign(s2.dx) === -Math.sign(s1.dx) && s2.dy >= c.zMinDrop && s2.len >= c.zMinStroke)) return { ok: false, why: "2nd stroke not diagonally down and back" };
  if (!(across(s3) && Math.sign(s3.dx) === Math.sign(s1.dx) && s3.len >= c.zMinStroke)) return { ok: false, why: "3rd stroke not across in the 1st stroke's direction" };
  return { ok: true, why: "Z" };
}

// J: a downward stroke, then (as the last stroke or two) a sideways hook.
export function matchJ(st) {
  const c = MOTION_CFG;
  if (st.length < 2) return { ok: false, why: "J needs a down stroke + a hook" };
  for (let i = st.length - 2; i >= Math.max(0, st.length - 3); i--) {
    const down = st[i];
    if (!(down.dy >= c.jMinDown && down.dy >= Math.abs(down.dx))) continue;
    // The hook: everything after the down stroke, added together.
    let hx = 0, hy = 0;
    for (let k = i + 1; k < st.length; k++) { hx += st[k].dx; hy += st[k].dy; }
    const hook = { dx: hx, dy: hy };
    const turn = angleDiff(angleOf(down), angleOf(hook));
    if (Math.abs(hx) >= c.jMinHook && turn >= c.jMinTurn && hy > -c.jMinDown) return { ok: true, why: "J" };
    return { ok: false, why: `hook too small (sideways ${Math.abs(hx).toFixed(2)}, turn ${Math.round(turn)}°)` };
  }
  return { ok: false, why: "no clear downward stroke" };
}

// ---- The tracker (one per Sign mode) ----

const CAP = 160; // ring buffer size (> 2 s at 60 fps)

export class MotionTracker {
  constructor() {
    this.buf = Array.from({ length: CAP }, () => ({ x: 0, y: 0, t: 0, size: 1 }));
    this.start = 0;
    this.count = 0;
    this.shape = "";        // "I" (for J) | "point" (for Z) | ""
    this.lastShapeAt = 0;
    this.lastMovingAt = -1e9;
    this.cooldownUntil = 0;
    this.inProgress = false;
    this.debug = "";
  }

  reset() {
    this.count = 0;
    this.start = 0;
    this.shape = "";
    this.inProgress = false;
  }

  at(i) { return this.buf[(this.start + i) % CAP]; }

  // Add a point. Slots are reused (no allocation); when full, the oldest
  // point (at `start`) is overwritten and becomes the newest.
  push(x, y, t, size) {
    let slot;
    if (this.count < CAP) {
      slot = this.buf[(this.start + this.count) % CAP];
      this.count++;
    } else {
      slot = this.buf[this.start];
      this.start = (this.start + 1) % CAP;
    }
    slot.x = x; slot.y = y; slot.t = t; slot.size = size;
  }

  // The path as plain points in hand sizes, relative to the first point.
  path() {
    const out = [];
    if (!this.count) return out;
    const o = this.at(0);
    let size = 0;
    for (let i = 0; i < this.count; i++) size += this.at(i).size;
    size = size / this.count || 1;
    for (let i = 0; i < this.count; i++) {
      const p = this.at(i);
      out.push({ x: (p.x - o.x) / size, y: (p.y - o.y) / size });
    }
    return out;
  }

  // Call every frame with the main hand (or null) and its feature object
  // (finger states filled in). Returns "J", "Z" or "".
  update(hand, f, now) {
    const c = MOTION_CFG;
    const shape = !hand ? "" : isIShape(f) ? "I" : isPointShape(f) ? "point" : "";

    if (shape) {
      if (shape !== this.shape) { this.count = 0; this.start = 0; this.shape = shape; }
      this.lastShapeAt = now;
      const tip = hand.pts[shape === "I" ? PINKY_TIP : INDEX_TIP];
      this.push(tip.x, tip.y, now, dist(hand.pts[WRIST], hand.pts[MIDDLE_MCP]) || 1);
    } else if (now - this.lastShapeAt > c.shapeGraceMs) {
      this.reset();
    }
    // Forget points older than the window.
    while (this.count && now - this.at(0).t > c.windowMs) { this.start = (this.start + 1) % CAP; this.count--; }

    // "Drawing" = recent fingertip speed above the threshold.
    let speed = 0;
    if (this.count >= 2) {
      const last = this.at(this.count - 1);
      let j = this.count - 1;
      while (j > 0 && last.t - this.at(j - 1).t <= c.movingWindowMs) j--;
      const first = this.at(j);
      const dt = (last.t - first.t) / 1000;
      if (dt > 0) speed = Math.hypot(last.x - first.x, last.y - first.y) / last.size / dt;
    }
    if (this.shape && speed > c.movingSpeed) this.lastMovingAt = now;
    this.inProgress = !!this.shape && now - this.lastMovingAt < c.movingHoldMs;

    if (!this.shape || this.count < 6) {
      this.debug = this.shape ? "collecting…" : "no J/Z handshape";
      return "";
    }
    if (now < this.cooldownUntil) {
      this.debug = "cooldown";
      return "";
    }
    const st = strokes(rdp(this.path(), c.rdpEpsilon));
    const m = this.shape === "point" ? matchZ(st) : matchJ(st);
    this.debug = `${this.shape === "point" ? "Z" : "J"}: ${st.length} stroke(s) · ${m.why}`;
    if (!m.ok) return "";
    this.cooldownUntil = now + c.cooldownMs;
    this.count = 0;
    this.start = 0;
    return this.shape === "point" ? "Z" : "J";
  }

  // Draw the tracked path (for feedback while drawing).
  draw(ctx, color, scale) {
    if (this.count < 2 || !this.shape) return;
    ctx.save();
    ctx.lineCap = "round";
    ctx.lineJoin = "round";
    const newest = this.at(this.count - 1).t;
    for (let i = 1; i < this.count; i++) {
      const a = this.at(i - 1), b = this.at(i);
      ctx.globalAlpha = Math.max(0.1, 1 - (newest - b.t) / MOTION_CFG.windowMs);
      ctx.strokeStyle = color;
      ctx.lineWidth = 4 * scale;
      ctx.beginPath();
      ctx.moveTo(a.x, a.y);
      ctx.lineTo(b.x, b.y);
      ctx.stroke();
    }
    ctx.restore();
  }
}
