// =============================================================================
//  sign/features.js: measurements + preprocessing shared by EVERYTHING in
//  Sign mode (the rules, data recording, and the learned models)
// =============================================================================
//
//  Two levels:
//
//  1. computeFeatures(pts, f): ~20 hand-designed MEASUREMENTS (how extended
//     each finger is, how bent, where the thumb is, gaps, spreads,
//     orientation). The rule-based classifier (rules.js) reads these.
//
//  2. toVector(pts, handLabel): the PREPROCESSED feature vector the learned
//     models use. The exact same function runs when recording/training and
//     when predicting live. If they differed, the model would be trained on
//     one kind of input and tested on another ("training/serving skew").
//
//       a) translate: wrist moved to the origin      -> position-invariant
//       b) scale: divide by hand size (wrist->middle knuckle) -> distance-invariant
//       c) mirror: LEFT hands are flipped (x -> -x) into right-hand
//          coordinates, so one model serves both hands
//       d) NO rotation normalisation: G/H point sideways and P/Q point down,
//          so orientation is real information we must keep
//       e) 21 landmarks x (x, y, z) = 63 numbers, plus 22 engineered
//          measurements from step 1 = 85 numbers in total
//
//  Recorded samples store the RAW landmarks (see packLandmarks), not this
//  vector. Preprocessing runs at training time, so if it ever changes, old
//  recordings (and friends' recordings) stay usable.
// =============================================================================

import {
  WRIST, THUMB_MCP, THUMB_TIP, INDEX_MCP, INDEX_PIP, INDEX_TIP, MIDDLE_MCP,
  MIDDLE_PIP, MIDDLE_TIP, RING_TIP, PINKY_MCP, PINKY_TIP, FINGERS,
  dist, extensionRatio, thumbOutRatio,
} from "../gestures.js";

// =============================================================================
//  Geometry helpers
// =============================================================================

// How bent a finger is, in degrees: the angle between its first bone
// (MCP -> PIP) and the line from the PIP joint to the fingertip.
//   0° = dead straight, ~60° = C curve, ~90° = O, ~120°+ = hooked / fist
// Uses x, y AND z, so a finger bending toward the camera still counts.
//
// Why not add up the PIP and DIP joint angles? Those come from the two
// SHORTEST bones (~0.2 hand sizes), so a little landmark jitter (worst in
// z) makes a big angle error, and summing two angles adds two errors. A
// noise stress test showed straight fingers being read as bent. PIP -> tip
// is more than twice as long and gives one angle, so it's much steadier.
function fingerBend(mcp, pip, tip) {
  const ux = pip.x - mcp.x, uy = pip.y - mcp.y, uz = pip.z - mcp.z;
  const vx = tip.x - pip.x, vy = tip.y - pip.y, vz = tip.z - pip.z;
  const lu = Math.hypot(ux, uy, uz), lv = Math.hypot(vx, vy, vz);
  if (!lu || !lv) return 0;
  const cos = Math.max(-1, Math.min(1, (ux * vx + uy * vy + uz * vz) / (lu * lv)));
  return (Math.acos(cos) * 180) / Math.PI;
}

// Direction of a->b relative to the SCREEN, as an unsigned angle from
// straight up: 0° = up, 90° = sideways (either way), 180° = down.
// Unsigned makes it mirror-safe (left/right hands both read 90° sideways).
function tiltFromUp(a, b) {
  const dx = b.x - a.x, dy = b.y - a.y;
  const l = Math.hypot(dx, dy);
  if (!l) return 0;
  return (Math.acos(Math.max(-1, Math.min(1, -dy / l))) * 180) / Math.PI; // screen y grows downward
}

// Angle in degrees between two bones (a1->a2) and (b1->b2), in the image plane.
function angleBetween(a1, a2, b1, b2) {
  const ax = a2.x - a1.x, ay = a2.y - a1.y;
  const bx = b2.x - b1.x, by = b2.y - b1.y;
  const l = Math.hypot(ax, ay) * Math.hypot(bx, by);
  if (!l) return 0;
  return (Math.acos(Math.max(-1, Math.min(1, (ax * bx + ay * by) / l))) * 180) / Math.PI;
}

// =============================================================================
//  1. Measurements
// =============================================================================

export function makeFeatures() {
  return {
    ratio: [0, 0, 0, 0],     // extension ratio per finger (index..pinky), see gestures.js
    bend: [0, 0, 0, 0],      // bend per finger, degrees (see fingerBend)
    // Fuzzy finger states, filled in by rules.js (fingerStates):
    ext: [0, 0, 0, 0],       // "extended"
    curl: [0, 0, 0, 0],      // "curled"
    curve: [0, 0, 0, 0],     // "curved" (C / O)
    thumbOut: 0,             // thumbOutRatio
    thumbSide: 0,            // thumb tip position across the palm, hand sizes
    gapTI: 0,                // thumb tip <-> index tip, hand sizes
    gapTM: 0,                // thumb tip <-> middle tip
    tipGap: 0,               // mean gap between neighbouring fingertips
    spreadIM: 0,             // angle index vs middle, degrees
    spreadIR: 0,             // angle index vs ring
    angleTI: 0,              // angle thumb vs index
    crossIM: 0,              // index tip minus middle tip across the palm (< 0 = crossed, R)
    gapTMp: 0,               // thumb tip <-> middle PIP joint (K / P), hand sizes
    indexReach: 0,           // wrist->index PIP / wrist->index MCP (X: knuckle straight)
    indexTilt: 0,            // index direction vs screen-up, degrees (0 up, 90 side, 180 down)
    handTilt: 0,             // wrist->middle knuckle direction vs screen-up
    palmFacing: 0,           // 1 = palm flat to the camera, 0 = edge-on
  };
}

export function computeFeatures(pts, f) {
  const w = pts[WRIST];
  const size = dist(w, pts[MIDDLE_MCP]) || 1;

  // --- Fingers ---
  for (let i = 0; i < 4; i++) {
    const [mcp, pip, , tip] = FINGERS[i];
    f.ratio[i] = extensionRatio(pts, FINGERS[i]);
    f.bend[i] = fingerBend(pts[mcp], pts[pip], pts[tip]);
  }

  // --- Hand-local side axis: pinky knuckle -> index knuckle, with the "up"
  // (wrist -> middle knuckle) component removed. For a left hand, a right
  // hand, mirrored or not, "toward the index side" is always +side.
  const upx = (pts[MIDDLE_MCP].x - w.x) / size, upy = (pts[MIDDLE_MCP].y - w.y) / size;
  let sx = pts[INDEX_MCP].x - pts[PINKY_MCP].x, sy = pts[INDEX_MCP].y - pts[PINKY_MCP].y;
  const along = sx * upx + sy * upy;
  sx -= along * upx;
  sy -= along * upy;
  const sl = Math.hypot(sx, sy) || 1;
  sx /= sl;
  sy /= sl;
  const sideOf = (p) => ((p.x - w.x) * sx + (p.y - w.y) * sy) / size;
  // > 0: thumb tip is beyond the index knuckle (beside the hand, as in A)
  // < 0: thumb tip is across the palm/fingers (tucked, as in B or a fist)
  f.thumbSide = sideOf(pts[THUMB_TIP]) - sideOf(pts[INDEX_MCP]);
  f.thumbOut = thumbOutRatio(pts);

  // --- Gaps and angles ---
  f.gapTI = dist(pts[THUMB_TIP], pts[INDEX_TIP]) / size;
  f.gapTM = dist(pts[THUMB_TIP], pts[MIDDLE_TIP]) / size;
  f.tipGap = (dist(pts[INDEX_TIP], pts[MIDDLE_TIP]) + dist(pts[MIDDLE_TIP], pts[RING_TIP]) +
    dist(pts[RING_TIP], pts[PINKY_TIP])) / 3 / size;
  f.spreadIM = angleBetween(pts[INDEX_MCP], pts[INDEX_TIP], pts[MIDDLE_MCP], pts[MIDDLE_TIP]);
  f.spreadIR = angleBetween(pts[INDEX_MCP], pts[INDEX_TIP], pts[13], pts[RING_TIP]);
  f.angleTI = angleBetween(pts[THUMB_MCP], pts[THUMB_TIP], pts[INDEX_MCP], pts[INDEX_TIP]);

  // Crossing (R): normally the index tip is on the index side of the middle
  // tip (crossIM > 0). Crossed fingers swap that order, so it goes negative.
  f.crossIM = sideOf(pts[INDEX_TIP]) - sideOf(pts[MIDDLE_TIP]);
  // K / P: the thumb tip pokes up between index and middle, near the middle PIP.
  f.gapTMp = dist(pts[THUMB_TIP], pts[MIDDLE_PIP]) / size;
  // X: if the index knuckle (MCP) is straight, the PIP joint stays far from
  // the wrist even though the finger is hooked. In a fist the knuckle bends
  // too, pulling the PIP joint in.
  f.indexReach = dist(w, pts[INDEX_PIP]) / (dist(w, pts[INDEX_MCP]) || 1);

  // --- Orientation relative to the screen ---
  f.indexTilt = tiltFromUp(pts[INDEX_MCP], pts[INDEX_TIP]);
  f.handTilt = tiltFromUp(w, pts[MIDDLE_MCP]);
  // Palm normal = cross product of two palm edges (wrist->index knuckle,
  // wrist->pinky knuckle). Its z part says how directly the palm faces the
  // camera. We use the size only (not the sign): the sign flips between left
  // and right hands, and we don't want to trust the Left/Right label here.
  const ax = pts[INDEX_MCP].x - w.x, ay = pts[INDEX_MCP].y - w.y, az = pts[INDEX_MCP].z - w.z;
  const bx = pts[PINKY_MCP].x - w.x, by = pts[PINKY_MCP].y - w.y, bz = pts[PINKY_MCP].z - w.z;
  const nx = ay * bz - az * by, ny = az * bx - ax * bz, nz = ax * by - ay * bx;
  f.palmFacing = Math.abs(nz) / (Math.hypot(nx, ny, nz) || 1);
  return f;
}

// =============================================================================
//  2. Preprocessed feature vector for the learned models
// =============================================================================

// Names of the 22 engineered values appended after the 63 coordinates.
// Angles are divided by 90 or 180 so every value is roughly in -2..2,
// on the same scale as the coordinates (matters for distance-based k-NN).
export const ENGINEERED_NAMES = [
  "ratio_index", "ratio_middle", "ratio_ring", "ratio_pinky",
  "bend_index", "bend_middle", "bend_ring", "bend_pinky",   // / 180
  "thumbOut", "thumbSide", "gapTI", "gapTM", "tipGap",
  "spreadIM", "spreadIR",                                   // / 90
  "angleTI",                                                // / 180
  "crossIM", "gapTMp", "indexReach",
  "indexTilt", "handTilt",                                  // / 180
  "palmFacing",
];
// 21 landmarks x 3 + engineered
export const VECTOR_LENGTH = 63 + ENGINEERED_NAMES.length;

const scratch = makeFeatures();

// pts: 21 {x, y, z} points in canvas pixels (as tracked by main.js).
// handLabel: "Left" or "Right". Left hands are mirrored into right-hand coordinates.
export function toVector(pts, handLabel, out = new Float32Array(VECTOR_LENGTH)) {
  const w = pts[WRIST];
  const size = dist(w, pts[MIDDLE_MCP]) || 1;
  const flip = handLabel === "Left" ? -1 : 1;

  // a-c) translate to the wrist, scale by hand size, mirror left hands
  for (let i = 0; i < 21; i++) {
    out[i * 3] = (flip * (pts[i].x - w.x)) / size;
    out[i * 3 + 1] = (pts[i].y - w.y) / size;
    out[i * 3 + 2] = (pts[i].z - w.z) / size;
  }

  // e) engineered measurements (already invariant to position, size and mirroring)
  const f = computeFeatures(pts, scratch);
  let k = 63;
  for (let i = 0; i < 4; i++) out[k++] = f.ratio[i];
  for (let i = 0; i < 4; i++) out[k++] = f.bend[i] / 180;
  out[k++] = f.thumbOut;
  out[k++] = f.thumbSide;
  out[k++] = f.gapTI;
  out[k++] = f.gapTM;
  out[k++] = f.tipGap;
  out[k++] = f.spreadIM / 90;
  out[k++] = f.spreadIR / 90;
  out[k++] = f.angleTI / 180;
  out[k++] = f.crossIM;
  out[k++] = f.gapTMp;
  out[k++] = f.indexReach;
  out[k++] = f.indexTilt / 180;
  out[k++] = f.handTilt / 180;
  out[k++] = f.palmFacing;
  return out;
}

// =============================================================================
//  Raw landmark storage (what a recorded sample keeps)
// =============================================================================

// 21 points -> flat array of 63 numbers (canvas pixels, 2 decimals).
export function packLandmarks(pts) {
  const out = new Array(63);
  for (let i = 0; i < 21; i++) {
    out[i * 3] = Math.round(pts[i].x * 100) / 100;
    out[i * 3 + 1] = Math.round(pts[i].y * 100) / 100;
    out[i * 3 + 2] = Math.round(pts[i].z * 100) / 100;
  }
  return out;
}

const unpacked = Array.from({ length: 21 }, () => ({ x: 0, y: 0, z: 0 }));

// Flat 63 numbers -> 21 points (reuses one scratch array: copy if you keep it).
export function unpackLandmarks(lm, out = unpacked) {
  for (let i = 0; i < 21; i++) {
    out[i].x = lm[i * 3];
    out[i].y = lm[i * 3 + 1];
    out[i].z = lm[i * 3 + 2];
  }
  return out;
}

// A stored sample -> its preprocessed feature vector.
export function sampleToVector(sample, out) {
  return toVector(unpackLandmarks(sample.lm), sample.hand, out);
}
