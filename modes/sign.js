// =============================================================================
//  modes/sign.js: SIGN mode, static ASL fingerspelling (rule-based)
// =============================================================================
//
//  Recognises 19 letters that can be told apart from ONE frame of landmarks:
//      A  B  C  D  F  G  H  I  K  L  O  P  Q  R  U  V  W  X  Y
//  (J and Z are drawn in the air, so they need motion. E, M, N, S and T
//   differ only in where the thumb hides among curled fingers, which is too
//   subtle for hand-written rules. The learned classifier will handle those.)
//
//  HOW THE CLASSIFIER WORKS
//  ------------------------
//  1. FEATURES: turn the 21 landmarks into a handful of meaningful numbers
//     (computeFeatures):
//       - per finger: extension ratio (shared gestures.js measure) and
//         bend angle (first bone vs the line from PIP joint to tip, in 3D)
//       - thumb: how far it sticks out, and which side of the index
//         knuckle its tip is on
//       - gaps between fingertips, spread angles between fingers
//     Every distance is divided by hand size, so it works near or far.
//
//  2. FUZZY CHECKS: each letter is a list of conditions, e.g.
//        V = index extended AND middle extended AND ring curled AND
//            pinky curled AND index/middle spread apart
//     Instead of hard true/false, each condition gives a score 0..1:
//        above(v, threshold, soft) = 0.5 exactly at the threshold,
//                                    1.0 when comfortably above it,
//                                    0.0 when clearly below it.
//     A letter's score is the MINIMUM of its conditions (a fuzzy AND):
//     a letter is only as believable as its weakest condition.
//
//  3. DECISION: the letter with the highest score wins if it's above
//     CFG.minConfidence. That score is the confidence bar on screen.
//
//  LEFT vs RIGHT HAND, AND THE MIRRORED VIDEO
//  ------------------------------------------
//  Nothing here uses screen left/right. "Which side is the thumb on" is
//  measured in a HAND-LOCAL frame built from the hand's own landmarks:
//     up   axis: wrist -> middle knuckle
//     side axis: pinky knuckle -> index knuckle
//  For a left hand, a right hand, mirrored or not, "toward the index side"
//  is always the +side direction. Angles between bones don't change when
//  an image is mirrored either. So one set of rules works for both hands.
//
//  ORIENTATION (for G/H, which point sideways, and P/Q, which point down)
//  ----------------------------------------------------------------------
//  Most features are deliberately rotation-invariant, so a tilted hand still
//  reads the same. But G vs Q, H vs U and K vs P have the SAME handshape and
//  differ only in where the hand points, so we also measure direction
//  relative to the SCREEN, as an unsigned angle from straight up:
//      0° = up      90° = sideways (left OR right)      180° = down
//  Being unsigned makes it mirror-safe: pointing left with your right hand
//  and pointing right with your left hand both read 90°.
//
//  All thresholds live in CFG below. Turn on the debug panel (D) to see the
//  live feature values and letter scores while you tune them. CFG is also
//  exposed as window.SIGN_CONFIG so you can tweak it from the console.
// =============================================================================

import {
  WRIST, THUMB_MCP, THUMB_TIP, INDEX_MCP, INDEX_PIP, INDEX_TIP, MIDDLE_MCP,
  MIDDLE_PIP, MIDDLE_TIP, RING_TIP, PINKY_MCP, PINKY_TIP, FINGERS,
  dist, extensionRatio, thumbOutRatio, Debouncer,
} from "../gestures.js";

// ---- Every threshold in one place ----
const CFG = {
  // --- Finger states (from the shared extension ratio + 3D bend angle) ---
  extendedRatio: 1.15,    // ratio above this = extended (matches gestures.js)
  curledRatio: 0.95,      // ratio below this = curled (tip folded back past the PIP joint)
  ratioSoft: 0.2,         // fuzzy width for ratio checks
  // Bend (see fingerBend): straight ≈ 0-15°, C ≈ 60°, O ≈ 90°, hook/fist ≈ 120°+
  straightBendMax: 45,    // extended fingers bend less than this (degrees)
  curvedBendMin: 30,      // "curved" (C / O): bent at least this much...
  curvedBendMax: 110,     // ...but less than a tight fist (~125)
  curvedRatioMin: 0.65,   // ...and the tip isn't tucked deep into the palm
  bendSoft: 30,           // fuzzy width for bend-angle checks

  // --- Thumb ---
  // "Thumb out" (L, Y) passes if EITHER the thumb sticks out from the palm
  // (thumbOutRatio; it can't go much above ~1.2 because the tip is only one
  // short bone past the IP joint) OR its tip is far out to the side.
  thumbOutRatio: 1.1,
  thumbOutSoft: 0.12,
  thumbOutSide: 0.4,      // thumb tip this far beyond the index knuckle line = out
  thumbInSide: -0.12,     // thumb tip this far INSIDE the index knuckle line = tucked (B, I, V, W)
  aThumbSide: -0.05,      // A: thumb tip beside the index, i.e. at/outside that line
  aGapMin: 0.2,           // A: ...but NOT touching the index tip (that would be O)
  sideSoft: 0.25,

  // --- Letter-specific (distances in hand sizes, angles in degrees) ---
  bTipGapMax: 0.45,       // B: fingers together (mean gap between neighbouring tips)
  cGapMin: 0.45,          // C: thumb tip and index tip clearly apart...
  cGapMax: 1.5,           //    ...but still facing each other
  oGapMax: 0.35,          // O: thumb tip touches the index tip
  dTouchMax: 0.5,         // D: thumb tip touches the middle fingertip
  gapSoft: 0.25,
  lAngleMin: 50,          // L: angle between thumb and index
  vSpreadMin: 12,         // V: angle between index and middle
  uSpreadMax: 10,         // U / H: index and middle held together (below this angle)
  wSpreadMin: 22,         // W: angle between index and ring
  gParallelMax: 35,       // G / Q: thumb roughly parallel to the index
  angleSoft: 20,
  crossMax: 0,            // R: index tip crossed to the pinky side of the middle tip (crossIM < 0)
  crossSoft: 0.12,
  kThumbGapMax: 0.4,      // K / P: thumb tip between index and middle (near the middle PIP joint)
  fGapMax: 0.35,          // F: thumb tip touches the index tip
  xReachMin: 1.35,        // X: index knuckle straight (PIP far from the wrist)...
  reachSoft: 0.15,
  xBendMin: 90,           //    ...but the finger is hooked at PIP + DIP

  // --- Orientation (degrees from screen-up: 0 = up, 90 = sideways, 180 = down) ---
  uprightMax: 45,         // U, K: pointing up
  sidewaysMin: 55,        // G, H: pointing sideways...
  sidewaysMax: 125,
  downMin: 120,           // P, Q: pointing down
  tiltSoft: 30,

  // --- Decision + acceptance ---
  minConfidence: 0.55,    // below this, no letter is shown
  acceptConfidence: 0.65, // a letter must be at least this sure to be typed
  holdMs: 800,            // ...and held steady this long
  stableFrames: 3,        // frames in a row before the shown letter changes
};
window.SIGN_CONFIG = CFG; // live tuning from the browser console

const LETTERS = ["A", "B", "C", "D", "F", "G", "H", "I", "K", "L", "O", "P", "Q", "R", "U", "V", "W", "X", "Y"];
const FINGER_NAMES = ["Index", "Middle", "Ring", "Pinky"];

// Reference handshapes for Practice mode (palm facing the camera).
const DESCRIPTIONS = {
  A: "Make a fist with your thumb resting against the side of your index finger, not across your fingers.",
  B: "Hold four fingers straight up and together, with your thumb folded across your palm.",
  C: "Curve your fingers and thumb into a C shape, like holding a cup.",
  D: "Point your index finger up. Touch your other fingertips to your thumb tip to make a circle.",
  I: "Raise only your little finger. Curl the rest into a fist with your thumb across them.",
  L: "Point your index finger up and stick your thumb out to the side to make an L.",
  O: "Touch all your fingertips to your thumb tip to make an O shape.",
  V: "Raise your index and middle fingers and spread them apart (peace sign). Fold the rest down.",
  W: "Raise your index, middle and ring fingers spread apart. Your thumb holds your little finger down.",
  Y: "Stick out your thumb and little finger. Curl your middle three fingers down (hang loose).",
  F: "Touch your thumb tip to your index fingertip to make a circle. Hold the other three fingers up and apart (like “OK”).",
  G: "Point your index finger sideways, with your thumb parallel just above it. Curl the other fingers.",
  H: "Hold your index and middle fingers together and point them sideways. Curl the other fingers.",
  K: "Raise your index and middle fingers in a V, and put your thumb tip up between them, against the middle finger.",
  P: "Make a K, then point it down: index forward/down, middle finger down, thumb between them.",
  Q: "Make a G, then point it down: index and thumb both pointing at the floor.",
  R: "Raise your index and middle fingers and cross them, with the index over the middle (like “fingers crossed”).",
  U: "Raise your index and middle fingers held together, pointing up. Thumb holds the other fingers down.",
  X: "Raise your index finger and hook it, bending the top two joints like a hook. Curl the other fingers.",
};

// =============================================================================
//  Fuzzy helpers
// =============================================================================
const clamp01 = (v) => (v < 0 ? 0 : v > 1 ? 1 : v);
// 0.5 at the threshold, 1 when `soft/2` past it in the right direction, 0 the other way.
const above = (v, threshold, soft) => clamp01(0.5 + (v - threshold) / soft);
const below = (v, threshold, soft) => clamp01(0.5 + (threshold - v) / soft);

// =============================================================================
//  Geometry
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
//  1. Features
// =============================================================================

function makeFeatures() {
  return {
    ratio: [0, 0, 0, 0],     // extension ratio per finger (index..pinky)
    bend: [0, 0, 0, 0],      // bend per finger, degrees (see fingerBend)
    ext: [0, 0, 0, 0],       // fuzzy "extended" score per finger
    curl: [0, 0, 0, 0],      // fuzzy "curled" score
    curve: [0, 0, 0, 0],     // fuzzy "curved" score (C / O)
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

function computeFeatures(pts, f) {
  const w = pts[WRIST];
  const size = dist(w, pts[MIDDLE_MCP]) || 1;

  // --- Fingers ---
  for (let i = 0; i < 4; i++) {
    const [mcp, pip, , tip] = FINGERS[i];
    const r = extensionRatio(pts, FINGERS[i]);
    const b = fingerBend(pts[mcp], pts[pip], pts[tip]);
    f.ratio[i] = r;
    f.bend[i] = b;
    // Extended: tip well past the PIP joint AND the finger is fairly straight.
    f.ext[i] = Math.min(above(r, CFG.extendedRatio, CFG.ratioSoft), below(b, CFG.straightBendMax, CFG.bendSoft));
    // Curled: tip folded back to (or behind) the PIP joint, like in a fist.
    f.curl[i] = below(r, CFG.curledRatio, CFG.ratioSoft);
    // Curved: clearly bent, but not folded into the palm (the C / O shape).
    f.curve[i] = Math.min(
      above(b, CFG.curvedBendMin, CFG.bendSoft),
      below(b, CFG.curvedBendMax, CFG.bendSoft),
      above(r, CFG.curvedRatioMin, CFG.ratioSoft)
    );
  }

  // --- Hand-local side axis (see the header): pinky knuckle -> index knuckle,
  // with the "up" (wrist -> middle knuckle) component removed.
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
  // Measured on the hand-local side axis, so it works for both hands.
  f.crossIM = sideOf(pts[INDEX_TIP]) - sideOf(pts[MIDDLE_TIP]);
  // K / P: the thumb tip pokes up between index and middle, near the middle PIP.
  f.gapTMp = dist(pts[THUMB_TIP], pts[MIDDLE_PIP]) / size;
  // X: if the index knuckle (MCP) is straight, the PIP joint stays far from
  // the wrist even though the finger is hooked. In a fist the knuckle bends
  // too, pulling the PIP joint in.
  f.indexReach = dist(w, pts[INDEX_PIP]) / (dist(w, pts[INDEX_MCP]) || 1);

  // --- Orientation relative to the screen (see the header) ---
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
//  2. Letter rules (fuzzy AND = Math.min of all conditions)
// =============================================================================
// Finger indices: 0 index, 1 middle, 2 ring, 3 pinky.

function scoreLetters(f, s) {
  const [ie, me, re, pe] = f.ext;
  const [ic, mc, rc, pc] = f.curl;
  const [iv, mv, rv, pv] = f.curve;
  const notExt = (i) => 1 - f.ext[i];
  // Fuzzy OR = Math.max: either kind of evidence is enough.
  const thumbOut = Math.max(
    above(f.thumbOut, CFG.thumbOutRatio, CFG.thumbOutSoft),
    above(f.thumbSide, CFG.thumbOutSide, CFG.sideSoft)
  );
  const thumbIn = below(f.thumbSide, CFG.thumbInSide, CFG.sideSoft);

  // A: fist + thumb BESIDE the index (not across the fingers).
  //    A plain fist (thumb across the fingers) fails the thumb-side check, so
  //    it isn't mistaken for A. The gap check stops O (thumb touching the
  //    index tip) from counting as A.
  s.A = Math.min(ic, mc, rc, pc,
    above(f.thumbSide, CFG.aThumbSide, CFG.sideSoft), above(f.gapTI, CFG.aGapMin, CFG.gapSoft));

  // B: four straight fingers held together, thumb folded across the palm.
  s.B = Math.min(ie, me, re, pe, below(f.tipGap, CFG.bTipGapMax, CFG.gapSoft), thumbIn);

  // C: all fingers curved, thumb opposite with a clear gap (the "cup").
  s.C = Math.min(iv, mv, rv, pv,
    above(f.gapTI, CFG.cGapMin, CFG.gapSoft), below(f.gapTI, CFG.cGapMax, CFG.gapSoft));

  // O: like C, but the thumb tip touches the index tip, closing the circle.
  //    (Ring/pinky only need to be "not extended": they often curl tighter.)
  s.O = Math.min(iv, mv, notExt(2), notExt(3), below(f.gapTI, CFG.oGapMax, CFG.gapSoft));

  // D: index up; middle finger CURVED round to touch the thumb tip (a circle).
  //    Requiring "curved" (not tightly curled) plus the touch is what
  //    separates D from just pointing with a fist, where the tucked thumb
  //    can also rest near the middle fingertip.
  s.D = Math.min(ie, mv, notExt(2), notExt(3), below(f.gapTM, CFG.dTouchMax, CFG.gapSoft));

  // I: only the pinky up, thumb tucked across the fingers.
  s.I = Math.min(ic, mc, rc, pe, thumbIn);

  // L: index up + thumb out at a wide angle, the rest curled.
  s.L = Math.min(ie, mc, rc, pc, thumbOut, above(f.angleTI, CFG.lAngleMin, CFG.angleSoft));

  // --- Shared building blocks for the two-finger letters (U V R K H P) ---
  const together = below(f.spreadIM, CFG.uSpreadMax, CFG.angleSoft);
  const uncrossed = above(f.crossIM, CFG.crossMax, CFG.crossSoft);
  const crossed = below(f.crossIM, CFG.crossMax, CFG.crossSoft);
  const thumbBetween = below(f.gapTMp, CFG.kThumbGapMax, CFG.gapSoft);
  const thumbNotBetween = 1 - thumbBetween;
  // Orientation (screen-relative, see the header)
  const upright = below(f.indexTilt, CFG.uprightMax, CFG.tiltSoft);
  const sideways = Math.min(above(f.indexTilt, CFG.sidewaysMin, CFG.tiltSoft),
    below(f.indexTilt, CFG.sidewaysMax, CFG.tiltSoft));
  // For "down", either the index OR the whole hand pointing down counts
  // (in P the index often points forward while the hand points down).
  const down = Math.max(above(f.indexTilt, CFG.downMin, CFG.tiltSoft),
    above(f.handTilt, CFG.downMin, CFG.tiltSoft));

  // V: index + middle up AND spread apart, not crossed, thumb not between them.
  s.V = Math.min(ie, me, rc, pc, above(f.spreadIM, CFG.vSpreadMin, CFG.angleSoft), uncrossed, thumbNotBetween);

  // U: index + middle together, pointing UP (sideways it would be H).
  s.U = Math.min(ie, me, rc, pc, together, uncrossed, thumbNotBetween, upright);

  // H: the U handshape pointing SIDEWAYS.
  s.H = Math.min(ie, me, rc, pc, together, uncrossed, sideways);

  // R: index and middle up and CROSSED (index tip on the pinky side).
  s.R = Math.min(ie, me, rc, pc, crossed);

  // K: index + middle up, thumb tip between them, pointing UP.
  s.K = Math.min(ie, me, rc, pc, thumbBetween, upright);

  // P: the K handshape pointing DOWN.
  s.P = Math.min(ie, me, rc, pc, thumbBetween, down);

  // F: thumb and index tips touch (a circle), the other three fingers up.
  //    (O also touches, but its other fingers are curved, not straight.)
  s.F = Math.min(notExt(0), me, re, pe, below(f.gapTI, CFG.fGapMax, CFG.gapSoft));

  // X: index knuckle straight but the finger hooked; others curled; thumb
  //    not touching the index tip (that would be O).
  s.X = Math.min(above(f.indexReach, CFG.xReachMin, CFG.reachSoft), above(f.bend[0], CFG.xBendMin, CFG.bendSoft),
    mc, rc, pc, above(f.gapTI, CFG.aGapMin, CFG.gapSoft));

  // G: index pointing SIDEWAYS, thumb parallel to it (not tucked), rest curled.
  //    (L has the thumb at ~90° to the index instead.)
  const gShape = Math.min(ie, mc, rc, pc, below(f.angleTI, CFG.gParallelMax, CFG.angleSoft),
    above(f.thumbSide, CFG.aThumbSide, CFG.sideSoft));
  s.G = Math.min(gShape, sideways);

  // Q: the G handshape pointing DOWN.
  s.Q = Math.min(gShape, down);

  // W: index + middle + ring up and spread, pinky down.
  s.W = Math.min(ie, me, re, pc, above(f.spreadIR, CFG.wSpreadMin, CFG.angleSoft));

  // Y: thumb + pinky out, middle three curled. (I is the same without the thumb.)
  s.Y = Math.min(ic, mc, rc, pe, thumbOut);
  return s;
}

// =============================================================================
//  3. Decision, hold-to-accept, spelling, practice
// =============================================================================

let app = null;
const perHand = {};   // label -> { feat, scores, letter, conf }
const stable = new Debouncer(CFG.stableFrames, "");

let bestHand = null;  // the hand whose result is shown this frame
let bestLetter = "";
let bestConf = 0;
let heldLetter = "";  // the debounced letter on screen
let holdStart = 0;
let holdProgress = 0;
let locked = false;   // true after accepting, until the letter changes

let text = "";
let practice = false;
let practiceOrder = [];
let practiceIdx = 0;
let practiceTarget = "";
let practiceMatched = 0;
let practiceNextAt = 0; // after a match, advance at this time
let debugOn = false;
let debugTick = 0;

// DOM
let letterEl, letterCard, confBar, confText, holdRing, textEl;
let practiceEl, practiceLetterEl, practiceDescEl, practiceScoreEl, practiceTickEl, practiceNote;
let debugEl, spellBtn, practiceBtn, nextBtn, debugToggle;
const shown = { letter: null, faded: null, text: null };

function handResult(label) {
  if (!perHand[label]) {
    perHand[label] = { feat: makeFeatures(), scores: Object.fromEntries(LETTERS.map((l) => [l, 0])), letter: "", conf: 0 };
  }
  return perHand[label];
}

function classify(r) {
  r.letter = "";
  r.conf = 0;
  for (const l of LETTERS) {
    if (r.scores[l] > r.conf) {
      r.conf = r.scores[l];
      r.letter = l;
    }
  }
  if (r.conf < CFG.minConfidence) r.letter = "";
}

function accept(letter, now) {
  letterCard.classList.remove("accepted");
  void letterCard.offsetWidth; // restart the CSS animation
  letterCard.classList.add("accepted");
  if (!practice) {
    setText(text + letter);
  } else if (letter === practiceTarget && !practiceNextAt) {
    practiceMatched++;
    practiceTickEl.classList.add("show");
    practiceNote.textContent = "Nice! ✓";
    practiceScoreEl.textContent = `${practiceMatched} matched`;
    practiceNextAt = now + 1200;
  } else if (!practiceNextAt) {
    practiceNote.textContent = `That looked like “${letter}”. Check the description and try again.`;
  }
}

function setText(t) {
  text = t.slice(-80);
  if (shown.text !== text) textEl.textContent = shown.text = text;
}

function shuffle(a) {
  for (let i = a.length - 1; i > 0; i--) {
    const j = (Math.random() * (i + 1)) | 0;
    [a[i], a[j]] = [a[j], a[i]];
  }
  return a;
}

function showPracticeTarget() {
  practiceTarget = practiceOrder[practiceIdx % practiceOrder.length];
  practiceLetterEl.textContent = practiceTarget;
  practiceDescEl.textContent = DESCRIPTIONS[practiceTarget];
  practiceTickEl.classList.remove("show");
  practiceNote.textContent = "";
  practiceNextAt = 0;
}

function nextPractice() {
  practiceIdx++;
  if (practiceIdx % practiceOrder.length === 0) shuffle(practiceOrder);
  showPracticeTarget();
}

function setPractice(on) {
  practice = on;
  spellBtn.classList.toggle("active", !on);
  practiceBtn.classList.toggle("active", on);
  practiceEl.hidden = !on;
  nextBtn.hidden = !on;
  textEl.parentElement.hidden = on;
  if (on) {
    practiceOrder = shuffle([...LETTERS]);
    practiceIdx = 0;
    practiceMatched = 0;
    practiceScoreEl.textContent = "0 matched";
    showPracticeTarget();
  }
}

function setDebug(on) {
  debugOn = on;
  debugToggle.checked = on;
  debugEl.hidden = !on;
}

// ---- Debug panel ----
const fmt = (v, d = 2) => v.toFixed(d);
const tiltWord = (t) => (t < CFG.uprightMax ? "up" : t < CFG.sidewaysMax ? "side" : "down");
function fingerState(f, i) {
  const e = f.ext[i], c = f.curl[i], v = f.curve[i];
  const m = Math.max(e, c, v);
  if (m < 0.5) return ["?", "unk"];
  return m === e ? ["extended", "ext"] : m === c ? ["curled", "curl"] : ["curved", "curve"];
}

function renderDebug() {
  if (!debugOn || (debugTick++ & 1)) return; // every 2nd frame is plenty
  if (!bestHand) {
    debugEl.innerHTML = `<div class="dbg-title">Debug</div><div class="dbg-muted">Show a hand to the camera</div>`;
    return;
  }
  const r = perHand[bestHand.label];
  const f = r.feat;
  let rows = "";
  for (let i = 0; i < 4; i++) {
    const [label, cls] = fingerState(f, i);
    rows += `<tr><td>${FINGER_NAMES[i]}</td><td>${fmt(f.ratio[i])}</td><td>${Math.round(f.bend[i])}°</td><td class="${cls}">${label}</td></tr>`;
  }
  const thumbState = f.thumbOut > CFG.thumbOutRatio || f.thumbSide > CFG.thumbOutSide ? ["out", "ext"]
    : f.thumbSide < CFG.thumbInSide ? ["tucked", "curl"] : ["beside", "curve"];
  rows += `<tr><td>Thumb</td><td>${fmt(f.thumbOut)}</td><td>side ${fmt(f.thumbSide)}</td><td class="${thumbState[1]}">${thumbState[0]}</td></tr>`;

  const top = [...LETTERS].sort((a, b) => r.scores[b] - r.scores[a]).slice(0, 4);
  const bars = top.map((l) =>
    `<div class="dbg-score"><span>${l}</span><div class="dbg-bar"><div style="transform:scaleX(${r.scores[l].toFixed(3)})"></div></div><span>${Math.round(r.scores[l] * 100)}%</span></div>`
  ).join("");

  debugEl.innerHTML =
    `<div class="dbg-title">Debug · ${bestHand.label} hand</div>` +
    `<table><tr><th>Finger</th><th>Ratio</th><th>Bend</th><th>State</th></tr>${rows}</table>` +
    `<div class="dbg-grid">` +
    `<span>Thumb–index gap</span><b>${fmt(f.gapTI)}</b>` +
    `<span>Thumb–middle gap</span><b>${fmt(f.gapTM)}</b>` +
    `<span>Tip gap (B)</span><b>${fmt(f.tipGap)}</b>` +
    `<span>Index–middle spread</span><b>${Math.round(f.spreadIM)}°</b>` +
    `<span>Index–ring spread</span><b>${Math.round(f.spreadIR)}°</b>` +
    `<span>Thumb–index angle</span><b>${Math.round(f.angleTI)}°</b>` +
    `<span>Index/middle cross (R &lt; 0)</span><b>${fmt(f.crossIM)}</b>` +
    `<span>Thumb–middle PIP (K, P)</span><b>${fmt(f.gapTMp)}</b>` +
    `<span>Index knuckle reach (X)</span><b>${fmt(f.indexReach)}</b>` +
    `</div><div class="dbg-sub">Orientation (0° up · 90° side · 180° down)</div><div class="dbg-grid">` +
    `<span>Index direction</span><b>${Math.round(f.indexTilt)}° ${tiltWord(f.indexTilt)}</b>` +
    `<span>Hand direction</span><b>${Math.round(f.handTilt)}° ${tiltWord(f.handTilt)}</b>` +
    `<span>Palm facing camera</span><b>${fmt(f.palmFacing)}</b>` +
    `</div><div class="dbg-sub">Top scores</div>${bars}`;
}

// The pure classifier pieces, exported for testing / experiments.
export const signClassifier = { CFG, LETTERS, makeFeatures, computeFeatures, scoreLetters };

// =============================================================================
//  Mode interface (see modes/draw.js for what each method is for)
// =============================================================================

export const signMode = {
  id: "sign",
  name: "Sign",

  init(ctx) {
    app = ctx;
    const $ = (id) => document.getElementById(id);
    letterEl = $("signLetter");
    letterCard = $("signLetterCard");
    confBar = $("signConfBar");
    confText = $("signConfText");
    holdRing = $("signHoldRing");
    textEl = $("signText");
    practiceEl = $("signPractice");
    practiceLetterEl = $("practiceLetter");
    practiceDescEl = $("practiceDesc");
    practiceScoreEl = $("practiceScore");
    practiceTickEl = $("practiceTick");
    practiceNote = $("practiceNote");
    debugEl = $("signDebug");
    spellBtn = $("signSpellBtn");
    practiceBtn = $("signPracticeBtn");
    nextBtn = $("practiceNextBtn");
    debugToggle = $("signDebugToggle");

    spellBtn.addEventListener("click", () => setPractice(false));
    practiceBtn.addEventListener("click", () => setPractice(true));
    nextBtn.addEventListener("click", nextPractice);
    $("signSpaceBtn").addEventListener("click", () => setText(text + " "));
    $("signBackBtn").addEventListener("click", () => setText(text.slice(0, -1)));
    $("signClearBtn").addEventListener("click", () => setText(""));
    debugToggle.addEventListener("change", () => setDebug(debugToggle.checked));
    setText("");
  },

  enter() {
    stable.reset("");
    heldLetter = "";
    locked = false;
    shown.letter = null;
  },

  exit() {},

  update(hands, now) {
    // Classify every visible hand; show the most confident one.
    bestHand = null;
    bestLetter = "";
    bestConf = 0;
    for (const hand of hands) {
      const r = handResult(hand.label);
      computeFeatures(hand.pts, r.feat);
      scoreLetters(r.feat, r.scores);
      classify(r);
      if (!bestHand || r.conf > bestConf) {
        bestHand = hand;
        bestConf = r.conf;
        bestLetter = r.letter;
      }
    }

    // Debounce, so one odd frame doesn't change the letter on screen.
    const letter = stable.update(bestLetter);
    if (letter !== heldLetter) {
      heldLetter = letter;
      holdStart = now;
      locked = false; // a new letter can be accepted
    }

    // Hold-to-accept: the same letter, confident enough, for CFG.holdMs.
    holdProgress = 0;
    if (heldLetter && !locked && bestLetter === heldLetter && bestConf >= CFG.acceptConfidence) {
      holdProgress = Math.min(1, (now - holdStart) / CFG.holdMs);
      if (holdProgress >= 1) {
        accept(heldLetter, now);
        locked = true; // to type the same letter again, change pose or drop the hand
      }
    }
    if (locked) holdProgress = 1;

    if (practice && practiceNextAt && now >= practiceNextAt) nextPractice();
  },

  render() {
    // Big letter + confidence + hold ring (DOM text only changes when needed).
    const faded = !heldLetter;
    const show = heldLetter || (bestLetter ? bestLetter : "?");
    if (shown.letter !== show || shown.faded !== faded) {
      shown.letter = show;
      shown.faded = faded;
      letterEl.textContent = show;
      letterEl.classList.toggle("faded", faded);
    }
    const conf = bestLetter ? bestConf : 0;
    confBar.style.transform = `scaleX(${conf.toFixed(3)})`;
    confBar.dataset.level = conf >= CFG.acceptConfidence ? "high" : conf >= CFG.minConfidence ? "mid" : "low";
    confText.textContent = bestHand ? `${Math.round(conf * 100)}%` : "no hand";
    holdRing.style.setProperty("--p", holdProgress.toFixed(3));
    holdRing.classList.toggle("done", locked);
    renderDebug();
  },

  handLost(label) {
    delete perHand[label];
  },

  trailColor(hand) { return app.handColors[hand.label]; },

  labelSuffix(hand) {
    return perHand[hand.label]?.letter || "";
  },

  badge() {
    return { text: practice ? "PRACTICE" : "SPELL", tone: heldLetter ? "DRAW" : "HOVER" };
  },

  onKey(e) {
    if (e.ctrlKey || e.metaKey || e.altKey) return false;
    const key = e.key.toLowerCase();
    if (e.code === "Space") {
      e.preventDefault();
      if (!practice) setText(text + " ");
    } else if (e.key === "Backspace") {
      e.preventDefault();
      if (!practice) setText(text.slice(0, -1));
    } else if (key === "c") setText("");
    else if (key === "p") setPractice(!practice);
    else if (key === "n" && practice) nextPractice();
    else if (key === "d") setDebug(!debugOn);
    else return false;
    return true;
  },
};
