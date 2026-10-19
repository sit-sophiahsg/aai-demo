// =============================================================================
//  gestures.js: shared hand geometry used by every mode
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
// =============================================================================

export const WRIST = 0;
export const THUMB_CMC = 1, THUMB_MCP = 2, THUMB_IP = 3, THUMB_TIP = 4;
export const INDEX_MCP = 5, INDEX_PIP = 6, INDEX_DIP = 7, INDEX_TIP = 8;
export const MIDDLE_MCP = 9, MIDDLE_PIP = 10, MIDDLE_DIP = 11, MIDDLE_TIP = 12;
export const RING_MCP = 13, RING_PIP = 14, RING_DIP = 15, RING_TIP = 16;
export const PINKY_MCP = 17, PINKY_PIP = 18, PINKY_DIP = 19, PINKY_TIP = 20;

// [MCP, PIP, DIP, TIP] chains for the four fingers.
export const INDEX = [5, 6, 7, 8];
export const MIDDLE = [9, 10, 11, 12];
export const RING = [13, 14, 15, 16];
export const PINKY = [17, 18, 19, 20];
export const FINGERS = [INDEX, MIDDLE, RING, PINKY];
export const FINGERTIPS = [4, 8, 12, 16, 20];

// Pairs of landmark indices that form the "bones" we draw as lines.
export const HAND_CONNECTIONS = [
  [0, 1], [1, 2], [2, 3], [3, 4],          // thumb
  [0, 5], [5, 6], [6, 7], [7, 8],          // index
  [5, 9], [9, 10], [10, 11], [11, 12],     // middle
  [9, 13], [13, 14], [14, 15], [15, 16],   // ring
  [13, 17], [17, 18], [18, 19], [19, 20],  // pinky
  [0, 17],                                 // base of the palm
];

// Thresholds shared by all modes. Exported so a debug panel can tune them live.
export const GESTURE_CONFIG = {
  // A finger is extended when wrist->tip > wrist->PIP * extendedRatio ...
  extendedRatio: 1.15,
  // ... and wrist->tip > wrist->MCP * beyondKnuckleRatio.
  beyondKnuckleRatio: 1.3,
  // Thumb is "out" when tip->pinkyMCP > thumbIP->pinkyMCP * thumbOutRatio.
  thumbOutRatio: 1.1,
  // Strict open palm (used for "clear" and "start game" gestures):
  palmStraight: 0.9,       // each finger's straightness (1.0 = dead straight)
  palmThumbSpread: 0.6,    // thumb tip <-> index knuckle, in hand sizes
  palmFingerSpread: 0.85,  // index tip <-> pinky tip, in hand sizes
};

// =============================================================================
//  Basic geometry
// =============================================================================

export function dist(a, b) {
  return Math.hypot(a.x - b.x, a.y - b.y);
}

// Write normalized landmarks (0..1) into a reused array of canvas-pixel points.
// The video is mirrored on screen, so we mirror x here too (x -> 1 - x).
// That keeps the dots on top of your hand in the mirrored video, while
// the canvas itself stays un-mirrored so text is readable.
export function toPixels(landmarks, out, width, height) {
  for (let i = 0; i < 21; i++) {
    out[i].x = (1 - landmarks[i].x) * width;
    out[i].y = landmarks[i].y * height;
  }
}

// "Hand size" = distance from the wrist (0) to the middle-finger knuckle (9).
// This stays roughly constant for a given hand regardless of finger pose, but
// it gets bigger when the hand is close to the camera and smaller when far.
// Dividing other distances by it makes them distance-independent.
export function handSize(pts) {
  return dist(pts[WRIST], pts[MIDDLE_MCP]) || 1;
}

// Straightness of a chain of joints: straight-line length divided by the
// length along the bones. 1.0 = perfectly straight, curled fingers drop to ~0.5.
export function straightness(pts, a, b, c, d) {
  const along = dist(pts[a], pts[b]) + dist(pts[b], pts[c]) + dist(pts[c], pts[d]);
  return dist(pts[a], pts[d]) / (along || 1);
}

// =============================================================================
//  Finger state
// =============================================================================

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
export function isFingerExtended(pts, [mcp, pip, , tip]) {
  const w = pts[WRIST];
  const dTip = dist(w, pts[tip]);
  return (
    dTip > dist(w, pts[pip]) * GESTURE_CONFIG.extendedRatio &&
    dTip > dist(w, pts[mcp]) * GESTURE_CONFIG.beyondKnuckleRatio
  );
}

// THE THUMB IS DIFFERENT: it bends sideways across the palm rather than
// curling toward the wrist, so its wrist distance barely changes. Instead we
// check whether the thumb tip sticks out away from the palm: it's farther
// from the pinky knuckle (17) than the thumb's own IP joint (3) is.
export function isThumbExtended(pts) {
  return (
    dist(pts[THUMB_TIP], pts[PINKY_MCP]) >
    dist(pts[THUMB_IP], pts[PINKY_MCP]) * GESTURE_CONFIG.thumbOutRatio
  );
}

// Fill a reused object with all five finger states (no allocation per frame).
export function readFingers(pts, out) {
  out.thumb = isThumbExtended(pts);
  out.index = isFingerExtended(pts, INDEX);
  out.middle = isFingerExtended(pts, MIDDLE);
  out.ring = isFingerExtended(pts, RING);
  out.pinky = isFingerExtended(pts, PINKY);
  out.count = out.thumb + out.index + out.middle + out.ring + out.pinky;
  return out;
}

// Draw-mode poses from the four fingers (thumb ignored on purpose, so it's
// fine whether yours is tucked in or sticking out).
//   ☝ index only     -> DRAW
//   ✌ index + middle -> HOVER
//   anything else    -> IDLE
export function classifyDrawPose(f) {
  if (f.ring || f.pinky) return "IDLE";   // open palm, rock sign, etc.
  if (f.index && !f.middle) return "DRAW";
  if (f.index && f.middle) return "HOVER";
  return "IDLE";                          // fist, or only the middle finger
}

// STRICT open palm, so a relaxed hand doesn't trigger it by accident:
//  - all four fingers straight, and pointing away from the wrist
//  - thumb straight, sticking out, and spread away from the index knuckle
//  - fingers fanned out (index tip far from pinky tip)
export function isOpenPalm(pts) {
  const c = GESTURE_CONFIG;
  const size = handSize(pts);
  for (const chain of FINGERS) {
    if (straightness(pts, chain[0], chain[1], chain[2], chain[3]) < c.palmStraight) return false;
    if (!isFingerExtended(pts, chain)) return false;
  }
  const thumbStraight =
    dist(pts[THUMB_MCP], pts[THUMB_TIP]) /
    (dist(pts[THUMB_MCP], pts[THUMB_IP]) + dist(pts[THUMB_IP], pts[THUMB_TIP]) || 1);
  if (thumbStraight < c.palmStraight || !isThumbExtended(pts)) return false;
  if (dist(pts[THUMB_TIP], pts[INDEX_MCP]) < c.palmThumbSpread * size) return false;
  if (dist(pts[INDEX_TIP], pts[PINKY_TIP]) < c.palmFingerSpread * size) return false;
  return true;
}

// =============================================================================
//  Filters
// =============================================================================

// ONE EURO FILTER
// A low-pass filter whose cutoff frequency adapts to speed:
//   - moving slowly -> low cutoff  -> heavy smoothing -> steady, no jitter
//   - moving fast   -> high cutoff -> light smoothing -> no lag
//   cutoff = minCutoff + beta * |speed|
// `config` is a shared object { minCutoff, beta }, so UI sliders can tune
// every filter live. Positions are in canvas pixels, speed in pixels/second.
// Paper: Casiez et al., "1€ Filter", CHI 2012.
function smoothingAlpha(cutoff, dt) {
  const tau = 1 / (2 * Math.PI * cutoff);
  return 1 / (1 + tau / dt);
}

export class OneEuroFilter {
  constructor(config, dCutoff = 1.0) {
    this.config = config;
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
    const cutoff = this.config.minCutoff + this.config.beta * Math.abs(this.dx);
    // 3. Standard exponential smoothing with that cutoff.
    this.x += smoothingAlpha(cutoff, dt) * (value - this.x);
    return this.x;
  }
}

// DEBOUNCER
// Only reports a new value once it has been seen `frames` frames in a row.
// One noisy frame can't flip the state.
export class Debouncer {
  constructor(frames, initial) {
    this.frames = frames;
    this.reset(initial);
  }
  reset(value) {
    this.value = value;       // the confirmed (stable) value
    this.candidate = value;   // what recent frames have been showing
    this.count = 0;
  }
  update(raw) {
    if (raw === this.candidate) this.count++;
    else {
      this.candidate = raw;
      this.count = 1;
    }
    if (this.count >= this.frames) this.value = raw;
    return this.value;
  }
}
