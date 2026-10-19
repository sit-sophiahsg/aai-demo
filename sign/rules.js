// =============================================================================
//  sign/rules.js: the RULE-BASED letter classifier (hand-written fuzzy rules)
// =============================================================================
//
//  Recognises 19 letters that can be told apart from ONE frame of landmarks:
//      A  B  C  D  F  G  H  I  K  L  O  P  Q  R  U  V  W  X  Y
//  (J and Z are drawn in the air, so they need motion. E, M, N, S and T
//   differ only in where the thumb hides among curled fingers, which is too
//   subtle for hand-written rules. The learned classifier handles those.)
//
//  HOW IT WORKS
//  ------------
//  1. FEATURES: sign/features.js turns the 21 landmarks into ~20 meaningful
//     measurements (finger extension and bend, thumb position, gaps, spreads,
//     orientation). Distances are divided by hand size.
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
//     Math.max is used as a fuzzy OR where either piece of evidence counts.
//
//  3. DECISION (in modes/sign.js): the highest-scoring letter wins if it's
//     above CFG.minConfidence. That score is the confidence bar on screen.
//
//  LEFT vs RIGHT HAND, AND THE MIRRORED VIDEO
//  ------------------------------------------
//  "Which side is the thumb on" is measured in a HAND-LOCAL frame built from
//  the hand's own landmarks (pinky knuckle -> index knuckle), and angles
//  between bones don't change when an image is mirrored. So one set of rules
//  works for both hands.
//
//  ORIENTATION (for G/H, which point sideways, and P/Q, which point down)
//  ----------------------------------------------------------------------
//  G vs Q, H vs U and K vs P have the SAME handshape and differ only in where
//  the hand points, so some rules also use direction relative to the SCREEN,
//  as an unsigned angle from straight up:
//      0° = up      90° = sideways (left OR right)      180° = down
//
//  All thresholds live in CFG below. The debug panel (D) shows the live
//  measurements and letter scores; CFG is also exposed as
//  window.SIGN_CONFIG so you can tweak it from the browser console.
// =============================================================================

// ---- Every threshold in one place ----
export const CFG = {
  // --- Finger states (from the shared extension ratio + 3D bend angle) ---
  extendedRatio: 1.15,    // ratio above this = extended (matches gestures.js)
  curledRatio: 0.95,      // ratio below this = curled (tip folded back past the PIP joint)
  ratioSoft: 0.2,         // fuzzy width for ratio checks
  // Bend (see features.js fingerBend): straight ≈ 0-15°, C ≈ 60°, O ≈ 90°, hook/fist ≈ 120°+
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
  xBendMin: 90,           //    ...but the finger is hooked

  // --- Orientation (degrees from screen-up: 0 = up, 90 = sideways, 180 = down) ---
  uprightMax: 45,         // U, K: pointing up
  sidewaysMin: 55,        // G, H: pointing sideways...
  sidewaysMax: 125,
  downMin: 120,           // P, Q: pointing down
  tiltSoft: 30,

  // --- Decision + acceptance (used by modes/sign.js) ---
  minConfidence: 0.55,    // below this, no letter is shown
  acceptConfidence: 0.65, // a letter must be at least this sure to be typed
  holdMs: 800,            // ...and held steady this long
  stableFrames: 3,        // frames in a row before the shown letter changes
};
window.SIGN_CONFIG = CFG; // live tuning from the browser console

export const RULE_LETTERS = ["A", "B", "C", "D", "F", "G", "H", "I", "K", "L", "O", "P", "Q", "R", "U", "V", "W", "X", "Y"];

// =============================================================================
//  Fuzzy helpers
// =============================================================================
const clamp01 = (v) => (v < 0 ? 0 : v > 1 ? 1 : v);
// 0.5 at the threshold, 1 when `soft/2` past it in the right direction, 0 the other way.
const above = (v, threshold, soft) => clamp01(0.5 + (v - threshold) / soft);
const below = (v, threshold, soft) => clamp01(0.5 + (threshold - v) / soft);

// Fuzzy finger states from the raw measurements (filled into f.ext/curl/curve).
export function fingerStates(f) {
  for (let i = 0; i < 4; i++) {
    const r = f.ratio[i], b = f.bend[i];
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
  return f;
}

// =============================================================================
//  Letter rules (fuzzy AND = Math.min of all conditions)
// =============================================================================
// Finger indices: 0 index, 1 middle, 2 ring, 3 pinky.
// f: measurements from features.js computeFeatures(). s: output scores object.

export function scoreLetters(f, s) {
  fingerStates(f);
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
