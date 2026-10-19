// =============================================================================
//  tests/synthetic-hands.js: a tiny 3D hand model for automated tests
// =============================================================================
//  Builds idealised landmark sets for letter handshapes, so the rules and the
//  preprocessing can be checked without a camera. Coordinates: wrist at the
//  origin, fingers pointing up (-y), hand size (wrist -> middle knuckle) = 1,
//  +x = index side. Flexing a finger curls it toward the camera (-z).
//  These are idealised: real hands and MediaPipe's estimates are noisier.
// =============================================================================

const MCP = { index: [0.32, -0.95], middle: [0, -1], ring: [-0.28, -0.95], pinky: [-0.52, -0.85] };
const LEN = { index: [0.45, 0.28, 0.22], middle: [0.5, 0.3, 0.23], ring: [0.46, 0.28, 0.22], pinky: [0.36, 0.22, 0.2] };

// One finger from its knuckle: spread angle (degrees, + = toward index side)
// and flex angles at MCP, PIP, DIP (degrees).
function finger(name, spreadDeg, flex) {
  const [mx, my] = MCP[name];
  const a = (spreadDeg * Math.PI) / 180;
  const d = [Math.sin(a), -Math.cos(a)];
  const out = [{ x: mx, y: my, z: 0 }];
  let th = 0, c = out[0];
  for (let k = 0; k < 3; k++) {
    th += (flex[k] * Math.PI) / 180;
    const L = LEN[name][k];
    c = { x: c.x + d[0] * Math.cos(th) * L, y: c.y + d[1] * Math.cos(th) * L, z: c.z - Math.sin(th) * L };
    out.push(c);
  }
  return out;
}

const P = (x, y, z = 0) => ({ x, y, z });
const S = [0, 5, 5], F = [80, 100, 60], CURVE = [30, 50, 30], OC = [45, 70, 50], HOOK = [10, 95, 55];
const fist = () => ({ index: finger("index", 0, F), middle: finger("middle", 0, F), ring: finger("ring", 0, F), pinky: finger("pinky", 0, F) });
const TH = {
  tucked: [P(0.28, -0.18), P(0.32, -0.45, -0.1), P(0.15, -0.68, -0.2), P(-0.05, -0.78, -0.25)],
  beside: [P(0.28, -0.18), P(0.42, -0.45, -0.05), P(0.48, -0.7, -0.1), P(0.5, -0.92, -0.12)],
  out: [P(0.28, -0.18), P(0.5, -0.38), P(0.75, -0.5), P(1, -0.6)],
  between: [P(0.28, -0.18), P(0.36, -0.5, -0.1), P(0.3, -0.95, -0.2), P(0.15, -1.3, -0.25)],
  parallel: [P(0.28, -0.18), P(0.4, -0.45, -0.05), P(0.45, -0.72, -0.08), P(0.48, -0.98, -0.1)],
};
// A thumb whose tip lands on a given point (for C / O / D / F).
const thumbTo = (tip, dx = 0.05, dy = 0.08) => [P(0.28, -0.18), P(0.45, -0.4, -0.1),
  P((0.45 + tip.x) / 2 + 0.08, (-0.4 + tip.y) / 2 + 0.05, tip.z / 2), P(tip.x + dx, tip.y + dy, tip.z)];
const hand = (f, t) => [P(0, 0), ...t, ...f.index, ...f.middle, ...f.ring, ...f.pinky];

// Handshapes for the 19 rule letters (in the upright frame).
export const SHAPES = {};
SHAPES.A = hand(fist(), TH.beside);
SHAPES.B = hand({ index: finger("index", -5, S), middle: finger("middle", 0, S), ring: finger("ring", 5, S), pinky: finger("pinky", 10, S) }, TH.tucked);
{ const f = { index: finger("index", 0, CURVE), middle: finger("middle", 0, CURVE), ring: finger("ring", 0, CURVE), pinky: finger("pinky", 0, CURVE) }; SHAPES.C = hand(f, thumbTo(f.index[3], 0.15, 0.55)); }
{ const f = { index: finger("index", 0, OC), middle: finger("middle", 0, OC), ring: finger("ring", 0, OC), pinky: finger("pinky", 0, OC) }; SHAPES.O = hand(f, thumbTo(f.index[3])); }
{ const f = { index: finger("index", 0, S), middle: finger("middle", 0, OC), ring: finger("ring", 0, OC), pinky: finger("pinky", 0, OC) }; SHAPES.D = hand(f, thumbTo(f.middle[3])); }
{ const f = { index: finger("index", 0, OC), middle: finger("middle", 0, S), ring: finger("ring", -8, S), pinky: finger("pinky", -16, S) }; SHAPES.F = hand(f, thumbTo(f.index[3])); }
SHAPES.G = hand({ ...fist(), index: finger("index", 0, S) }, TH.parallel);
SHAPES.H = hand({ ...fist(), index: finger("index", -1, S), middle: finger("middle", 1, S) }, TH.tucked);
SHAPES.I = hand({ ...fist(), pinky: finger("pinky", 0, S) }, TH.tucked);
SHAPES.K = hand({ ...fist(), index: finger("index", 8, S), middle: finger("middle", -8, S) }, TH.between);
SHAPES.L = hand({ ...fist(), index: finger("index", 0, S) }, TH.out);
SHAPES.P = SHAPES.K;
SHAPES.Q = SHAPES.G;
SHAPES.R = hand({ ...fist(), index: finger("index", -12, S), middle: finger("middle", 12, S) }, TH.tucked);
SHAPES.U = SHAPES.H;
SHAPES.V = hand({ ...fist(), index: finger("index", 10, S), middle: finger("middle", -10, S) }, TH.tucked);
SHAPES.W = hand({ ...fist(), index: finger("index", 15, S), middle: finger("middle", 0, S), ring: finger("ring", -15, S) }, TH.tucked);
SHAPES.X = hand({ ...fist(), index: finger("index", 0, HOOK) }, TH.tucked);
SHAPES.Y = hand({ ...fist(), pinky: finger("pinky", -20, S) }, TH.out);

// Natural orientation of each letter: degrees of rotation in the image
// plane (0 = fingers up). G/H point sideways, P/Q point down.
export const ORIENT = { G: 90, H: 90, P: 155, Q: 165 };

// Poses that should NOT be recognised as any rule letter.
export const REJECTS = {
  "fist(S)": hand(fist(), TH.tucked),
  point: hand({ ...fist(), index: finger("index", 0, S) }, TH.tucked),
  open5: hand({ index: finger("index", 12, S), middle: finger("middle", 0, S), ring: finger("ring", -12, S), pinky: finger("pinky", -24, S) }, TH.out),
  "G-upright": SHAPES.G,
};

// ---- Transforms ----
export function rotate(pts, deg) {
  const a = (deg * Math.PI) / 180, c = Math.cos(a), s = Math.sin(a);
  return pts.map((p) => ({ x: p.x * c - p.y * s, y: p.x * s + p.y * c, z: p.z }));
}

// Model units -> canvas pixels. mirror = the other hand (x flipped).
export function toPx(pts, { mirror = false, scale = 100, cx = 320, cy = 260 } = {}) {
  return pts.map((p) => ({ x: cx + (mirror ? -p.x : p.x) * scale, y: cy + p.y * scale, z: p.z * scale }));
}

function gauss() {
  let u = 0, v = 0;
  while (!u) u = Math.random();
  while (!v) v = Math.random();
  return Math.sqrt(-2 * Math.log(u)) * Math.cos(2 * Math.PI * v);
}

// Random jitter on every landmark (model units: hand size = 1).
export function noisy(pts, sxy, sz) {
  return pts.map((p) => ({ x: p.x + gauss() * sxy, y: p.y + gauss() * sxy, z: p.z + gauss() * sz }));
}
