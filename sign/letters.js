// =============================================================================
//  sign/letters.js: the letter sets and reference handshape descriptions
// =============================================================================

// All 24 static ASL letters (A-Y without J; J and Z need motion).
// These are the classes for data collection and the learned models.
export const STATIC_LETTERS = [
  "A", "B", "C", "D", "E", "F", "G", "H", "I", "K", "L", "M",
  "N", "O", "P", "Q", "R", "S", "T", "U", "V", "W", "X", "Y",
];

// The two moving letters, recognised from the fingertip's path (sign/motion.js).
export const MOTION_LETTERS = ["J", "Z"];

// Reference handshapes (palm facing the camera unless stated).
export const DESCRIPTIONS = {
  A: "Make a fist with your thumb resting against the side of your index finger, not across your fingers.",
  B: "Hold four fingers straight up and together, with your thumb folded across your palm.",
  C: "Curve your fingers and thumb into a C shape, like holding a cup.",
  D: "Point your index finger up. Touch your other fingertips to your thumb tip to make a circle.",
  E: "Curl all four fingertips down onto your thumb, which is folded across your palm under them.",
  F: "Touch your thumb tip to your index fingertip to make a circle. Hold the other three fingers up and apart (like “OK”).",
  G: "Point your index finger sideways, with your thumb parallel just above it. Curl the other fingers.",
  H: "Hold your index and middle fingers together and point them sideways. Curl the other fingers.",
  I: "Raise only your little finger. Curl the rest into a fist with your thumb across them.",
  K: "Raise your index and middle fingers in a V, and put your thumb tip up between them, against the middle finger.",
  L: "Point your index finger up and stick your thumb out to the side to make an L.",
  M: "Make a fist with your thumb tucked under your index, middle and ring fingers.",
  N: "Make a fist with your thumb tucked under your index and middle fingers.",
  O: "Touch all your fingertips to your thumb tip to make an O shape.",
  P: "Make a K, then point it down: index forward/down, middle finger down, thumb between them.",
  Q: "Make a G, then point it down: index and thumb both pointing at the floor.",
  R: "Raise your index and middle fingers and cross them, with the index over the middle (like “fingers crossed”).",
  S: "Make a fist with your thumb wrapped across the front of your fingers.",
  T: "Make a fist with your thumb tucked between your index and middle fingers, tip poking up.",
  U: "Raise your index and middle fingers held together, pointing up. Thumb holds the other fingers down.",
  V: "Raise your index and middle fingers and spread them apart (peace sign). Fold the rest down.",
  W: "Raise your index, middle and ring fingers spread apart. Your thumb holds your little finger down.",
  X: "Raise your index finger and hook it, bending the top two joints like a hook. Curl the other fingers.",
  Y: "Stick out your thumb and little finger. Curl your middle three fingers down (hang loose).",
  J: "Make an I (little finger up), then draw a J in the air with your little finger: down, then hook sideways.",
  Z: "Point with your index finger and draw a Z in the air: across, diagonally down and back, then across again.",
};
