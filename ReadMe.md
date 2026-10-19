# Hand Lab: Hand Tracking in the Browser

Real-time hand tracking with MediaPipe's `HandLandmarker` (`@mediapipe/tasks-vision`).
It tracks up to 2 hands, draws the 21-point skeleton, and has three modes:

- **Draw:** air draw with your index finger.
- **Game:** Bubble Pop, popping bubbles with your fingertips.
- **Sign:** ASL fingerspelling for 19 static letters, with a Spell mode and a Practice mode.

Glowing particle trails follow your index fingertip in every mode.

## Run it

Webcams only work in a **secure context** (`https://` or `http://localhost`), so opening
`index.html` straight from disk (`file://`) **won't work**. Serve the folder with any one of these:

```bash
npx serve .                    # Node.js, then open the URL it prints (usually http://localhost:3000)
python -m http.server 8000     # Python, then open http://localhost:8000
```

No Python or Node? On Windows, use the included PowerShell server:

```powershell
powershell -ExecutionPolicy Bypass -File serve.ps1      # then open http://localhost:8000
```

Allow camera access when the browser asks. You need an internet connection, because the
library, WASM runtime and model load from jsDelivr and Google.

## Controls

### Everywhere

| Action | How |
| --- | --- |
| Switch mode | `1` Draw · `2` Game · `3` Sign, or the tabs at the top |
| Particle trails on/off | `T`, or the "Trails" checkbox. The slider sets intensity |
| Toggle skeleton | `S` |
| Smoothing | "Smoothing" menu: One Euro `min cutoff` and `beta` |

### Draw mode

| Action | How |
| --- | --- |
| **DRAW** ☝ | Only the index finger extended: the fingertip draws |
| **HOVER** ✌ | Index + middle extended: the cursor moves, pen up (use it between letters) |
| **IDLE** | Any other pose (fist, open palm…): nothing happens |
| Undo last stroke | `U`, `Ctrl+Z`, or the Undo button |
| Clear | `C` or the Clear button |
| Palm clear (off by default) | Tick "Palm clear", then hold a wide-open, spread hand still for 1.5 s |
| Color | Click a swatch or press `Shift+1` to `Shift+6` |
| Brush size | Slider, or `[` / `]` |

### Game mode: Bubble Pop

| Action | How |
| --- | --- |
| Start / restart | Hold an **open palm** (all 5 fingers out) for 1 s, or press `Space` / "Start round" |
| Pop a bubble | Touch it with an **index fingertip**. Either hand and any pose works |
| Scoring | Normal bubbles 10–20 (smaller = more) · ★ **gold** +50 · 💣 **bomb** −30, and the combo resets |
| Combo | Each pop within 1 s of the last raises the multiplier (x2, x3… up to x5) |

- Rounds last 60 s. Bubbles get faster and more numerous over the round, and bombs get more common.
- The high score is saved in your browser (`localStorage`).
- After game over, put your hand down before holding your palm up again, so you can't restart by accident.
- Switching away from Game mode abandons the current round.

### Sign mode: ASL fingerspelling

Recognises **A B C D F G H I K L O P Q R U V W X Y**. Either hand works.
- Most letters are signed with the fingers up.
- **G and H point sideways**, and **P and Q point down**. These pairs have the same handshape
  (G/Q, H/U, K/P), so the classifier also looks at which way your hand points on screen.
- Not covered by the rules:
  - J and Z need motion.
  - E, M, N, S and T differ only in where the thumb hides among curled fingers, which is too
    subtle for hand-written rules.

| Action | How |
| --- | --- |
| Type a letter | Hold the handshape steady for ~0.8 s. The ring around the big letter fills, then it's added |
| Same letter twice | Change pose or drop your hand briefly between them |
| Space / delete / clear | `Space` · `Backspace` · `C`, or the toolbar buttons |
| Practice | `P` or "Practice". Shows a target letter with a description, gives a ✓ when you match it, then moves on. `N` skips |
| Train | `T` or "Train". Records labelled samples for the learned classifier (see below). In Sign mode `T` means Train, so use the checkbox to toggle trails |
| Debug panel | `D`. Shows each finger's ratio, bend angle and state (extended / curved / curled), thumb position, key gaps and angles, and the top letter scores |

The bar under the letter is the classifier's confidence. Grey means no letter, yellow means
it's shown but not sure enough to type, and green means it's confident enough to type.

#### Train: collecting data for the learned classifier

Press `T` (or "Train") in Sign mode. A panel opens next to the video.

1. Type your name in **Signer**. It's stored with every sample, which later lets us test how
   well the model works for people it hasn't seen.
2. Pick a letter in the grid (or use `←` / `→`). All **24 static letters** (A–Y without J)
   are there, including E, M, N, S and T, which the rules can't do.
3. Make the sign and **hold `R`** (or hold the Record button). About 10 samples per second are
   recorded while it's held. While recording, the video gets a red frame and the badge shows `REC`.
4. Aim for **100 samples per letter**. The bar under each letter turns green when it gets there.
   Vary your hand a little while recording (distance, small tilts, both hands) so the model
   learns the handshape, not one exact position.

| Action | How |
| --- | --- |
| Export | "Export JSON" downloads every sample. Do this regularly: it's your backup |
| Import / merge | "Import JSON" adds samples from an exported file (e.g. a friend's). Samples already present are skipped, so importing twice is harmless |
| Delete | "Delete all “X” samples" removes one letter's samples (asks first) |

Samples are saved in the browser's IndexedDB, so they survive a reload. They're stored
per browser and per site, so clearing site data deletes them. If storage is blocked (e.g. a
private window), the panel shows "⚠ Not saved: export to keep".

The top-right corner shows the active delegate (hover it to see the WebGL renderer),
the camera resolution, model inference time and FPS.

## How it works (short version)

- **Landmarks:** 0 = wrist. Each finger has 4 points going outwards, and the tip is always
  the last: thumb 1–4, index 5–8, middle 9–12, ring 13–16, pinky 17–20. There's a diagram at
  the top of `gestures.js`.
- **Finger extended?** A finger counts as extended when its tip is clearly farther from the
  wrist than its PIP joint (and its knuckle). That uses distances, not "is the tip higher",
  so it still works when your hand is tilted. The thumb bends sideways, so it's checked
  separately (does its tip stick out away from the pinky knuckle?). DRAW and HOVER ignore the
  thumb, so it can be tucked in or out.
- **Mode switching:** a pose has to be seen 3 frames in a row before the mode changes
  (debounce), so the pen doesn't flicker. Entering DRAW starts a new stroke, and leaving it
  ends the stroke. The state shows in the top-left badge and next to each hand's label.
- **Strokes:** each stroke is stored as `{ color, size, points }` in an array. The drawing
  canvas is a separate layer that's never cleared per frame. Undo pops the last stroke and
  redraws the rest. Strokes are kept (just hidden) while you're in another mode.
- **Smoothing:** a One Euro filter on the fingertip (steady when slow, responsive when fast).
  Strokes are drawn as quadratic Bezier curves through the midpoints of consecutive points,
  and big jumps between frames are filled in with interpolated points.
- **Particle trails:** a fixed pool of 300 particles stored in typed arrays, so no objects are
  created per frame. The emission rate grows with fingertip speed, and particles are spread
  along the path the fingertip moved this frame. Each particle drifts, shrinks and fades over
  0.5–1 s. They're drawn as pre-rendered glow sprites with additive blending
  (`globalCompositeOperation = "lighter"`), so overlaps glow brighter. In Draw mode, trails only
  appear while you're actually drawing, in the brush color.
- **Bubble Pop:** bubbles and score popups come from fixed pools, and each bubble style is
  pre-rendered once to a sprite. Hits are tested against the *path* your fingertip moved
  this frame (distance from the bubble's center to a line segment), so a fast swipe still pops
  bubbles it passed between camera frames. The difficulty values (`*Start` → `*End`) are
  interpolated over the round. All tuning is in the `CONFIG` object at the top of `modes/game.js`.
- **Sign classifier (rule-based, `modes/sign.js`):**
  1. *Features:* each finger gets the shared extension ratio plus its 3D bend angle (PIP + DIP).
     The thumb gets an "out" ratio and its position across the palm. There are also gaps between
     fingertips and spread angles between fingers. All distances are divided by hand size.
  2. *Fuzzy rules:* each letter is a list of conditions, e.g. V = index extended AND middle
     extended AND ring/pinky curled AND index–middle spread > 12°. Each condition scores 0–1
     (0.5 exactly at its threshold), and a letter's score is the **minimum** of its conditions
     (a fuzzy AND).
  3. *Decision:* the highest-scoring letter wins if it's above `minConfidence`. A short debounce
     and the 0.8 s hold stop flickering letters from being typed.
  - Look-alikes are separated by specific checks:

    | Look-alikes | Separated by |
    | --- | --- |
    | A vs plain fist | Thumb beside the index, or across the fingers |
    | O vs C | Thumb tip touching the index tip, or a clear gap |
    | F vs O | Thumb touches the index tip in both; F's other fingers are straight, O's are curved |
    | V vs U vs R | Index and middle spread, together, or crossed (index tip on the pinky side of the middle tip) |
    | K vs V | Thumb tip between index and middle (near the middle PIP), or tucked |
    | X vs fist | Index knuckle straight but finger hooked, or knuckle bent too |
    | D vs pointing | Middle finger curved round to the thumb, or curled into a fist |
    | G vs L | Thumb parallel to the index, or at 90° |
  - **Orientation:** the index direction and hand direction are measured as an *unsigned* angle
    from screen-up (0° up, 90° sideways either way, 180° down), so they work for both hands and
    when mirrored. U/H, K/P and G/Q share a handshape and are told apart by this angle alone.
    "Palm facing" (how directly the palm faces the camera, 0–1) is shown in the debug panel.
  - **Finger bend** is the angle between a finger's first bone and the line from its PIP joint
    to the tip. An earlier version summed the two small joint angles near the tip. A noise
    stress test showed that was far too sensitive to landmark jitter, and accuracy under noise
    went from 24% to 72% after the change.
  - **Synthetic tests:** all 19 letters, each upright and tilted ±20°, plus mirrored, plus
    4 rejection poses (122 checks), all pass. With random landmark noise added, mean accuracy is
    99% at moderate noise and 72% at harsh noise. Under harsh noise almost every miss is
    "no letter" rather than a wrong letter (1.4% wrong). The weakest letters under noise are
    B, F, O, W and H.
  - **Left/right hands and mirroring:** "which side is the thumb on" is measured in a
    hand-local frame (pinky knuckle → index knuckle), not in screen coordinates, and angles
    don't change when an image is mirrored. So the same rules work for both hands.
  - All thresholds are in the `CFG` object at the top of `sign/rules.js`. They can also be
    tweaked live via `window.SIGN_CONFIG` in the browser console while watching the debug panel.
- **Training data and preprocessing (`sign/features.js`, `sign/dataset.js`):**
  - A recorded sample stores the **raw** 21 landmarks plus its letter, signer and handedness,
    not a finished feature vector. Preprocessing runs when training, so it can be improved later
    without throwing away recordings.
  - `toVector()` is the **one** preprocessing function, used both when training and when
    predicting live. Using two different versions would give "training/serving skew": the
    model would be tested on a different kind of input than it learned from.
    1. **Translate:** the wrist is moved to the origin, so it doesn't matter where the hand is in the frame.
    2. **Scale:** divide by hand size (wrist → middle knuckle), so distance from the camera doesn't matter.
    3. **Mirror:** left hands are flipped (`x → −x`) into right-hand coordinates, so one model serves both hands.
    4. **No rotation normalisation,** on purpose: G/H point sideways and P/Q point down, so
       orientation is real information.
    5. **Vector** = 21 landmarks × (x, y, z) = 63 numbers, plus 22 engineered measurements
       (finger extension and bend, thumb position, gaps, spreads, orientation) = **85 numbers**.
       Angles are divided by 90 or 180 so every value is on a similar scale.
- **Mirroring:** the video is mirrored with CSS. The canvases aren't; instead `x` is flipped in
  code (`1 - x`) so text stays readable. MediaPipe's Left/Right labels assume a mirrored input,
  so they're swapped back (`SWAP_HANDEDNESS` in `main.js`).
- **Performance:** the camera runs at 640×480, and the model only runs when there's a new video frame.
  It uses the GPU delegate and falls back to CPU if that fails. The active delegate is logged to the console.
  An **adaptive effects budget** watches the render rate. If it drops below 45 fps, particle
  trails and bursts are scaled back (down to 25%), and they slowly return when it's smooth again.
  When the budget is active, the FPS pill shows `FX nn%`.

## Files

| File | What's in it |
| --- | --- |
| `index.html` | Layout, mode tabs and controls |
| `style.css` | Dark UI |
| `main.js` | Camera, model, the shared tracking loop, skeleton, trails, mode switching |
| `gestures.js` | Landmark map, finger-extended detection, pose helpers, One Euro filter, debouncer |
| `particles.js` | Allocation-free glowing particle pool |
| `modes/draw.js` | Draw mode: strokes, undo/clear, DRAW/HOVER/IDLE state machine |
| `modes/game.js` | Game mode: Bubble Pop (states, spawning, scoring, combo, high score) |
| `modes/sign.js` | Sign mode UI: Spell / Practice / Train sub-modes, hold-to-type, recording, debug panel |
| `sign/features.js` | Measurements + the shared preprocessing `toVector()` (used by rules, recording and models) |
| `sign/rules.js` | Rule-based classifier: `CFG` thresholds and fuzzy letter rules |
| `sign/letters.js` | Letter sets (19 rule letters, 24 static letters) and handshape descriptions |
| `sign/dataset.js` | Recorded samples: IndexedDB storage, JSON export/import, counts |
| `tests/sign-rules.html` | Synthetic tests for the rules (clean, tilted, mirrored, noisy) |
| `tests/sign-data.html` | Tests for preprocessing invariances and the dataset store (separate test database) |
| `tests/synthetic-hands.js` | Tiny 3D hand model used by the tests |
| `serve.ps1` | Tiny static server for Windows |

Every mode exports the same interface (`init`, `enter`, `exit`, `update`, `render`,
`handLost`, `trailColor`, `labelSuffix`, `badge`, `onKey`, optional `onKeyUp`). `main.js`
only calls the **active** mode, so Draw poses can't draw while you're in Game or Sign mode.

**Tests:** with the server running, open `http://localhost:8000/tests/sign-rules.html` or
`http://localhost:8000/tests/sign-data.html`. Results appear on the page.
