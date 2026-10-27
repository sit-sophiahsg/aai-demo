# Hand Lab: Hand Tracking in the Browser

Real-time hand tracking with MediaPipe's `HandLandmarker` (`@mediapipe/tasks-vision`).
It tracks up to 2 hands, draws the 21-point skeleton, and has three modes:

- **Draw:** air draw with your index finger.
- **Game:** Bubble Pop, popping bubbles with your fingertips.
- **Sign:** ASL fingerspelling for the full alphabet: 24 static letters plus the moving letters
  J and Z. It has Spell, Practice and Train sub-modes, three classifiers (Rules, k-NN, MLP),
  and an evaluation view.

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

## The interface

- **Header:**
  - Left: the app name.
  - Centre: the mode tabs (Draw / Game / Sign, keys `1` `2` `3`).
  - Right: one **status chip**, e.g. `34 fps · GPU · 640×480`. The dot is green at 25 fps or
    more and amber below. `FX nn%` appears in the chip while particle effects are being reduced.
    Hover the chip for the model's inference time and the WebGL renderer.
  - The `?` button (or the `?` key) opens a help window with instructions for each mode.
  - The full-screen button (or `F`) makes the camera fill the window in every mode.
- **Video:** in the top corners are a live **state chip** (Draw: `DRAWING` / `HOVER` / `IDLE`;
  Game: the game state; Sign: `LETTER A`, or `DRAWING J` / `DRAWING Z`; `NO HAND` when no hand
  is visible), the detected hand and a `Mirrored` chip. With no hand in view, a short hint in
  the middle explains how to position it.
- **Gesture legend:** below the video, the main gestures for the current mode.
- **Side panel:** the controls for the current mode. On narrow screens it moves below the video.
- **Design:** colours, fonts (Sora for UI text, JetBrains Mono for numbers, keys and labels)
  and radii are CSS variables at the top of `style.css`. Icons are inline SVG.

## Controls

### Everywhere

| Action | How |
| --- | --- |
| Switch mode | `1` Draw · `2` Game · `3` Sign, or the tabs in the header |
| Particle trails on/off | `T` (in Draw and Game), or the **Particle trails** switch in the Draw panel; the slider under it sets intensity |
| Toggle skeleton | `S`, or the **Hand skeleton** switch in the Draw panel |
| Smoothing | Draw panel → **Advanced**: One Euro `min cutoff` and `beta` |

### Draw mode

| Action | How |
| --- | --- |
| **DRAW** | Only the index finger extended: the fingertip draws |
| **HOVER** | Index + middle extended: the cursor moves, pen up (use it between letters) |
| **IDLE** | Any other pose (fist, open palm…): nothing happens |
| Undo last stroke | `U`, `Ctrl+Z`, or the Undo button |
| Clear | `C` or the Clear button |
| Palm clear (off by default) | Turn on **Palm-hold clears canvas**, then hold a wide-open, spread hand still for 1.5 s |
| Color | Click a swatch or press `Shift+1` to `Shift+6` |
| Brush size | Slider, or `[` / `]` |

### Game mode: Bubble Pop

| Action | How |
| --- | --- |
| Start / restart | Hold an **open palm** (all 5 fingers out) for 1 s, or press `Space` / "Start round" |
| Pop a bubble | Touch it with an **index fingertip**. Either hand and any pose works |
| Scoring | Normal bubbles 10–20 (smaller = more) · **gold** +50 · **bomb** −30, and the combo resets |
| Combo | Each pop within 1 s of the last raises the multiplier (x2, x3… up to x5) |

- Rounds last 60 s. Bubbles get faster and more numerous over the round, and bombs get more common.
- The high score is saved in your browser (`localStorage`).
- After game over, put your hand down before holding your palm up again, so you can't restart by accident.
- Switching away from Game mode abandons the current round.
- During a round, score, time and combo appear as three cards at the top of the video. The best
  score is in the side panel.

### Sign mode: ASL fingerspelling

Either hand works.
- **Rules classifier:** A B C D F G H I K L O P Q R U V W X Y. E, M, N, S and T differ only in
  where the thumb hides among curled fingers, which is too subtle for hand-written rules.
- **k-NN / MLP classifiers:** all 24 static letters, learned from your recordings (see Train below).
- **J and Z (moving letters):** recognised from the fingertip's path, with any classifier.
  - **J:** make an I (little finger up) and draw a hook with the little finger: down, then
    sideways.
  - **Z:** point with your index finger and draw a Z: across, diagonally down and back, across.
  - Start moving within ~0.8 s of forming the handshape, otherwise the static I or D gets typed
    first. While you're drawing, the path is shown in yellow and the state chip says `DRAWING J` / `DRAWING Z`.
- Most letters are signed with the fingers up.
- **G and H point sideways**, and **P and Q point down**. These pairs have the same handshape
  (G/Q, H/U, K/P), so the classifier also looks at which way your hand points on screen.

| Action | How |
| --- | --- |
| Type a letter | Hold the handshape steady for ~0.8 s. The thin inner ring fills, then the letter is added |
| Same letter twice | Change pose or drop your hand briefly between them |
| Space / delete / clear | `Space` · `Backspace` · `C`, or the buttons in the **Typed text** card |
| Practice | `P` or "Practice". Shows a target letter with a description, shows a tick when you match it, then moves on. `N` skips |
| Train | `T` or "Train". Records labelled samples for the learned classifier (see below). In Sign mode `T` means Train; switch trails off in the Draw panel instead |
| Debug panel | `D`, or the **Debug panel** switch. Shows each finger's ratio, bend angle and state (extended / curved / curled), thumb position, key gaps and angles, and the top letter scores |

The letter ring has two arcs:
- **Outer arc:** the classifier's confidence. Grey means no letter, amber means it's shown but
  not sure enough to type, and green means it's confident enough to type.
- **Inner arc:** fills while you hold the letter.

Below the ring are the confidence %, the active classifier, and the Rules / k-NN / MLP switch.

#### Train: collecting data for the learned classifier

Press `T` (or "Train") in Sign mode. The Train card replaces the letter ring and the typed
text in the side panel.

1. Type your name in **Signer**. It's stored with every sample, which later lets us test how
   well the model works for people it hasn't seen.
2. Pick a letter in the grid (or use `←` / `→`). All **24 static letters** (A–Y without J)
   are there, including E, M, N, S and T, which the rules can't do.
3. Make the sign and **hold `R`** (or hold the Record button). About 10 samples per second are
   recorded while it's held. While recording, the video gets a red frame and the state chip shows `REC`.
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

#### Classifiers and evaluation

| Action | How |
| --- | --- |
| Choose the classifier | The **Rules / k-NN / MLP** switch under the letter ring (remembered between visits) |
| k-NN settings | Train panel → Models: `k` and distance (Euclidean / cosine). Takes effect immediately; nothing to train |
| Train the MLP | Train panel → Models → **Train MLP**. Shows loss and accuracy per epoch, then saves the model in the browser, so it's there after a reload |
| Compare all three | **Evaluate classifiers** in the Sign panel (see the pipeline section below) |

Spell, Practice, the confidence ring and the debug panel work with all three. For k-NN and
MLP the debug panel shows the **top-4 class probabilities**. Practice offers the letters the
selected classifier knows: 19 for the rules, and every letter with 10+ samples for the models.

## The sign-language pipeline (data → preprocessing → models → evaluation)

### 1. Data collection (`modes/sign.js` Train, `sign/dataset.js`)
- **Labelled examples:** one sample = one video frame of one hand making a known letter. It
  stores the 21 raw landmarks, the letter (the label), the signer and the hand.
- **Rate and target:** ~10 samples per second while you hold `R`, aiming for 100 per letter
  across all 24 static letters.
- **Storage:** saved in IndexedDB. JSON export/import lets several people pool their data, and
  imports skip duplicates using a unique id per sample.
- **Signer name:** stored with every sample so we can later test on people the model hasn't seen.

### 2. Preprocessing (`sign/features.js` → `toVector()`)
The **same** function runs at training time and at prediction time. If those differed, you'd
get *training/serving skew*.
1. Translate (wrist → origin) and scale (÷ hand size), so the result doesn't depend on where
   your hand is or how far away it is.
2. Mirror left hands into right-hand coordinates, so one model serves both hands.
3. Keep orientation (no rotation normalisation), because G/H/P/Q depend on it.
4. 63 coordinates + 22 engineered measurements (finger extension and bend, thumb position,
   gaps, spreads, orientation) = **85 features**.
5. **Standardise** each feature (z-score: subtract the mean, divide by the standard deviation,
   both taken from the training data). Without this, features with large ranges would dominate
   k-NN distances. The MLP saves its mean/std with the model.

### 3. Models
| | Rules (`sign/rules.js`) | k-NN (`sign/knn.js`) | MLP (`sign/mlp.js`) |
| --- | --- | --- | --- |
| Idea | Hand-written fuzzy conditions per letter | Vote of the *k* most similar recorded samples | Neural network learned from the samples |
| Training | None (a human wrote them) | None: the data *is* the model | ~10–40 s in the browser (TensorFlow.js) |
| Letters | 19 | All 24 | All 24 |
| Confidence | Weakest condition's fuzzy score | Distance-weighted vote share | Softmax probability |
| Strength | Explainable, no data needed | Simple, adapts instantly to new data | Learns subtle shapes (E/M/N/S/T), compact |

- **k-NN:** standardised Euclidean or cosine distance to every sample. The *k* nearest vote,
  weighted by 1/distance, so closer neighbours count more.
- **MLP:** 85 → Dense 64 (ReLU) → Dropout 0.3 → Dense 64 (ReLU) → Dropout 0.3 → 24 softmax.
  It uses Adam (learning rate 0.003) and categorical cross-entropy. Data is shuffled *before*
  the 15% validation split, because TF.js takes the *last* rows as validation, and those would
  otherwise be just the last few letters.
- **CPU backend:** TF.js runs on the CPU on purpose. MediaPipe already uses the GPU, and this
  ~11k-weight network takes ~0.2 ms per prediction on the CPU.
- **Unknown-pose gate:** the models always pick *some* letter, even for a relaxed hand. If the
  current hand is much farther from every training sample than samples normally are from each
  other (2 × the 95th-percentile nearest-neighbour distance), confidence is scaled down.
  Without the gate, a hand at rest would type letters.
- **Thresholds per classifier** (fuzzy scores and probabilities aren't on the same scale):

  | Classifier | Shown at | Typed at |
  | --- | --- | --- |
  | Rules | 0.55 | 0.65 |
  | k-NN | 0.5 | 0.7 |
  | MLP | 0.5 | 0.8 |

### 4. Evaluation (`sign/evaluate.js`, **Evaluate** button)
- **80/20 split,** stratified so every letter keeps ~20% for testing. The split is seeded, so
  it's reproducible. k-NN and a *fresh* MLP see only the 80%, so evaluation never overwrites
  your deployed model.
  - **Random:** shuffle each letter's frames. Neighbouring frames are nearly identical, so test
    frames have near-twins in training and **accuracy is optimistic**.
  - **Time-blocked:** hold out the *last* 20% of each signer's recording of each letter. That's
    closer to "a new attempt" and is the more honest single-person estimate.
- **Metrics:**
  - Overall accuracy, and accuracy on the 19 rule letters. The second is fairer to the rules,
    which can't do E, M, N, S or T.
  - Per-letter accuracy for all three classifiers side by side.
  - A **confusion matrix** (rows = true letter, columns = prediction, ∅ = no confident answer)
    with a list of the most common mistakes.
- **Leave-one-signer-out** (2+ signers): train on everyone else, test on one person, repeat for
  each person. This measures **generalisation to new people**, usually the lowest and most
  important number. With only your own data, the model may be learning *your* hand.
- **What to expect:** random split > time-blocked > leave-one-signer-out. Big gaps between them
  are a sign of overfitting to individual recordings or people, and a good thing to discuss in
  a presentation.

### Testing the pipeline
`tests/sign-models.html` builds a synthetic dataset from a 3D hand model (19 letters × 2
"signers", with noise, tilt and both hands) and checks the split, k-NN, MLP training,
save/reload and the full evaluation (17 checks). On this idealised data all three score highly
(Rules ~72–95%, k-NN and MLP ~99–100%), so it shows the pipeline is *correct*. Real-world
accuracy must be measured on your recordings with **Evaluate**.

### 5. Moving letters J and Z (`sign/motion.js`)
- **Rolling buffer:** while the I handshape (for J) or the pointing handshape (for Z) is held,
  the last ~2 s of the little or index fingertip's positions are kept in a fixed ring buffer.
- **Every frame:**
  1. **Simplify** the path with the Ramer–Douglas–Peucker algorithm. Points within 0.1 hand
     sizes of a straight line are dropped, so jitter disappears and a Z becomes ~4 corner points
     whatever the drawing speed.
  2. **Merge** neighbouring strokes that point the same way.
  3. **Match** the latest strokes:
     - **Z** = across · diagonally down-and-back · across in the first stroke's direction.
     - **J** = a downward stroke, then a sideways hook that turns at least 40°.

  Everything is in hand sizes, and Z/J are accepted in either horizontal direction, so it works
  for both hands without trusting the Left/Right label. All thresholds are in `MOTION_CFG`.
- **Why path checks rather than dynamic time warping (DTW):**
  - DTW compares the motion with recorded templates, so it's only as good as its templates
    (ideally per person). It also needs the gesture's start and end cut out.
  - J and Z have very distinctive shapes, so rules written from the letter's definition work
    for anyone with no templates, ignore drawing speed, and are easy to read and tune.
  - Trade-off: a very sloppy J/Z may fail a strict check, whereas DTW could learn a person's style.
- **Look-alike protection:** I has the same handshape as J, and pointing/D the same as Z. While
  the fingertip is moving in one of those handshapes (and for ~0.45 s after), static letters
  can't be typed. After a J or Z, the static letter for the pose you're still holding isn't
  typed until you change pose.
- **Tests:** `tests/sign-motion.html` (28 checks).
  - J and Z must be recognised in both directions, at slow and fast speeds, and near or far
    from the camera.
  - These must **not** trigger: a still hand, a straight drop, a side-to-side wave, a circle, a
    Z drawn with an open hand.
  - In the real Sign mode, I and D must not type mid-motion.


## Known issues

- **Z is unreliable in live use.** It passes the synthetic tests, but with a real hand it often
  isn't recognised. The likely causes are the stroke-length and "sideways" thresholds in
  `MOTION_CFG` (`sign/motion.js`) being too strict for real, curved strokes, and the pointing
  handshape flickering mid-motion. The debug panel's *Motion* line shows which check fails.
  J works more reliably.
- **U / V are often confused** by the learned models with the current recordings. More varied
  samples of both should help.
- **P and X** are hard to trigger with the rules classifier (P is often read as V, X as C). The
  k-NN / MLP classifiers handle them better.

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
  When the budget is active, the header's status chip shows `FX nn%`.

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
| `sign/knn.js` | k-NN classifier, z-score scaler (shared with the MLP), unknown-pose gate |
| `sign/mlp.js` | TF.js MLP: lazy loading, training, save/load (IndexedDB), prediction |
| `sign/classifiers.js` | One `predict()` interface over Rules / k-NN / MLP, per-classifier thresholds, the deployed models |
| `sign/evaluate.js` | Splits (random / time-blocked), metrics, confusion matrix, leave-one-signer-out |
| `sign/eval-view.js` | The Evaluate window |
| `sign/charts.js` | Training-curve line charts and the confusion-matrix heatmap |
| `sign/motion.js` | J and Z: fingertip path buffer, RDP simplification, stroke pattern matching (`MOTION_CFG`) |
| `tests/sign-rules.html` | Synthetic tests for the rules (clean, tilted, mirrored, noisy) |
| `tests/sign-data.html` | Tests for preprocessing invariances and the dataset store (separate test database) |
| `tests/sign-models.html` | Tests for the split, k-NN, MLP training/save/reload and the evaluation pipeline |
| `tests/sign-motion.html` | Tests for J/Z recognition (positives, negatives, no I/D typed mid-motion) |
| `tests/synthetic-hands.js` | Tiny 3D hand model used by the tests |
| `serve.ps1` | Tiny static server for Windows |

Every mode exports the same interface (`init`, `enter`, `exit`, `update`, `render`,
`handLost`, `trailColor`, `labelSuffix`, `badge`, `onKey`, optional `onKeyUp`). `main.js`
only calls the **active** mode, so Draw poses can't draw while you're in Game or Sign mode.

**Tests:** with the server running, open `http://localhost:8000/tests/sign-rules.html`,
`http://localhost:8000/tests/sign-data.html`, `http://localhost:8000/tests/sign-models.html` or
`http://localhost:8000/tests/sign-motion.html`.
Results appear on the page. Keep the models test tab in the foreground while it runs:
browsers slow down timers in background tabs, which makes TF.js training crawl.
