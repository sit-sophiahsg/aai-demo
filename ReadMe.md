# Hand Lab: Hand Tracking in the Browser

Real-time hand tracking with MediaPipe's `HandLandmarker` (`@mediapipe/tasks-vision`).
It tracks up to 2 hands, draws the 21-point skeleton, and has three modes:

- **Draw:** air draw with your index finger.
- **Game:** Bubble Pop, popping bubbles with your fingertips.
- **Sign:** ASL fingerspelling (coming in Phase 3).

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
| `modes/sign.js` | Sign mode (Phase 3) |
| `serve.ps1` | Tiny static server for Windows |

Every mode exports the same interface (`init`, `enter`, `exit`, `update`, `render`,
`handLost`, `trailColor`, `labelSuffix`, `badge`, `onKey`). `main.js` only calls the
**active** mode, so Draw poses can't draw while you're in Game or Sign mode.
