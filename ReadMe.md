# Air Draw: Hand Tracking in the Browser

Real-time hand tracking with MediaPipe's `HandLandmarker` (`@mediapipe/tasks-vision`).
It draws the 21-point hand skeleton for up to 2 hands, and you can **air draw** with your
index finger while you pinch your thumb and index finger together.

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

| Action | How |
| --- | --- |
| Draw | Pinch thumb tip (4) + index tip (8), then move |
| Lift pen | Release the pinch |
| Undo last stroke | `U`, `Ctrl+Z`, or the Undo button |
| Clear | `C` or the Clear button |
| Palm clear (off by default) | Tick "Palm clear", then hold a wide-open, spread hand still for 1.5 s |
| Color | Click a swatch or press `1` to `6` |
| Brush size | Slider, or `[` / `]` |
| Smoothing | "Smoothing" menu: One Euro `min cutoff` and `beta` |
| Toggle skeleton | `S` |

The top-right corner shows the active delegate (hover it to see the WebGL renderer),
the camera resolution, model inference time and FPS.

## How it works (short version)

- **Landmarks:** 0 = wrist. Each finger has 4 points going outwards, and the tip is always
  the last: thumb 1–4, index 5–8, middle 9–12, ring 13–16, pinky 17–20. There's a diagram at
  the top of `main.js`.
- **Pinch:** `distance(4, 8) / distance(0, 9)`. Dividing by the wrist-to-middle-knuckle length
  makes it independent of how far you are from the camera. The pen goes down below `0.3` for
  2 frames in a row, and lifts above `0.5` for 3 frames in a row (hysteresis + debounce).
- **Strokes:** each stroke is stored as `{ color, size, points }` in an array. The drawing
  canvas is a separate layer that's never cleared per frame. Undo pops the last stroke and
  redraws the rest.
- **Smoothing:** a One Euro filter on the fingertip (steady when slow, responsive when fast).
  Strokes are drawn as quadratic Bezier curves through the midpoints of consecutive points,
  and big jumps between frames are filled in with interpolated points.
- **Mirroring:** the video is mirrored with CSS. The canvases aren't; instead `x` is flipped in
  code (`1 - x`) so text stays readable. MediaPipe's Left/Right labels assume a mirrored input,
  so they're swapped back (`SWAP_HANDEDNESS` in `main.js`).
- **Performance:** the camera runs at 640×480, and the model only runs when there's a new video frame.
  It uses the GPU delegate and falls back to CPU if that fails. The active delegate is logged to the console.

## Files

- `index.html`: layout and controls
- `style.css`: dark UI
- `main.js`: all the tracking, gesture and drawing logic (heavily commented)
- `serve.ps1`: tiny static server for Windows
