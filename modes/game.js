// =============================================================================
//  modes/game.js: GAME mode, Bubble Pop
// =============================================================================
//
//  Bubbles float up from the bottom. Touch one with an index fingertip (either
//  hand, any pose) to pop it. 60-second rounds.
//
//  Game states:
//    READY      start screen, hold an open palm 1 s (or press Space)
//    COUNTDOWN  3-2-1
//    PLAYING    60 s round
//    OVER       final score, open palm again (or Space) to restart
//
//  Scoring:
//    normal bubble  10-20 pts (smaller = more), x combo multiplier
//    gold ★         50 pts, x combo multiplier
//    bomb 💣        -30 pts and the combo resets
//    combo          each pop within `comboWindow` s of the previous one adds
//                   +1 to the multiplier (capped at x5)
//
//  Performance: bubbles and score popups come from fixed pools, and each
//  bubble look is pre-rendered once to a sprite (no gradients per frame).
// =============================================================================

import { MIDDLE_MCP } from "../gestures.js";

const CONFIG = {
  roundSeconds: 60,
  countdownSeconds: 3,
  palmHoldMs: 1000,        // open palm hold to start
  restartDelayMs: 1200,    // after game over, ignore the palm for this long
  comboWindow: 1.0,        // seconds between pops to keep the combo going
  maxMultiplier: 5,
  minRadius: 0.03,         // bubble radius, fraction of canvas width
  maxRadius: 0.07,
  touchMargin: 0.006,      // extra touch tolerance, fraction of canvas width
  // Difficulty ramps linearly from *Start to *End over the round:
  spawnIntervalStart: 0.85, spawnIntervalEnd: 0.3,  // seconds between spawns
  speedStart: 0.13, speedEnd: 0.32,                 // canvas heights per second
  maxBubblesStart: 6, maxBubblesEnd: 16,            // on screen at once
  bombChanceStart: 0.06, bombChanceEnd: 0.16,
  goldChance: 0.06,
  points: { small: 20, large: 10, gold: 50, bomb: -30 },
};

const POOL_SIZE = 32;
const POPUP_POOL_SIZE = 16;
const HIGH_SCORE_KEY = "handlab.bubblePop.highScore";
const BUBBLE_COLORS = ["#38bdf8", "#a78bfa", "#34d399", "#f472b6"];
const GOLD = "#facc15";
const BOMB = "#ef4444";

let app = null;

// ---- Game state ----
let state = "READY";
let stateStart = 0;      // performance.now() when the state began
let lastNow = 0;
let gameTime = 0;        // seconds played this round
let spawnTimer = 0;
let score = 0;
let best = 0;
let newBest = false;
let combo = 0;           // current multiplier (0 = no combo running)
let lastPopTime = -10;
let flash = 0;           // red screen flash after a bomb (seconds left)
let palmStart = -1;
let palmArmed = true;
let palmCenter = null;   // landmark the hold ring is drawn on
let palmProgress = -1;

// ---- Pools (allocated once) ----
const bubbles = Array.from({ length: POOL_SIZE }, () => ({
  alive: false, x: 0, y: 0, baseX: 0, r: 0, vy: 0,
  phase: 0, wobbleAmp: 0, wobbleFreq: 0, kind: "normal", color: "", sprite: null,
}));
const popups = Array.from({ length: POPUP_POOL_SIZE }, () => ({
  alive: false, x: 0, y: 0, r: 0, text: "", color: "", life: 0, max: 0,
}));
const sprites = {};
let popupScale = 0;
let popupFont = "";

// ---- DOM (HUD + screens) ----
let hudScore, hudTime, hudBest, hudCombo, hudComboText, hudComboBar, hudTimeBox;
let screenEl, screenTitle, screenText;
const shown = { score: -1, time: -1, best: -1, combo: -1, screen: "" };

// =============================================================================
//  Helpers
// =============================================================================

const lerp = (a, b, t) => a + (b - a) * t;
const rand = (a, b) => a + Math.random() * (b - a);

function loadBest() {
  try {
    return Number(localStorage.getItem(HIGH_SCORE_KEY)) || 0;
  } catch {
    return 0; // storage blocked (private mode etc.): just start at 0
  }
}

function saveBest(value) {
  try {
    localStorage.setItem(HIGH_SCORE_KEY, String(value));
  } catch {
    // ignore: the high score just won't persist
  }
}

// Distance from point p to the segment a-b. Testing against the path the
// fingertip moved this frame (not just where it ended up) means a fast
// swipe still pops bubbles it passed through between camera frames.
function distToSegment(p, a, b) {
  const dx = b.x - a.x, dy = b.y - a.y;
  const len2 = dx * dx + dy * dy;
  let t = len2 ? ((p.x - a.x) * dx + (p.y - a.y) * dy) / len2 : 0;
  t = Math.max(0, Math.min(1, t));
  return Math.hypot(p.x - (a.x + dx * t), p.y - (a.y + dy * t));
}

// =============================================================================
//  Sprites: each bubble look is drawn ONCE to a 128px offscreen canvas
// =============================================================================

function makeSprite(color, kind) {
  const S = 128, h = S / 2;
  const c = document.createElement("canvas");
  c.width = c.height = S;
  const g = c.getContext("2d");

  g.beginPath();
  g.arc(h, h, h - 3, 0, Math.PI * 2);
  if (kind === "normal") {
    // Soap bubble: see-through middle, colored rim
    const grad = g.createRadialGradient(h * 0.75, h * 0.7, h * 0.1, h, h, h);
    grad.addColorStop(0, color + "14");
    grad.addColorStop(0.7, color + "44");
    grad.addColorStop(0.93, color + "cc");
    grad.addColorStop(1, color + "00");
    g.fillStyle = grad;
    g.fill();
    g.lineWidth = 3;
    g.strokeStyle = color + "ee";
    g.stroke();
  } else {
    const grad = g.createRadialGradient(h * 0.7, h * 0.65, h * 0.1, h, h, h);
    if (kind === "gold") {
      grad.addColorStop(0, "#fff7cc");
      grad.addColorStop(0.55, GOLD);
      grad.addColorStop(1, "#a16207");
    } else {
      grad.addColorStop(0, "#fca5a5");
      grad.addColorStop(0.55, "#dc2626");
      grad.addColorStop(1, "#450a0a");
    }
    g.fillStyle = grad;
    g.fill();
    g.fillStyle = "#fff";
    g.textAlign = "center";
    g.textBaseline = "middle";
    g.font = kind === "gold" ? "bold 64px system-ui, sans-serif" : "56px 'Segoe UI Emoji', 'Apple Color Emoji', sans-serif";
    g.fillText(kind === "gold" ? "★" : "💣", h, h + 4);
  }

  // Specular highlight (upper-left shine)
  g.save();
  g.translate(h * 0.62, h * 0.55);
  g.rotate(-0.6);
  g.beginPath();
  g.ellipse(0, 0, h * 0.22, h * 0.12, 0, 0, Math.PI * 2);
  g.fillStyle = "rgba(255,255,255,0.6)";
  g.fill();
  g.restore();
  return c;
}

// =============================================================================
//  State changes
// =============================================================================

function resetRound() {
  for (const b of bubbles) b.alive = false;
  for (const p of popups) p.alive = false;
  score = 0;
  combo = 0;
  gameTime = 0;
  spawnTimer = 0.4;
  lastPopTime = -10;
  flash = 0;
  newBest = false;
}

function setState(next, now) {
  state = next;
  stateStart = now;
}

function startCountdown(now) {
  resetRound();
  palmArmed = false;
  palmStart = -1;
  setState("COUNTDOWN", now);
}

function endGame(now) {
  if (score > best) {
    best = score;
    newBest = true;
    saveBest(best);
  }
  combo = 0;
  palmArmed = false; // the palm has to drop before it can restart
  setState("OVER", now);
}

// =============================================================================
//  Bubbles
// =============================================================================

function spawnBubble(d) {
  const b = bubbles.find((x) => !x.alive);
  if (!b) return;
  const W = app.overlay.width, H = app.overlay.height;
  const roll = Math.random();
  const bombChance = lerp(CONFIG.bombChanceStart, CONFIG.bombChanceEnd, d);
  b.kind = roll < CONFIG.goldChance ? "gold" : roll < CONFIG.goldChance + bombChance ? "bomb" : "normal";
  b.color = b.kind === "gold" ? GOLD : b.kind === "bomb" ? BOMB
    : BUBBLE_COLORS[(Math.random() * BUBBLE_COLORS.length) | 0];
  b.sprite = b.kind === "normal" ? sprites[b.color] : sprites[b.kind];
  b.r = rand(CONFIG.minRadius, CONFIG.maxRadius) * W;
  if (b.kind !== "normal") b.r = Math.max(b.r, CONFIG.minRadius * 1.4 * W); // keep specials readable
  b.baseX = rand(b.r, W - b.r);
  b.x = b.baseX;
  b.y = H + b.r;
  b.vy = -H * lerp(CONFIG.speedStart, CONFIG.speedEnd, d) * rand(0.8, 1.25);
  b.phase = rand(0, Math.PI * 2);
  b.wobbleAmp = rand(0.01, 0.03) * W;
  b.wobbleFreq = rand(1, 2.5);
  b.alive = true;
}

function addPopup(x, y, r, text, color) {
  const p = popups.find((q) => !q.alive) || popups[0];
  p.alive = true;
  p.x = x; p.y = y; p.r = r;
  p.text = text;
  p.color = color;
  p.life = p.max = 0.8;
}

function pop(b) {
  b.alive = false;
  const s = app.pxScale();
  const H = app.overlay.height;

  if (b.kind === "bomb") {
    score = Math.max(0, score + CONFIG.points.bomb);
    combo = 0;
    flash = 0.35;
    addPopup(b.x, b.y, b.r, `${CONFIG.points.bomb}`, BOMB);
    app.particles.burst(b.x, b.y, BOMB, Math.ceil(30 * app.fxQuality()), H * 0.7, 5 * s, 0.7);
    return;
  }

  // Combo: a pop soon after the previous one raises the multiplier.
  combo = gameTime - lastPopTime <= CONFIG.comboWindow ? Math.min(combo + 1, CONFIG.maxMultiplier) : 1;
  lastPopTime = gameTime;

  let pts;
  if (b.kind === "gold") pts = CONFIG.points.gold;
  else {
    // Smaller bubbles are harder to hit, so they're worth more.
    const W = app.overlay.width;
    const t = (CONFIG.maxRadius * W - b.r) / ((CONFIG.maxRadius - CONFIG.minRadius) * W);
    pts = Math.round(lerp(CONFIG.points.large, CONFIG.points.small, Math.max(0, Math.min(1, t))));
  }
  pts *= combo;
  score += pts;
  addPopup(b.x, b.y, b.r, combo > 1 ? `+${pts} x${combo}` : `+${pts}`, b.color);
  // Burst size follows the adaptive effects budget (see main.js).
  const n = Math.ceil((b.kind === "gold" ? 28 : 16) * app.fxQuality());
  app.particles.burst(b.x, b.y, b.color, n, H * 0.5, 4 * s, 0.6);
}

function updateBubbles(dt, hands, canPop) {
  const H = app.overlay.height;
  const margin = CONFIG.touchMargin * app.overlay.width;
  for (const b of bubbles) {
    if (!b.alive) continue;
    b.y += b.vy * dt;
    b.phase += b.wobbleFreq * dt;
    b.x = b.baseX + Math.sin(b.phase) * b.wobbleAmp;
    if (b.y < -b.r) {
      b.alive = false; // floated away, no penalty
      continue;
    }
    if (!canPop) continue;
    // Any visible hand's index fingertip can pop it.
    for (const hand of hands) {
      if (distToSegment(b, hand.prevTip, hand.tip) < b.r + margin) {
        pop(b);
        break;
      }
    }
  }
}

// =============================================================================
//  Open palm to start / restart
// =============================================================================
// Here "open palm" just means all 5 fingers extended (from gestures.js). It's
// looser than Draw mode's strict palm-clear, because starting a game by
// accident does no harm.
function updatePalm(hands, now) {
  palmProgress = -1;
  const canStart = state === "READY" || (state === "OVER" && now - stateStart > CONFIG.restartDelayMs);
  if (!canStart) return;

  const open = hands.find((h) => h.fingers.count === 5);
  if (!open) {
    palmStart = -1;
    palmArmed = true;
    return;
  }
  if (!palmArmed) return;
  if (palmStart < 0) palmStart = now;
  palmCenter = open.pts[MIDDLE_MCP];
  palmProgress = (now - palmStart) / CONFIG.palmHoldMs;
  if (palmProgress >= 1) {
    palmProgress = -1;
    startCountdown(now);
  }
}

// =============================================================================
//  HUD + screens (DOM is only touched when a value changes)
// =============================================================================

function setScreen(key, title, html) {
  if (shown.screen === key) return;
  shown.screen = key;
  screenEl.classList.toggle("show", key !== "");
  screenTitle.textContent = title;
  screenText.innerHTML = html;
}

function updateHud(now) {
  const timeLeft = state === "PLAYING" ? Math.ceil(CONFIG.roundSeconds - gameTime)
    : state === "OVER" ? 0 : CONFIG.roundSeconds;
  if (shown.score !== score) hudScore.textContent = shown.score = score;
  if (shown.time !== timeLeft) {
    hudTime.textContent = shown.time = timeLeft;
    hudTimeBox.classList.toggle("urgent", state === "PLAYING" && timeLeft <= 10);
  }
  if (shown.best !== best) hudBest.textContent = shown.best = best;

  const showCombo = state === "PLAYING" && combo > 1;
  if (shown.combo !== (showCombo ? combo : 0)) {
    shown.combo = showCombo ? combo : 0;
    hudCombo.classList.toggle("show", showCombo);
    hudComboText.textContent = `x${combo}`;
  }
  if (showCombo) {
    const left = 1 - (gameTime - lastPopTime) / CONFIG.comboWindow;
    hudComboBar.style.transform = `scaleX(${Math.max(0, left)})`;
  }

  const palmHint = "Hold an <b>open palm ✋</b> for 1 second to start, or press <kbd>Space</kbd>";
  if (state === "READY") {
    setScreen("ready", "Bubble Pop",
      `${palmHint}<br><span class="muted">Pop bubbles with your index fingertip · ★ = +50 · 💣 = −30 · quick pops build a combo</span>`);
  } else if (state === "COUNTDOWN") {
    const n = CONFIG.countdownSeconds - Math.floor((now - stateStart) / 1000);
    setScreen(`cd${n}`, String(Math.max(n, 1)), "Get ready!");
  } else if (state === "PLAYING") {
    if (gameTime < 0.6) setScreen("go", "GO!", "");
    else setScreen("", "", "");
  } else if (state === "OVER") {
    setScreen(`over${score}`, "Game over",
      `Score <b>${score}</b> · ${newBest ? "🎉 <b>New high score!</b>" : `Best ${best}`}<br>` +
        `<span class="muted">Open palm ✋ or <kbd>Space</kbd> to play again</span>`);
  }
}

// =============================================================================
//  Mode interface (see modes/draw.js for what each method is for)
// =============================================================================

export const gameMode = {
  id: "game",
  name: "Game",

  init(ctx) {
    app = ctx;
    for (const c of BUBBLE_COLORS) sprites[c] = makeSprite(c, "normal");
    sprites.gold = makeSprite(GOLD, "gold");
    sprites.bomb = makeSprite(BOMB, "bomb");
    best = loadBest();

    const $ = (id) => document.getElementById(id);
    hudScore = $("hudScore");
    hudTime = $("hudTime");
    hudTimeBox = $("hudTimeBox");
    hudBest = $("hudBest");
    hudCombo = $("hudCombo");
    hudComboText = $("hudComboText");
    hudComboBar = $("hudComboBar");
    screenEl = $("gameScreen");
    screenTitle = $("gameScreenTitle");
    screenText = $("gameScreenText");
    $("gameStartBtn").addEventListener("click", () => {
      if (state === "READY" || state === "OVER") startCountdown(performance.now());
    });
  },

  enter() {
    resetRound();
    setState("READY", performance.now());
    lastNow = 0;
    palmStart = -1;
    palmArmed = true;
    shown.screen = "-"; // force a screen refresh
    updateHud(performance.now());
  },

  // Leaving the mode abandons the round.
  exit() {
    resetRound();
    setState("READY", 0);
    setScreen("", "", "");
  },

  update(hands, now) {
    const dt = lastNow ? Math.min((now - lastNow) / 1000, 0.1) : 0;
    lastNow = now;

    updatePalm(hands, now);

    if (state === "COUNTDOWN" && now - stateStart >= CONFIG.countdownSeconds * 1000) {
      setState("PLAYING", now);
    }

    if (state === "PLAYING") {
      gameTime += dt;
      const d = Math.min(gameTime / CONFIG.roundSeconds, 1); // difficulty 0 -> 1
      spawnTimer -= dt;
      let alive = 0;
      for (const b of bubbles) alive += b.alive;
      if (spawnTimer <= 0 && alive < Math.round(lerp(CONFIG.maxBubblesStart, CONFIG.maxBubblesEnd, d))) {
        spawnBubble(d);
        spawnTimer = lerp(CONFIG.spawnIntervalStart, CONFIG.spawnIntervalEnd, d) * rand(0.7, 1.3);
      }
      if (combo > 0 && gameTime - lastPopTime > CONFIG.comboWindow) combo = 0;
      updateBubbles(dt, hands, true);
      if (gameTime >= CONFIG.roundSeconds) endGame(now);
    } else {
      // Leftover bubbles drift away after the round, but can't be popped.
      updateBubbles(dt, hands, false);
    }

    for (const p of popups) if (p.alive && (p.life -= dt) <= 0) p.alive = false;
    flash = Math.max(0, flash - dt);
    updateHud(now);
  },

  render(ctx, hands, now) {
    const s = app.pxScale();
    if (s !== popupScale) {
      popupScale = s;
      popupFont = `800 ${18 * s}px system-ui, sans-serif`; // rebuilt only on resize
    }

    for (const b of bubbles) {
      if (!b.alive) continue;
      // Bombs pulse a little so they read as "danger".
      const r = b.kind === "bomb" ? b.r * (1 + 0.06 * Math.sin(now / 90)) : b.r;
      ctx.drawImage(b.sprite, b.x - r, b.y - r, r * 2, r * 2);
    }

    // Pop feedback: an expanding ring + floating score text.
    ctx.textAlign = "center";
    ctx.textBaseline = "middle";
    ctx.font = popupFont; // set once: parsing a font string per popup is wasteful
    for (const p of popups) {
      if (!p.alive) continue;
      const t = 1 - p.life / p.max; // 0 -> 1
      ctx.globalAlpha = 1 - t;
      if (t < 0.4) {
        ctx.beginPath();
        ctx.arc(p.x, p.y, p.r * (1 + t * 1.5), 0, Math.PI * 2);
        ctx.lineWidth = 3 * s;
        ctx.strokeStyle = p.color;
        ctx.stroke();
      }
      ctx.fillStyle = p.color;
      ctx.fillText(p.text, p.x, p.y - t * 40 * s);
    }
    ctx.globalAlpha = 1;

    if (flash > 0) {
      ctx.fillStyle = `rgba(239,68,68,${flash})`;
      ctx.fillRect(0, 0, app.overlay.width, app.overlay.height);
    }

    // Fingertip markers, so you can see exactly what pops bubbles.
    for (const hand of hands) {
      ctx.beginPath();
      ctx.arc(hand.tip.x, hand.tip.y, 7 * s, 0, Math.PI * 2);
      ctx.fillStyle = "rgba(255,255,255,0.9)";
      ctx.fill();
      ctx.lineWidth = 2 * s;
      ctx.strokeStyle = app.handColors[hand.label];
      ctx.stroke();
    }

    if (palmProgress >= 0 && palmCenter) {
      app.drawHoldRing(palmCenter, palmProgress, state === "OVER" ? "AGAIN" : "START", "hold open", "#a3e635");
    }
  },

  handLost() {},
  trailColor(hand) { return app.handColors[hand.label]; },
  labelSuffix() { return ""; },

  badge() {
    if (state === "PLAYING") return { text: "PLAYING", tone: "DRAW" };
    if (state === "COUNTDOWN") return { text: "GET READY", tone: "HOVER" };
    if (state === "OVER") return { text: "GAME OVER", tone: "NONE" };
    return { text: "READY", tone: "NONE" };
  },

  onKey(e) {
    if (e.code === "Space" && !e.ctrlKey && !e.metaKey && !e.altKey) {
      e.preventDefault(); // don't "click" a focused button
      if (state === "READY" || state === "OVER") startCountdown(performance.now());
      return true;
    }
    return false;
  },
};
