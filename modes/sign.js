// =============================================================================
//  modes/sign.js: SIGN mode UI (ASL fingerspelling)
// =============================================================================
//
//  The recognition logic lives in the sign/ folder:
//    sign/features.js   measurements + the shared preprocessing (toVector)
//    sign/rules.js      the rule-based classifier (fuzzy rules + CFG)
//    sign/letters.js    letter sets + handshape descriptions
//    sign/dataset.js    recorded training samples (IndexedDB, export/import)
//
//  This file is the UI around them. Three sub-modes:
//    SPELL     hold a letter ~0.8 s to type it
//    PRACTICE  a target letter + description, ✓ when you match it
//    TRAIN     record labelled landmark samples for the learned models
// =============================================================================

import { Debouncer } from "../gestures.js";
import { makeFeatures, computeFeatures, packLandmarks } from "../sign/features.js";
import { CFG, RULE_LETTERS, scoreLetters } from "../sign/rules.js";
import { STATIC_LETTERS, DESCRIPTIONS } from "../sign/letters.js";
import * as dataset from "../sign/dataset.js";

const FINGER_NAMES = ["Index", "Middle", "Ring", "Pinky"];
const RECORD_INTERVAL_MS = 100;   // ~10 samples per second while recording
const TARGET_PER_LETTER = 100;
const SIGNER_KEY = "handlab.signer";

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

let sub = "spell";    // "spell" | "practice" | "train"
let text = "";
let practiceOrder = [];
let practiceIdx = 0;
let practiceTarget = "";
let practiceMatched = 0;
let practiceNextAt = 0; // after a match, advance at this time
let debugOn = false;
let debugTick = 0;

// Train state
let trainLetter = "A";
let recording = false;
let recordKeyDown = false;
let recordBtnDown = false;
let nextRecordAt = 0;
let takeCount = 0;      // samples recorded in the current press
let noHandWarned = false;

// DOM
let letterEl, letterCard, confBar, confText, holdRing, textEl, stageEl;
let practiceEl, practiceLetterEl, practiceDescEl, practiceScoreEl, practiceTickEl, practiceNote;
let debugEl, spellBtn, practiceBtn, trainBtn, nextBtn, debugToggle, spellControls;
let trainPanel, trainLetterEl, trainDescEl, trainGrid, trainStatus, trainSigner, trainSigners,
  trainTotal, trainStorage, trainDeleteBtn, trainRecordBtn, trainImportFile;
const gridButtons = {};
const shown = { letter: null, faded: null, text: null };

function handResult(label) {
  if (!perHand[label]) {
    perHand[label] = { feat: makeFeatures(), scores: Object.fromEntries(RULE_LETTERS.map((l) => [l, 0])), letter: "", conf: 0 };
  }
  return perHand[label];
}

function classify(r) {
  r.letter = "";
  r.conf = 0;
  for (const l of RULE_LETTERS) {
    if (r.scores[l] > r.conf) {
      r.conf = r.scores[l];
      r.letter = l;
    }
  }
  if (r.conf < CFG.minConfidence) r.letter = "";
}

// =============================================================================
//  Spell + Practice
// =============================================================================

function accept(letter, now) {
  letterCard.classList.remove("accepted");
  void letterCard.offsetWidth; // restart the CSS animation
  letterCard.classList.add("accepted");
  if (sub === "spell") {
    setText(text + letter);
  } else if (sub === "practice") {
    if (letter === practiceTarget && !practiceNextAt) {
      practiceMatched++;
      practiceTickEl.classList.add("show");
      practiceNote.textContent = "Nice! ✓";
      practiceScoreEl.textContent = `${practiceMatched} matched`;
      practiceNextAt = now + 1200;
    } else if (!practiceNextAt) {
      practiceNote.textContent = `That looked like “${letter}”. Check the description and try again.`;
    }
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

// Switch between the three sub-modes.
function setSub(next) {
  if (sub === "train" && next !== "train") stopRecording();
  sub = next;
  spellBtn.classList.toggle("active", sub === "spell");
  practiceBtn.classList.toggle("active", sub === "practice");
  trainBtn.classList.toggle("active", sub === "train");
  practiceEl.hidden = sub !== "practice";
  nextBtn.hidden = sub !== "practice";
  spellControls.hidden = sub !== "spell";
  textEl.parentElement.hidden = sub !== "spell";
  trainPanel.hidden = sub !== "train";
  app.refit(); // the train side panel changes the space available for the video
  if (sub === "practice") {
    practiceOrder = shuffle([...RULE_LETTERS]);
    practiceIdx = 0;
    practiceMatched = 0;
    practiceScoreEl.textContent = "0 matched";
    showPracticeTarget();
  }
  if (sub === "train") refreshTrainPanel();
}

function setDebug(on) {
  debugOn = on;
  debugToggle.checked = on;
  debugEl.hidden = !on;
}

// =============================================================================
//  Train: recording labelled samples
// =============================================================================

function loadSigner() {
  try {
    return localStorage.getItem(SIGNER_KEY) || "";
  } catch {
    return "";
  }
}

function saveSigner(name) {
  try {
    localStorage.setItem(SIGNER_KEY, name);
  } catch {
    // not persisted, that's fine
  }
}

function setTrainLetter(letter) {
  trainLetter = letter;
  trainLetterEl.textContent = letter;
  trainDescEl.textContent = DESCRIPTIONS[letter];
  trainDeleteBtn.textContent = `Delete all “${letter}” samples`;
  for (const l of STATIC_LETTERS) gridButtons[l].classList.toggle("active", l === letter);
}

function updateRecording() {
  const want = sub === "train" && (recordKeyDown || recordBtnDown);
  if (want && !recording) {
    recording = true;
    takeCount = 0;
    nextRecordAt = 0;
    noHandWarned = false;
  } else if (!want && recording) {
    recording = false;
    if (takeCount) trainStatus.textContent = `Recorded ${takeCount} “${trainLetter}” samples.`;
    refreshTrainPanel(); // per-signer counts and total
  }
  trainRecordBtn.classList.toggle("recording", recording);
  stageEl.classList.toggle("recording", recording);
}

function stopRecording() {
  recordKeyDown = recordBtnDown = false;
  updateRecording();
}

// Called every frame while recording: one sample every RECORD_INTERVAL_MS.
// We keep a fixed schedule (next = previous + 100 ms) rather than "100 ms
// since the last sample": camera frames come every ~33 ms, so the latter
// would really mean every 4th frame (132 ms, only ~7.5 samples/s).
function recordFrame(hands, now) {
  if (now < nextRecordAt) return;
  if (!hands.length) {
    if (!noHandWarned) trainStatus.textContent = "No hand visible: show your hand to the camera.";
    noHandWarned = true;
    return;
  }
  // With two hands visible, record the bigger (closer) one.
  let hand = hands[0];
  for (const h of hands) if (h.size > hand.size) hand = h;
  nextRecordAt = nextRecordAt && now - nextRecordAt < RECORD_INTERVAL_MS ? nextRecordAt + RECORD_INTERVAL_MS : now + RECORD_INTERVAL_MS;
  dataset.addSample({ letter: trainLetter, signer: trainSigner.value.trim(), hand: hand.label, lm: packLandmarks(hand.pts) });
  takeCount++;
  noHandWarned = false;
  trainStatus.textContent = `Recording “${trainLetter}”… ${takeCount} this take`;
  updateLetterCount(trainLetter);
}

function updateLetterCount(letter) {
  const n = dataset.counts().byLetter[letter] || 0;
  const b = gridButtons[letter];
  b.querySelector(".cnt").textContent = n;
  b.style.setProperty("--fill", Math.min(1, n / TARGET_PER_LETTER).toFixed(3));
  b.classList.toggle("full", n >= TARGET_PER_LETTER);
}

// Re-draw the whole overview (counts per letter + per signer + total).
function refreshTrainPanel() {
  const c = dataset.counts();
  for (const l of STATIC_LETTERS) updateLetterCount(l);
  const signers = Object.entries(c.bySigner).sort((a, b) => b[1] - a[1]);
  trainSigners.innerHTML = signers.length
    ? signers.map(([name, n]) => `<div><span>${escapeHtml(name)}</span><b>${n}</b></div>`).join("")
    : `<div class="sp-muted">No samples yet</div>`;
  const done = STATIC_LETTERS.filter((l) => c.byLetter[l] >= TARGET_PER_LETTER).length;
  trainTotal.textContent = `${c.total} samples · ${done}/${STATIC_LETTERS.length} letters at ${TARGET_PER_LETTER}+`;
  trainStorage.textContent = dataset.isPersistent() ? "Saved in this browser" : "⚠ Not saved: export to keep";
  trainStorage.classList.toggle("warn", !dataset.isPersistent());
}

function escapeHtml(s) {
  return s.replace(/[&<>"']/g, (ch) => ({ "&": "&amp;", "<": "&lt;", ">": "&gt;", '"': "&quot;", "'": "&#39;" })[ch]);
}

function exportData() {
  const blob = new Blob([dataset.exportJSON()], { type: "application/json" });
  const a = document.createElement("a");
  a.href = URL.createObjectURL(blob);
  const who = (trainSigner.value.trim() || "samples").replace(/[^a-z0-9_-]+/gi, "_");
  a.download = `handlab-signs-${who}-${new Date().toISOString().slice(0, 10)}.json`;
  a.click();
  setTimeout(() => URL.revokeObjectURL(a.href), 1000);
  trainStatus.textContent = `Exported ${dataset.counts().total} samples.`;
}

async function importData(file) {
  try {
    const r = await dataset.importJSON(await file.text());
    trainStatus.textContent = `Imported ${r.added} samples` +
      (r.duplicates ? `, skipped ${r.duplicates} already here` : "") +
      (r.invalid ? `, ignored ${r.invalid} invalid` : "") + ".";
  } catch (err) {
    trainStatus.textContent = `Import failed: ${err.message}`;
  }
  refreshTrainPanel();
}

async function deleteCurrentLetter() {
  const n = dataset.counts().byLetter[trainLetter] || 0;
  if (!n) return;
  if (!confirm(`Delete all ${n} samples of “${trainLetter}”? This can't be undone (unless you exported them).`)) return;
  await dataset.deleteLetter(trainLetter);
  trainStatus.textContent = `Deleted ${n} “${trainLetter}” samples.`;
  refreshTrainPanel();
}

function buildTrainGrid() {
  for (const l of STATIC_LETTERS) {
    const b = document.createElement("button");
    b.className = "sp-cell";
    b.innerHTML = `<span class="ltr">${l}</span><span class="cnt">0</span>`;
    b.title = `Record “${l}”`;
    b.addEventListener("click", () => setTrainLetter(l));
    trainGrid.appendChild(b);
    gridButtons[l] = b;
  }
}

// =============================================================================
//  Debug panel
// =============================================================================
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

  const top = [...RULE_LETTERS].sort((a, b) => r.scores[b] - r.scores[a]).slice(0, 4);
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
    `</div><div class="dbg-sub">Top scores (rules)</div>${bars}`;
}

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
    stageEl = $("stage");
    practiceEl = $("signPractice");
    practiceLetterEl = $("practiceLetter");
    practiceDescEl = $("practiceDesc");
    practiceScoreEl = $("practiceScore");
    practiceTickEl = $("practiceTick");
    practiceNote = $("practiceNote");
    debugEl = $("signDebug");
    spellBtn = $("signSpellBtn");
    practiceBtn = $("signPracticeBtn");
    trainBtn = $("signTrainBtn");
    nextBtn = $("practiceNextBtn");
    debugToggle = $("signDebugToggle");
    spellControls = $("signSpellControls");

    trainPanel = $("trainPanel");
    trainLetterEl = $("trainLetter");
    trainDescEl = $("trainDesc");
    trainGrid = $("trainGrid");
    trainStatus = $("trainStatus");
    trainSigner = $("trainSigner");
    trainSigners = $("trainSigners");
    trainTotal = $("trainTotal");
    trainStorage = $("trainStorage");
    trainDeleteBtn = $("trainDeleteBtn");
    trainRecordBtn = $("trainRecordBtn");
    trainImportFile = $("trainImportFile");

    spellBtn.addEventListener("click", () => setSub("spell"));
    practiceBtn.addEventListener("click", () => setSub("practice"));
    trainBtn.addEventListener("click", () => setSub("train"));
    nextBtn.addEventListener("click", nextPractice);
    $("signSpaceBtn").addEventListener("click", () => setText(text + " "));
    $("signBackBtn").addEventListener("click", () => setText(text.slice(0, -1)));
    $("signClearBtn").addEventListener("click", () => setText(""));
    debugToggle.addEventListener("change", () => setDebug(debugToggle.checked));

    // Train panel
    buildTrainGrid();
    setTrainLetter("A");
    trainSigner.value = loadSigner();
    trainSigner.addEventListener("input", () => saveSigner(trainSigner.value.trim()));
    // Hold-to-record button (pointer events cover mouse, touch and pen)
    trainRecordBtn.addEventListener("pointerdown", (e) => {
      trainRecordBtn.setPointerCapture(e.pointerId);
      recordBtnDown = true;
      updateRecording();
    });
    const release = () => { recordBtnDown = false; updateRecording(); };
    trainRecordBtn.addEventListener("pointerup", release);
    trainRecordBtn.addEventListener("pointercancel", release);
    window.addEventListener("blur", stopRecording); // don't get stuck recording
    trainDeleteBtn.addEventListener("click", deleteCurrentLetter);
    $("trainExportBtn").addEventListener("click", exportData);
    $("trainImportBtn").addEventListener("click", () => trainImportFile.click());
    trainImportFile.addEventListener("change", () => {
      if (trainImportFile.files[0]) importData(trainImportFile.files[0]);
      trainImportFile.value = "";
    });

    setText("");
    // Load the saved dataset in the background.
    dataset.openDataset().then(() => {
      if (sub === "train") refreshTrainPanel();
    });
  },

  enter() {
    stable.reset("");
    heldLetter = "";
    locked = false;
    shown.letter = null;
    if (sub === "train") trainPanel.hidden = false;
  },

  exit() {
    stopRecording();
    trainPanel.hidden = true; // it lives outside the sign layer, so hide it explicitly
  },

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

    if (sub === "train") {
      // No typing while collecting data, just record.
      holdProgress = 0;
      if (recording) recordFrame(hands, now);
      return;
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

    if (sub === "practice" && practiceNextAt && now >= practiceNextAt) nextPractice();
  },

  render() {
    // Big letter + confidence + hold ring (DOM text only changes when needed).
    const faded = !heldLetter || sub === "train";
    const show = (sub !== "train" && heldLetter) || (bestLetter ? bestLetter : "?");
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
    holdRing.classList.toggle("done", locked && sub !== "train");
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
    if (sub === "train") return { text: recording ? `REC ${trainLetter}` : `TRAIN ${trainLetter}`, tone: recording ? "REC" : "HOVER" };
    return { text: sub === "practice" ? "PRACTICE" : "SPELL", tone: heldLetter ? "DRAW" : "HOVER" };
  },

  onKey(e) {
    if (e.ctrlKey || e.metaKey || e.altKey) return false;
    const key = e.key.toLowerCase();
    // Sub-mode switches work everywhere in Sign mode
    if (key === "t") { setSub(sub === "train" ? "spell" : "train"); return true; }
    if (key === "p") { setSub(sub === "practice" ? "spell" : "practice"); return true; }
    if (key === "d") { setDebug(!debugOn); return true; }

    if (sub === "train") {
      if (key === "r") {
        e.preventDefault();
        if (!e.repeat) { recordKeyDown = true; updateRecording(); }
        return true;
      }
      const i = STATIC_LETTERS.indexOf(trainLetter);
      if (e.key === "ArrowRight" || e.key === "ArrowDown") {
        e.preventDefault();
        setTrainLetter(STATIC_LETTERS[(i + 1) % STATIC_LETTERS.length]);
        return true;
      }
      if (e.key === "ArrowLeft" || e.key === "ArrowUp") {
        e.preventDefault();
        setTrainLetter(STATIC_LETTERS[(i - 1 + STATIC_LETTERS.length) % STATIC_LETTERS.length]);
        return true;
      }
      return false;
    }

    if (sub === "practice") {
      if (key === "n") { nextPractice(); return true; }
      return false;
    }

    // Spell
    if (e.code === "Space") {
      e.preventDefault();
      setText(text + " ");
    } else if (e.key === "Backspace") {
      e.preventDefault();
      setText(text.slice(0, -1));
    } else if (key === "c") setText("");
    else return false;
    return true;
  },

  onKeyUp(e) {
    if (e.key.toLowerCase() === "r" && recordKeyDown) {
      recordKeyDown = false;
      updateRecording();
    }
  },
};
