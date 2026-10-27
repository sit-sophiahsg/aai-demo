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
//    PRACTICE  a target letter + description, a tick when you match it
//    TRAIN     record labelled landmark samples for the learned models
// =============================================================================

import { Debouncer } from "../gestures.js";
import { makeFeatures, computeFeatures, packLandmarks } from "../sign/features.js";
import { CFG, fingerStates } from "../sign/rules.js";
import { STATIC_LETTERS, MOTION_LETTERS, DESCRIPTIONS } from "../sign/letters.js";
import { MotionTracker } from "../sign/motion.js";
import * as dataset from "../sign/dataset.js";
import * as clf from "../sign/classifiers.js";
import { lineChart } from "../sign/charts.js";
import { initEvalView, openEvalView, isEvalOpen } from "../sign/eval-view.js";
import { MLP_DEFAULTS } from "../sign/mlp.js";

const FINGER_NAMES = ["Index", "Middle", "Ring", "Pinky"];
const RECORD_INTERVAL_MS = 100;   // ~10 samples per second while recording
const TARGET_PER_LETTER = 100;
const SIGNER_KEY = "handlab.signer";
const CLASSIFIER_KEY = "handlab.sign.classifier";

let app = null;
const perHand = {};   // label -> { feat (measurements), res (classifier result) }
const stable = new Debouncer(CFG.stableFrames, "");
let kind = loadClassifierChoice(); // "rules" | "knn" | "mlp"
// J and Z (moving letters) are recognised separately from the fingertip path.
const motion = new MotionTracker();
let motionShown = "";     // J/Z just typed: keep it on the big letter briefly
let motionShownUntil = 0;
let bestReady = true;
let bestNote = "";

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
let overlayEl, overlayLetterEl, overlayTextEl; // Spell: letter + text on the video
let practiceEl, practiceLetterEl, practiceDescEl, practiceScoreEl, practiceTickEl, practiceNote;
let debugEl, spellBtn, practiceBtn, trainBtn, nextBtn, debugToggle, spellControls;
let trainPanel, trainLetterEl, trainDescEl, trainGrid, trainStatus, trainSigner, trainSigners,
  trainTotal, trainStorage, trainDeleteBtn, trainRecordBtn, trainImportFile;
let clfButtons, ringCard, clfTag, mlpTrainBtn, mlpStatus, mlpAccChart, mlpLossChart, knnKInput, knnMetricSel, mlpEpochsInput;
const gridButtons = {};
const shown = { letter: null, faded: null, text: null };

function handResult(label) {
  if (!perHand[label]) perHand[label] = { feat: makeFeatures(), res: clf.makeResult() };
  return perHand[label];
}

// =============================================================================
//  Classifier choice: Rules / k-NN / MLP
// =============================================================================

function loadClassifierChoice() {
  try {
    const k = localStorage.getItem(CLASSIFIER_KEY);
    return clf.KINDS.includes(k) ? k : "rules";
  } catch {
    return "rules";
  }
}

function setClassifier(next) {
  kind = clf.KINDS.includes(next) ? next : "rules";
  try { localStorage.setItem(CLASSIFIER_KEY, kind); } catch { /* not persisted */ }
  clfButtons.forEach((b) => b.classList.toggle("active", b.dataset.clf === kind));
  clfTag.textContent = clf.KIND_NAMES[kind];
  stable.reset("");
  heldLetter = "";
  locked = false;
  if (kind === "mlp") clf.ensureMlpLoaded().then(updateMlpStatus);
  if (sub === "practice") setSub("practice"); // re-pick letters for this classifier
}

// Status line under "Train MLP".
function updateMlpStatus() {
  const { state, error, summary } = clf.mlpInfo();
  if (state === "training") return; // the training loop writes its own progress
  if (state === "error") mlpStatus.textContent = `Error: ${error}`;
  else if (summary?.trainedAt) {
    mlpStatus.textContent = `Saved model: ${summary.samples} samples, ${summary.epochs} epochs` +
      (summary.valAcc != null ? `, validation accuracy ${Math.round(summary.valAcc * 100)}%` : "") +
      ` · ${new Date(summary.trainedAt).toLocaleString()}`;
  } else mlpStatus.textContent = "No trained model yet.";
}

async function trainMlp() {
  const epochs = Math.max(5, Math.min(200, Number(mlpEpochsInput.value) || MLP_DEFAULTS.epochs));
  const hist = { acc: [], val_acc: [], loss: [], val_loss: [] };
  const draw = () => {
    lineChart(mlpAccChart, [
      { name: "train", color: "var(--series-1)", values: hist.acc },
      { name: "validation", color: "var(--series-2)", values: hist.val_acc },
    ], { title: "Accuracy", yMin: 0, yMax: 1, format: (v) => `${Math.round(v * 100)}%` });
    lineChart(mlpLossChart, [
      { name: "train", color: "var(--series-1)", values: hist.loss },
      { name: "validation", color: "var(--series-2)", values: hist.val_loss },
    ], { title: "Loss", yMin: 0 });
  };
  mlpTrainBtn.disabled = true;
  mlpStatus.textContent = "Loading TensorFlow.js…";
  try {
    const meta = await clf.trainDeployedMLP({
      epochs,
      onEpoch: async (e, logs) => {
        for (const key in hist) hist[key].push(logs[key] ?? null);
        mlpStatus.textContent = `Epoch ${e + 1}/${epochs} · loss ${logs.loss.toFixed(3)} · accuracy ${Math.round(logs.acc * 100)}%` +
          (logs.val_acc != null ? ` · validation ${Math.round(logs.val_acc * 100)}%` : "");
        draw();
        await new Promise((r) => setTimeout(r, 0)); // let the page repaint
      },
    });
    updateMlpStatus();
    mlpStatus.textContent = `Trained and saved. ` + mlpStatus.textContent.replace(/^Saved model: /, "");
    if (meta && kind !== "mlp") mlpStatus.textContent += " Select “MLP” as the classifier to use it.";
  } catch (err) {
    mlpStatus.textContent = `Training failed: ${err.message}`;
  } finally {
    mlpTrainBtn.disabled = false;
  }
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
      practiceNote.textContent = "Nice!";
      practiceScoreEl.textContent = `${practiceMatched} matched`;
      practiceNextAt = now + 1200;
    } else if (!practiceNextAt) {
      practiceNote.textContent = `That looked like “${letter}”. Check the description and try again.`;
    }
  }
}

function setText(t) {
  text = t.slice(-80);
  if (shown.text !== text) textEl.textContent = overlayTextEl.textContent = shown.text = text;
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
  ringCard.hidden = sub === "train"; // Train has its own panel; the letter ring is for Spell/Practice
  nextBtn.hidden = sub !== "practice";
  spellControls.hidden = sub !== "spell";
  textEl.parentElement.hidden = sub !== "spell";
  overlayEl.hidden = sub !== "spell";
  trainPanel.hidden = sub !== "train";
  app.refit(); // the train side panel changes the space available for the video
  app.updateFullscreen(); // full screen is for Spell only
  if (sub === "practice") {
    practiceOrder = shuffle([...clf.practiceLetters(kind), ...MOTION_LETTERS]);
    practiceIdx = 0;
    practiceMatched = 0;
    practiceScoreEl.textContent = "0 matched";
    showPracticeTarget();
  }
  if (sub === "train") { refreshTrainPanel(); updateMlpStatus(); }
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
    if (takeCount) clf.markDataChanged(); // k-NN refits from the new data on next use
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
  trainStorage.textContent = dataset.isPersistent() ? "Saved in this browser" : "Not saved: export to keep";
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
  clf.markDataChanged();
  refreshTrainPanel();
}

async function deleteCurrentLetter() {
  const n = dataset.counts().byLetter[trainLetter] || 0;
  if (!n) return;
  if (!confirm(`Delete all ${n} samples of “${trainLetter}”? This can't be undone (unless you exported them).`)) return;
  await dataset.deleteLetter(trainLetter);
  clf.markDataChanged();
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
  if (kind !== "rules") fingerStates(f); // finger states for the table (the rules fill these themselves)
  let rows = "";
  for (let i = 0; i < 4; i++) {
    const [label, cls] = fingerState(f, i);
    rows += `<tr><td>${FINGER_NAMES[i]}</td><td>${fmt(f.ratio[i])}</td><td>${Math.round(f.bend[i])}°</td><td class="${cls}">${label}</td></tr>`;
  }
  const thumbState = f.thumbOut > CFG.thumbOutRatio || f.thumbSide > CFG.thumbOutSide ? ["out", "ext"]
    : f.thumbSide < CFG.thumbInSide ? ["tucked", "curl"] : ["beside", "curve"];
  rows += `<tr><td>Thumb</td><td>${fmt(f.thumbOut)}</td><td>side ${fmt(f.thumbSide)}</td><td class="${thumbState[1]}">${thumbState[0]}</td></tr>`;

  // Top 4: fuzzy scores for the rules, class probabilities for the models.
  const p = r.res.probs;
  const top = clf.CLASSES.map((l, i) => i).sort((a, b) => p[b] - p[a]).slice(0, 4);
  const bars = r.res.ready
    ? top.map((i) =>
      `<div class="dbg-score"><span>${clf.CLASSES[i]}</span><div class="dbg-bar"><div style="transform:scaleX(${p[i].toFixed(3)})"></div></div><span>${Math.round(p[i] * 100)}%</span></div>`
    ).join("")
    : `<div class="dbg-muted">${clf.KIND_NAMES[kind]}: ${r.res.note}</div>`;
  const topTitle = kind === "rules" ? "Top scores (rules)" : `Top probabilities (${clf.KIND_NAMES[kind]})`;

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
    `</div><div class="dbg-sub">${topTitle}</div>${bars}` +
    `<div class="dbg-sub">Motion (J / Z)</div><div class="dbg-muted">${motion.inProgress ? "drawing · " : ""}${motion.debug}</div>`;
}

// =============================================================================
//  Mode interface (see modes/draw.js for what each method is for)
// =============================================================================

export const signMode = {
  id: "sign",
  name: "Sign",

  // Practice and Train need the side panel, so only Spell can go full screen.
  allowsFullscreen() {
    return sub === "spell";
  },

  init(ctx) {
    app = ctx;
    const $ = (id) => document.getElementById(id);
    letterEl = $("signLetter");
    letterCard = $("signLetterCard");
    confBar = $("signConfBar");
    confText = $("signConfText");
    holdRing = $("signHoldRing");
    textEl = $("signText");
    overlayEl = $("spellOverlay");
    overlayLetterEl = $("spellOverlayLetter");
    overlayTextEl = $("spellOverlayText");
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

    // Classifier selector + Evaluate window
    clfButtons = [...document.querySelectorAll("[data-clf]")];
    ringCard = $("signRingCard");
    clfTag = $("signClfTag");
    clfButtons.forEach((b) => b.addEventListener("click", () => setClassifier(b.dataset.clf)));
    initEvalView();
    $("signEvalBtn").addEventListener("click", () => { stopRecording(); openEvalView(); });

    // Models section of the Train panel
    mlpTrainBtn = $("mlpTrainBtn");
    mlpStatus = $("mlpStatus");
    mlpAccChart = $("mlpAccChart");
    mlpLossChart = $("mlpLossChart");
    knnKInput = $("knnK");
    knnMetricSel = $("knnMetric");
    mlpEpochsInput = $("mlpEpochs");
    knnKInput.value = clf.knnSettings.k;
    knnMetricSel.value = clf.knnSettings.metric;
    mlpEpochsInput.value = MLP_DEFAULTS.epochs;
    const knnChanged = () => clf.setKnnSettings(Number(knnKInput.value), knnMetricSel.value);
    knnKInput.addEventListener("change", knnChanged);
    knnMetricSel.addEventListener("change", knnChanged);
    mlpTrainBtn.addEventListener("click", trainMlp);
    updateMlpStatus();

    setText("");
    // Load the saved dataset in the background.
    dataset.openDataset().then(() => {
      clf.markDataChanged();
      if (sub === "train") refreshTrainPanel();
    });
    setClassifier(kind); // restores the last choice (loads the MLP if needed)
  },

  enter() {
    stable.reset("");
    heldLetter = "";
    locked = false;
    shown.letter = null;
    motion.reset();
    if (sub === "train") trainPanel.hidden = false;
  },

  exit() {
    stopRecording();
    trainPanel.hidden = true; // it lives outside the sign layer, so hide it explicitly
  },

  update(hands, now) {
    // Classify every visible hand with the selected classifier; show the most confident one.
    bestHand = null;
    bestLetter = "";
    bestConf = 0;
    bestReady = true;
    bestNote = "";
    for (const hand of hands) {
      const r = handResult(hand.label);
      computeFeatures(hand.pts, r.feat);
      clf.predict(kind, hand, r.feat, r.res);
      if (!bestHand || r.res.conf > bestConf) {
        bestHand = hand;
        bestConf = r.res.conf;
        bestLetter = r.res.letter;
        bestReady = r.res.ready;
        bestNote = r.res.note;
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

    // --- Moving letters J and Z (sign/motion.js) ---
    // The tracker needs fuzzy finger states; the rules fill them in, the models don't.
    let feat = null;
    if (bestHand) {
      feat = perHand[bestHand.label].feat;
      if (kind !== "rules") fingerStates(feat);
    }
    const motionLetter = motion.update(bestHand, feat, now);
    if (motionLetter) {
      accept(motionLetter, now);
      motionShown = motionLetter;
      motionShownUntil = now + 900;
      // The hand is still in the I / pointing shape: don't also type that
      // static letter until the pose changes.
      locked = true;
      holdProgress = 1;
      return;
    }
    // While a J or Z is being drawn, the static look-alikes (I for J,
    // pointing/D for Z) must not type: keep restarting their hold timer.
    if (motion.inProgress) holdStart = now;

    // Hold-to-accept: the same letter, confident enough (per-classifier
    // threshold), for CFG.holdMs.
    holdProgress = 0;
    if (heldLetter && !locked && !motion.inProgress && bestLetter === heldLetter && bestConf >= clf.THRESHOLDS[kind].accept) {
      holdProgress = Math.min(1, (now - holdStart) / CFG.holdMs);
      if (holdProgress >= 1) {
        accept(heldLetter, now);
        locked = true; // to type the same letter again, change pose or drop the hand
      }
    }
    if (locked) holdProgress = 1;

    if (sub === "practice" && practiceNextAt && now >= practiceNextAt) nextPractice();
  },

  render(ctx, hands, now) {
    // The J / Z path being drawn, as feedback.
    if (sub !== "train" && motion.inProgress) motion.draw(ctx, "#facc15", app.pxScale());

    // Big letter + confidence + hold ring (DOM text only changes when needed).
    const drawing = sub !== "train" && motion.inProgress;
    const justMoved = sub !== "train" && now < motionShownUntil;
    const faded = drawing || (!justMoved && (!heldLetter || sub === "train"));
    const show = justMoved ? motionShown
      : drawing ? (motion.shape === "I" ? "J" : "Z")
      : (sub !== "train" && heldLetter) || (bestLetter ? bestLetter : "?");
    if (shown.letter !== show || shown.faded !== faded) {
      shown.letter = show;
      shown.faded = faded;
      letterEl.textContent = overlayLetterEl.textContent = show;
      letterEl.classList.toggle("faded", faded);
    }
    const conf = bestLetter ? bestConf : 0;
    const th = clf.THRESHOLDS[kind];
    const level = conf >= th.accept ? "high" : conf >= th.min ? "mid" : "low";
    confBar.style.setProperty("--c", conf.toFixed(3)); // the outer ring arc
    confBar.dataset.level = level;
    overlayLetterEl.dataset.level = faded ? "low" : level;
    confText.textContent = !bestHand ? "no hand" : drawing ? "drawing…" : !bestReady ? bestNote : `${Math.round(conf * 100)}%`;
    holdRing.style.setProperty("--p", holdProgress.toFixed(3));
    holdRing.classList.toggle("done", locked && sub !== "train");
    renderDebug();
  },

  handLost(label) {
    delete perHand[label];
  },

  trailColor(hand) { return app.handColors[hand.label]; },

  labelSuffix(hand) {
    return perHand[hand.label]?.res.letter || "";
  },

  badge() {
    if (sub === "train") return { text: recording ? `REC ${trainLetter}` : `TRAIN ${trainLetter}`, tone: recording ? "REC" : "HOVER" };
    if (motion.inProgress) return { text: `DRAWING ${motion.shape === "I" ? "J" : "Z"}`, tone: "WARN" };
    const shownLetter = heldLetter || bestLetter;
    return shownLetter ? { text: `LETTER ${shownLetter}`, tone: heldLetter ? "DRAW" : "HOVER" } : { text: "NO LETTER", tone: "HOVER" };
  },

  onKey(e) {
    if (isEvalOpen()) return true; // the Evaluate window has the keyboard
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
