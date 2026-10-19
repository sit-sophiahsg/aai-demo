// =============================================================================
//  sign/classifiers.js: one interface over the three classifiers
// =============================================================================
//
//    Rules  hand-written fuzzy rules (sign/rules.js), 19 letters, no data needed
//    k-NN   nearest neighbours over your recordings (sign/knn.js), 24 letters
//    MLP    small neural network trained in the browser (sign/mlp.js), 24 letters
//
//  predict(kind, hand, feat, out) fills `out` the same way for all three:
//    out.letter  the predicted letter, or "" if not confident enough
//    out.conf    confidence 0..1 (rules: fuzzy score; models: probability)
//    out.probs   Float32Array(24): a score per letter in CLASSES order
//    out.ready   false if this classifier can't run yet (no data / not trained)
//    out.note    short reason when not ready
//  so the hold-to-type logic, the confidence ring, Practice mode and the
//  debug panel work identically whichever classifier is selected.
//
//  Each classifier has its own thresholds (THRESHOLDS): fuzzy scores and
//  probabilities aren't on the same scale, so one cut-off wouldn't suit all.
// =============================================================================

import { STATIC_LETTERS } from "./letters.js";
import { CFG, RULE_LETTERS, scoreLetters } from "./rules.js";
import { toVector, sampleToVector, VECTOR_LENGTH } from "./features.js";
import { KNN } from "./knn.js";
import { loadTF, trainMLP, loadSavedMLP, savedSummary, MLP_DEFAULTS } from "./mlp.js";
import * as dataset from "./dataset.js";

export const CLASSES = STATIC_LETTERS;    // class index i <-> letter CLASSES[i]
const NC = CLASSES.length;
const D = VECTOR_LENGTH;
export const KINDS = ["rules", "knn", "mlp"];
export const KIND_NAMES = { rules: "Rules", knn: "k-NN", mlp: "MLP" };

export const THRESHOLDS = {
  rules: { min: CFG.minConfidence, accept: CFG.acceptConfidence },
  knn: { min: 0.5, accept: 0.7 },
  mlp: { min: 0.5, accept: 0.8 },
};

// Letters with fewer samples than this are left out of Practice for the models.
const MIN_SAMPLES_FOR_PRACTICE = 10;
const KNN_KEY = "handlab.knn";

// ---- Dataset -> matrix ----

// All samples -> X (n x 85 feature vectors) and y (class indices).
export function buildMatrix(samples) {
  const n = samples.length;
  const X = new Float32Array(n * D);
  const y = new Int32Array(n);
  for (let i = 0; i < n; i++) {
    sampleToVector(samples[i], X.subarray(i * D, i * D + D));
    y[i] = CLASSES.indexOf(samples[i].letter);
  }
  return { X, y, n, D };
}

// ---- k-NN (always fitted on ALL recorded samples) ----

export const knnSettings = loadKnnSettings();
let knn = null;
let knnDirty = true;

function loadKnnSettings() {
  try {
    const s = JSON.parse(localStorage.getItem(KNN_KEY) || "{}");
    return { k: Math.max(1, Math.min(25, s.k | 0 || 5)), metric: s.metric === "cosine" ? "cosine" : "euclidean" };
  } catch {
    return { k: 5, metric: "euclidean" };
  }
}

export function setKnnSettings(k, metric) {
  knnSettings.k = Math.max(1, Math.min(25, k | 0 || 5));
  knnSettings.metric = metric === "cosine" ? "cosine" : "euclidean";
  knnDirty = true;
  try { localStorage.setItem(KNN_KEY, JSON.stringify(knnSettings)); } catch { /* not persisted */ }
}

// Call after recording / importing / deleting samples.
export function markDataChanged() {
  knnDirty = true;
}

export function getKnn() {
  if (knnDirty) {
    const { X, y, n } = buildMatrix(dataset.allSamples());
    knn = new KNN(knnSettings).fit(X, y, n, D, NC);
    knnDirty = false;
  }
  return knn;
}

// ---- MLP (the deployed model: trained on all samples, saved in IndexedDB) ----

let mlp = null;
let mlpState = savedSummary() ? "saved" : "none"; // none | saved | loading | ready | training | error
let mlpError = "";

export const mlpInfo = () => ({ state: mlpState, error: mlpError, summary: mlp?.meta?.trainedAt ? mlp.meta : savedSummary() });

// Load TF.js + the saved model (once). Safe to call repeatedly.
export async function ensureMlpLoaded() {
  if (mlp || mlpState === "loading" || mlpState === "training") return mlp;
  mlpState = "loading";
  try {
    const tf = await loadTF();
    mlp = await loadSavedMLP(tf);
    mlpState = mlp ? "ready" : "none";
  } catch (err) {
    mlpState = "error";
    mlpError = err.message;
  }
  return mlp;
}

// Train the deployed MLP on every recorded sample (15% held out to report
// validation accuracy), then save it. onEpoch(epoch, logs) reports progress.
export async function trainDeployedMLP({ epochs = MLP_DEFAULTS.epochs, onEpoch } = {}) {
  const samples = dataset.allSamples();
  if (samples.length < 50) throw new Error("Record at least 50 samples first.");
  const prevState = mlpState;
  mlpState = "training";
  try {
    const tf = await loadTF();
    const { X, y, n } = buildMatrix(samples);
    let last = {};
    const clf = await trainMLP(tf, X, y, n, D, NC, {
      epochs,
      validationSplit: MLP_DEFAULTS.validationSplit,
      onEpoch: (e, logs) => { last = logs; return onEpoch?.(e, logs); },
    });
    const counts = dataset.counts().byLetter;
    await clf.save({
      trainedAt: new Date().toISOString(),
      samples: n,
      epochs,
      trainAcc: last.acc,
      valAcc: last.val_acc,
      letters: CLASSES.filter((l) => counts[l] >= MIN_SAMPLES_FOR_PRACTICE),
    });
    mlp?.dispose();
    mlp = clf;
    mlpState = "ready";
    return clf.meta;
  } catch (err) {
    mlpState = mlp ? "ready" : prevState === "saved" ? "saved" : "none";
    throw err;
  }
}

// ---- Live prediction ----

const vec = new Float32Array(D);
const ruleScores = {};

export function makeResult() {
  return { letter: "", conf: 0, probs: new Float32Array(NC), ready: true, note: "" };
}

// kind: "rules" | "knn" | "mlp". hand: tracked hand (pts + label).
// feat: measurements from computeFeatures(hand.pts) (needed by the rules).
export function predict(kind, hand, feat, out) {
  out.letter = "";
  out.conf = 0;
  out.ready = true;
  out.note = "";
  out.probs.fill(0);
  const th = THRESHOLDS[kind];

  if (kind === "rules") {
    scoreLetters(feat, ruleScores);
    let best = "";
    for (const l of RULE_LETTERS) {
      out.probs[CLASSES.indexOf(l)] = ruleScores[l];
      if (ruleScores[l] > out.conf) { out.conf = ruleScores[l]; best = l; }
    }
    if (out.conf >= th.min) out.letter = best;
    return out;
  }

  const index = getKnn();
  if (!index.n) {
    out.ready = false;
    out.note = "no samples: record some in Train";
    return out;
  }
  toVector(hand.pts, hand.label, vec);

  let r;
  if (kind === "knn") {
    r = index.predict(vec, out.probs);
    out.conf = r.conf * index.noveltyGate(r.nearest);
  } else {
    if (!mlp) {
      out.ready = false;
      out.note = mlpState === "loading" ? "loading model…" : mlpState === "training" ? "training…" : "not trained yet: Train > Train MLP";
      if (mlpState === "saved") ensureMlpLoaded();
      return out;
    }
    r = mlp.predict(vec, out.probs);
    // The MLP's softmax is confident even for poses it has never seen, so
    // reuse k-NN's "is this anywhere near the data?" check as a gate.
    out.conf = r.conf * index.noveltyGate(index.nearestDistance(vec));
  }
  if (out.conf >= th.min) out.letter = CLASSES[r.cls];
  return out;
}

// Letters Practice mode should offer for a classifier.
export function practiceLetters(kind) {
  if (kind === "rules") return RULE_LETTERS;
  if (kind === "mlp" && mlp?.meta?.letters?.length) return mlp.meta.letters;
  const counts = dataset.counts().byLetter;
  const ls = CLASSES.filter((l) => counts[l] >= MIN_SAMPLES_FOR_PRACTICE);
  return ls.length ? ls : RULE_LETTERS;
}
