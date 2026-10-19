// =============================================================================
//  sign/evaluate.js: compare Rules vs k-NN vs MLP on held-out data
// =============================================================================
//
//  1. SPLIT the recorded samples into a TRAIN set (80%) and a TEST set (20%),
//     stratified per letter (every letter keeps ~20% in test).
//       random        shuffle each letter's samples, hold out 20%
//       time-blocked  hold out the LAST 20% (by time) of each signer's samples
//                     of each letter
//     Why offer both? Frames recorded at 10/s are almost identical to their
//     neighbours. A random split puts near-duplicates of every test frame in
//     the training set, so accuracy comes out OPTIMISTIC (the model has, in
//     effect, already seen the test). Time-blocked keeps whole stretches of
//     recording unseen, which is closer to "a new attempt at the sign".
//
//  2. FIT k-NN and TRAIN a fresh MLP on the train set only. (Rules need no
//     training.) This MLP is separate from the one used live: evaluation
//     never overwrites your deployed model.
//
//  3. PREDICT every test sample with all three and compare to the true letter:
//       accuracy          correct / total, over all 24 letters
//       rule-letter acc.  the same, but only the 19 letters the rules know
//                         (fairer to the rules, which can't do E M N S T)
//       per-letter acc.   which letters are weak
//       confusion matrix  rows = true letter, columns = predicted letter
//                         (+ a "none" column: no confident answer)
//
//  4. LEAVE-ONE-SIGNER-OUT (only with 2+ signers): for each person, train on
//     everyone ELSE and test on them. This answers the real question: how
//     well does it work for someone it has never seen? Usually the lowest
//     and most honest number.
//
//  The split uses a seeded random generator, so re-running with the same
//  data gives the same split (the MLP's own weight initialisation still varies).
// =============================================================================

import { STATIC_LETTERS } from "./letters.js";
import { RULE_LETTERS, CFG, scoreLetters } from "./rules.js";
import { makeFeatures, computeFeatures, unpackLandmarks, VECTOR_LENGTH } from "./features.js";
import { KNN } from "./knn.js";
import { loadTF, trainMLP, MLP_DEFAULTS } from "./mlp.js";
import { buildMatrix } from "./classifiers.js";

const NC = STATIC_LETTERS.length;
const NONE = NC;                       // confusion-matrix column for "no answer"
const D = VECTOR_LENGTH;
const RULE_SET = new Set(RULE_LETTERS.map((l) => STATIC_LETTERS.indexOf(l)));
const tick = () => new Promise((r) => setTimeout(r, 0)); // let the page breathe

function mulberry32(seed) {
  return () => {
    seed |= 0; seed = (seed + 0x6d2b79f5) | 0;
    let t = Math.imul(seed ^ (seed >>> 15), 1 | seed);
    t = (t + Math.imul(t ^ (t >>> 7), 61 | t)) ^ t;
    return ((t ^ (t >>> 14)) >>> 0) / 4294967296;
  };
}

// ---- 1. Splits ----

// Returns { train: [indices], test: [indices] }.
export function splitIndices(samples, mode = "random", testFrac = 0.2, seed = 42) {
  const rand = mulberry32(seed);
  const groups = new Map();
  samples.forEach((s, i) => {
    // random: group by letter. time-blocked: by letter AND signer, so each
    // person's recording of each letter is cut at its own 80% point.
    const key = mode === "time" ? `${s.letter}|${s.signer}` : s.letter;
    if (!groups.has(key)) groups.set(key, []);
    groups.get(key).push(i);
  });
  const train = [], test = [];
  for (const idx of groups.values()) {
    if (mode === "time") idx.sort((a, b) => samples[a].t - samples[b].t);
    else for (let i = idx.length - 1; i > 0; i--) { const j = Math.floor(rand() * (i + 1)); [idx[i], idx[j]] = [idx[j], idx[i]]; }
    // Groups too small to split stay entirely in training.
    const nTest = idx.length >= 5 ? Math.max(1, Math.round(idx.length * testFrac)) : 0;
    const cut = idx.length - nTest;
    for (let k = 0; k < idx.length; k++) (k < cut ? train : test).push(idx[k]);
  }
  return { train, test };
}

// Copy selected rows of X/y into new arrays.
function subset(X, y, idx) {
  const Xs = new Float32Array(idx.length * D), ys = new Int32Array(idx.length);
  idx.forEach((r, i) => { Xs.set(X.subarray(r * D, r * D + D), i * D); ys[i] = y[r]; });
  return { X: Xs, y: ys, n: idx.length };
}

// ---- 2/3. Classifiers on the test set (each returns predicted class per row, -1 = none) ----

// The rules are deterministic, so compute their answer for every sample once.
async function rulePredictionsAll(samples) {
  const f = makeFeatures(), s = {};
  const out = new Int32Array(samples.length);
  for (let i = 0; i < samples.length; i++) {
    computeFeatures(unpackLandmarks(samples[i].lm), f);
    scoreLetters(f, s);
    let best = -1, conf = 0;
    for (const l of RULE_LETTERS) if (s[l] > conf) { conf = s[l]; best = STATIC_LETTERS.indexOf(l); }
    out[i] = conf >= CFG.minConfidence ? best : -1;
    if (i % 500 === 499) await tick();
  }
  return out;
}

async function knnPredictions(train, testX, nTest, k, metric) {
  const knn = new KNN({ k, metric }).fit(train.X, train.y, train.n, D, NC);
  const probs = new Float32Array(NC);
  const out = new Int32Array(nTest);
  for (let i = 0; i < nTest; i++) {
    // In evaluation every test sample IS a real letter, so no unknown-pose gate:
    // we measure which letter k-NN picks.
    out[i] = knn.predict(testX.subarray(i * D, i * D + D), probs).cls;
    if (i % 200 === 199) await tick();
  }
  return out;
}

async function mlpPredictions(tf, train, testX, nTest, epochs, onEpoch) {
  const clf = await trainMLP(tf, train.X, train.y, train.n, D, NC, { epochs, validationSplit: 0, onEpoch });
  try {
    return Int32Array.from(clf.predictBatch(testX, nTest, D));
  } finally {
    clf.dispose();
  }
}

// ---- Metrics ----

function score(truth, pred) {
  const confusion = Array.from({ length: NC }, () => new Array(NC + 1).fill(0));
  const perLetter = Array.from({ length: NC }, () => ({ correct: 0, total: 0 }));
  let correct = 0, ruleCorrect = 0, ruleTotal = 0;
  for (let i = 0; i < truth.length; i++) {
    const t = truth[i], p = pred[i];
    const ok = t === p;
    confusion[t][p < 0 ? NONE : p]++;
    perLetter[t].total++;
    if (ok) { correct++; perLetter[t].correct++; }
    if (RULE_SET.has(t)) { ruleTotal++; if (ok) ruleCorrect++; }
  }
  return {
    acc: truth.length ? correct / truth.length : 0,
    ruleAcc: ruleTotal ? ruleCorrect / ruleTotal : 0,
    correct, total: truth.length,
    perLetter, confusion,
  };
}

// ---- Main entry ----

// opts: { split, k, metric, epochs, onProgress(text) }
export async function runEvaluation(samples, opts = {}) {
  const { split = "random", k = 5, metric = "euclidean", epochs = MLP_DEFAULTS.epochs, onProgress = () => {} } = opts;
  if (samples.length < 50) throw new Error("Record at least 50 samples first.");

  onProgress("Preparing feature vectors…");
  await tick();
  const { X, y } = buildMatrix(samples);
  const rulesAll = await rulePredictionsAll(samples);
  const tf = await loadTF();

  // --- 80/20 split ---
  const { train: trIdx, test: teIdx } = splitIndices(samples, split);
  const train = subset(X, y, trIdx);
  const test = subset(X, y, teIdx);
  onProgress(`Split: ${train.n} train / ${test.n} test. Running k-NN…`);
  await tick();
  const predRules = Int32Array.from(teIdx, (i) => rulesAll[i]);
  const predKnn = await knnPredictions(train, test.X, test.n, k, metric);
  const predMlp = await mlpPredictions(tf, train, test.X, test.n, epochs,
    (e) => onProgress(`Training MLP on the train split… epoch ${e + 1}/${epochs}`));

  const results = { rules: score(test.y, predRules), knn: score(test.y, predKnn), mlp: score(test.y, predMlp) };

  // --- Leave-one-signer-out ---
  const signers = [...new Set(samples.map((s) => s.signer))];
  let loso = null;
  if (signers.length >= 2) {
    loso = { rows: [] };
    for (let si = 0; si < signers.length; si++) {
      const who = signers[si];
      const tr = [], te = [];
      samples.forEach((s, i) => (s.signer === who ? te : tr).push(i));
      if (te.length < 10 || tr.length < 50) continue;
      const trn = subset(X, y, tr), tst = subset(X, y, te);
      onProgress(`Leave-one-signer-out ${si + 1}/${signers.length} (testing on “${who}”): k-NN…`);
      await tick();
      const pk = await knnPredictions(trn, tst.X, tst.n, k, metric);
      const pm = await mlpPredictions(tf, trn, tst.X, tst.n, epochs,
        (e) => onProgress(`Leave-one-signer-out ${si + 1}/${signers.length} (testing on “${who}”): MLP epoch ${e + 1}/${epochs}`));
      loso.rows.push({
        signer: who,
        n: tst.n,
        rules: score(tst.y, Int32Array.from(te, (i) => rulesAll[i])).acc,
        knn: score(tst.y, pk).acc,
        mlp: score(tst.y, pm).acc,
      });
    }
    const mean = (key) => loso.rows.reduce((s, r) => s + r[key], 0) / (loso.rows.length || 1);
    loso.mean = { rules: mean("rules"), knn: mean("knn"), mlp: mean("mlp") };
  }

  onProgress("Done.");
  return {
    split, k, metric, epochs,
    nTrain: train.n, nTest: test.n,
    nSigners: signers.length,
    letters: STATIC_LETTERS,
    results, loso,
  };
}
