// =============================================================================
//  sign/mlp.js: a small neural network (multi-layer perceptron) in TF.js
// =============================================================================
//
//  ARCHITECTURE (85 inputs -> 24 letter probabilities)
//
//     input 85 ─▶ Dense 64 + ReLU ─▶ Dropout 0.3 ─▶ Dense 64 + ReLU ─▶ Dropout 0.3
//                                                                   ─▶ Dense 24 + softmax
//
//  - Dense + ReLU: learns combinations of features ("index straight AND
//    thumb tucked AND …"), much like the hand-written rules but learned from
//    data.
//  - Dropout 0.3: during training, randomly switches off 30% of the units
//    on every step, so the network can't rely on any single feature. This
//    reduces overfitting to the exact frames it was trained on.
//  - Softmax: turns the 24 outputs into probabilities that sum to 1. The top
//    probability is the confidence.
//  - Loss: categorical cross-entropy (penalises low probability on the right
//    letter). Optimiser: Adam, learning rate 0.003.
//
//  INPUTS are the same 85-number vectors as k-NN (sign/features.js toVector),
//  standardised with the same z-score scaler (sign/knn.js fitScaler). The
//  scaler's mean/std are saved WITH the model, because live predictions must
//  be scaled exactly like the training data.
//
//  BACKEND: TF.js could use WebGL, but MediaPipe already uses the GPU every
//  frame. This network is tiny (~11k weights), so the CPU backend is fast
//  enough and avoids competing for the GPU, or stalling it to read results
//  back every frame.
//
//  TF.js (~1.4 MB) is only downloaded the first time the MLP is needed.
// =============================================================================

import { fitScaler, scaleInto } from "./knn.js";

const TFJS_URL = "https://cdn.jsdelivr.net/npm/@tensorflow/tfjs@4.22.0/dist/tf.min.js";
export const MODEL_URL = "indexeddb://handlab-sign-mlp";
const SUMMARY_KEY = "handlab.mlp.summary";

export const MLP_DEFAULTS = { hidden: 64, dropout: 0.3, learningRate: 0.003, epochs: 40, batchSize: 32, validationSplit: 0.15 };

let tfPromise = null;

// Load TF.js once (script tag, sets window.tf) and switch to the CPU backend.
export function loadTF() {
  if (!tfPromise) {
    tfPromise = new Promise((resolve, reject) => {
      if (window.tf) return resolve(window.tf);
      const s = document.createElement("script");
      s.src = TFJS_URL;
      s.onload = () => resolve(window.tf);
      s.onerror = () => reject(new Error("Couldn't load TensorFlow.js (check your internet connection)."));
      document.head.appendChild(s);
    }).then(async (tf) => {
      await tf.setBackend("cpu");
      await tf.ready();
      console.info(`[HandLab] TensorFlow.js ${tf.version.tfjs} ready (backend: ${tf.getBackend()})`);
      return tf;
    });
    tfPromise.catch(() => { tfPromise = null; }); // allow a retry later
  }
  return tfPromise;
}

export function buildModel(tf, inputDim, numClasses, opts = MLP_DEFAULTS) {
  const model = tf.sequential();
  model.add(tf.layers.dense({ inputShape: [inputDim], units: opts.hidden, activation: "relu" }));
  model.add(tf.layers.dropout({ rate: opts.dropout }));
  model.add(tf.layers.dense({ units: opts.hidden, activation: "relu" }));
  model.add(tf.layers.dropout({ rate: opts.dropout }));
  model.add(tf.layers.dense({ units: numClasses, activation: "softmax" }));
  model.compile({ optimizer: tf.train.adam(opts.learningRate), loss: "categoricalCrossentropy", metrics: ["accuracy"] });
  return model;
}

// Train on raw feature vectors X (n x D, Float32Array) with class indices y.
// onEpoch(epoch, logs) is called after every epoch with
// { loss, acc, val_loss, val_acc } (the val_ entries only if validationSplit > 0).
// Returns an MLPClassifier (the caller owns it: call dispose() when done).
export async function trainMLP(tf, X, y, n, D, numClasses, opts = {}) {
  const o = { ...MLP_DEFAULTS, ...opts };
  const scaler = fitScaler(X, n, D);

  // SHUFFLE FIRST: Keras/TF.js `validationSplit` takes the LAST x% of the
  // rows as validation data, before any shuffling. Our rows are grouped by
  // letter, so without this the validation set would be just the last few
  // letters, a very misleading accuracy.
  const order = Array.from({ length: n }, (_, i) => i);
  for (let i = n - 1; i > 0; i--) { const j = Math.floor(Math.random() * (i + 1)); [order[i], order[j]] = [order[j], order[i]]; }
  const Xs = new Float32Array(n * D);
  const ys = new Int32Array(n);
  for (let r = 0; r < n; r++) {
    scaleInto(X.subarray(order[r] * D, order[r] * D + D), scaler, Xs, r * D);
    ys[r] = y[order[r]];
  }

  const model = buildModel(tf, D, numClasses, o);
  const xt = tf.tensor2d(Xs, [n, D]);
  const yt = tf.tidy(() => tf.oneHot(tf.tensor1d(ys, "int32"), numClasses));
  try {
    await model.fit(xt, yt, {
      epochs: o.epochs,
      batchSize: o.batchSize,
      validationSplit: o.validationSplit,
      shuffle: true,
      callbacks: { onEpochEnd: async (epoch, logs) => { if (o.onEpoch) await o.onEpoch(epoch, logs); } },
    });
  } finally {
    xt.dispose();
    yt.dispose();
  }
  return new MLPClassifier(tf, model, scaler, numClasses);
}

export class MLPClassifier {
  constructor(tf, model, scaler, numClasses, meta = {}) {
    this.tf = tf;
    this.model = model;
    this.scaler = scaler;
    this.numClasses = numClasses;
    this.meta = meta;
    this.q = new Float32Array(scaler.mean.length);
  }

  // One vector -> probs (Float32Array(numClasses)). Returns { cls, conf }.
  predict(x, probs) {
    scaleInto(x, this.scaler, this.q);
    const out = this.tf.tidy(() => this.model.predict(this.tf.tensor2d(this.q, [1, this.q.length])).dataSync());
    let cls = 0;
    for (let c = 0; c < this.numClasses; c++) {
      probs[c] = out[c];
      if (out[c] > out[cls]) cls = c;
    }
    return { cls, conf: out[cls] };
  }

  // Many vectors at once (used by evaluation): X is n x D. Returns class indices.
  predictBatch(X, n, D) {
    const Xs = new Float32Array(n * D);
    for (let i = 0; i < n; i++) scaleInto(X.subarray(i * D, i * D + D), this.scaler, Xs, i * D);
    return this.tf.tidy(() => this.model.predict(this.tf.tensor2d(Xs, [n, D])).argMax(1).dataSync());
  }

  // Save the network AND the scaler (+ some info) to IndexedDB.
  // (`url` lets the test page save somewhere else, never over your model.)
  async save(info, url = MODEL_URL) {
    this.meta = { ...info, mean: Array.from(this.scaler.mean), std: Array.from(this.scaler.std), numClasses: this.numClasses };
    this.model.setUserDefinedMetadata(this.meta);
    await this.model.save(url);
    if (url !== MODEL_URL) return;
    try {
      localStorage.setItem(SUMMARY_KEY, JSON.stringify(info));
    } catch {
      // only used to show "trained" before TF.js loads
    }
  }

  dispose() {
    this.model.dispose();
  }
}

// Load the saved model (null if there isn't one).
export async function loadSavedMLP(tf, url = MODEL_URL) {
  try {
    const model = await tf.loadLayersModel(url);
    const meta = model.getUserDefinedMetadata() || {};
    if (!meta.mean || !meta.std) return null;
    const scaler = { mean: Float32Array.from(meta.mean), std: Float32Array.from(meta.std) };
    return new MLPClassifier(tf, model, scaler, meta.numClasses, meta);
  } catch {
    return null; // nothing saved yet
  }
}

// Info about the saved model without loading TF.js (from localStorage).
export function savedSummary() {
  try {
    return JSON.parse(localStorage.getItem(SUMMARY_KEY) || "null");
  } catch {
    return null;
  }
}
