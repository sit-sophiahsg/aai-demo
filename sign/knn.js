// =============================================================================
//  sign/knn.js: k-nearest-neighbours classifier (plain JS, no training)
// =============================================================================
//
//  IDEA: to classify a new hand, find the k recorded samples most similar to
//  it and let them vote. There is no training step, the data IS the model.
//
//  1. STANDARDISE (z-score): every feature is rescaled to mean 0, std 1 using
//     the training data's statistics: x' = (x - mean) / std.
//     Without this, features with a large numeric range would dominate the
//     distance and features with a small range would barely count. The same
//     scaler is reused by the MLP (sign/mlp.js).
//
//  2. DISTANCE between the new vector and every training vector:
//       euclidean: sqrt(sum (a_i - b_i)^2)     straight-line distance
//       cosine:    1 - (a . b) / (|a| |b|)       compares direction only,
//                                                ignores overall magnitude
//
//  3. VOTE: the k closest samples vote for their letter, each weighted by
//     1 / distance (closer neighbours count more). Normalised votes act as
//     class probabilities; the top one is the confidence.
//
//  4. UNKNOWN-POSE GATE: k-NN always answers *something*, even for a hand
//     that isn't making any letter. At fit time we measure how far training
//     samples typically are from their nearest neighbour (95th percentile).
//     If a new hand is much farther from everything than that, it's probably
//     not a letter we know, so confidence is scaled down (see noveltyGate).
//
//  Cost: n samples x D features per prediction (2400 x 85 ≈ 200k
//  multiply-adds), well under a millisecond, fine to run every frame.
// =============================================================================

const NOVELTY_PERCENTILE = 0.95;
const NOVELTY_FACTOR = 2.0;   // distances above (p95 x this) are "unknown"

// ---- Standardisation (shared with the MLP) ----

// X: Float32Array of n rows x D columns. Returns { mean, std } per column.
export function fitScaler(X, n, D) {
  const mean = new Float32Array(D), std = new Float32Array(D);
  for (let i = 0; i < n; i++) for (let j = 0; j < D; j++) mean[j] += X[i * D + j];
  for (let j = 0; j < D; j++) mean[j] /= n || 1;
  for (let i = 0; i < n; i++) for (let j = 0; j < D; j++) { const d = X[i * D + j] - mean[j]; std[j] += d * d; }
  // A feature that never changes would divide by zero; std 1 leaves it as-is.
  for (let j = 0; j < D; j++) std[j] = Math.sqrt(std[j] / (n || 1)) || 1;
  return { mean, std };
}

export function scaleInto(x, scaler, out, offset = 0) {
  const { mean, std } = scaler;
  for (let j = 0; j < mean.length; j++) out[offset + j] = (x[j] - mean[j]) / std[j];
  return out;
}

// Seeded random numbers (so the novelty sample is the same every run).
function mulberry32(seed) {
  return () => {
    seed |= 0; seed = (seed + 0x6d2b79f5) | 0;
    let t = Math.imul(seed ^ (seed >>> 15), 1 | seed);
    t = (t + Math.imul(t ^ (t >>> 7), 61 | t)) ^ t;
    return ((t ^ (t >>> 14)) >>> 0) / 4294967296;
  };
}

export class KNN {
  constructor({ k = 5, metric = "euclidean" } = {}) {
    this.k = k;
    this.metric = metric;
    this.n = 0;
  }

  // X: Float32Array (n x D) of RAW feature vectors, y: class index per row.
  fit(X, y, n, D, numClasses) {
    this.D = D;
    this.n = n;
    this.numClasses = numClasses;
    this.y = Int32Array.from(y.subarray ? y.subarray(0, n) : y.slice(0, n));
    this.scaler = fitScaler(X, n, D);
    // Store the standardised copy once, so predictions only scale the query.
    this.Xs = new Float32Array(n * D);
    for (let i = 0; i < n; i++) scaleInto(X.subarray(i * D, i * D + D), this.scaler, this.Xs, i * D);
    // For cosine distance, pre-compute each row's length.
    this.norms = new Float32Array(n);
    for (let i = 0; i < n; i++) {
      let s = 0;
      for (let j = 0; j < D; j++) s += this.Xs[i * D + j] ** 2;
      this.norms[i] = Math.sqrt(s) || 1;
    }
    this.q = new Float32Array(D);          // scratch: the scaled query
    this.kd = new Float32Array(this.k);    // scratch: k best distances
    this.ki = new Int32Array(this.k);      // scratch: k best indices
    this.threshold = this.computeNoveltyThreshold();
    return this;
  }

  dist(i) {
    const D = this.D, Xs = this.Xs, q = this.q, o = i * D;
    if (this.metric === "cosine") {
      let dot = 0;
      for (let j = 0; j < D; j++) dot += Xs[o + j] * q[j];
      return 1 - dot / (this.norms[i] * this.qNorm);
    }
    let s = 0;
    for (let j = 0; j < D; j++) { const d = Xs[o + j] - q[j]; s += d * d; }
    return Math.sqrt(s);
  }

  setQuery(x) {
    scaleInto(x, this.scaler, this.q);
    let s = 0;
    for (let j = 0; j < this.D; j++) s += this.q[j] * this.q[j];
    this.qNorm = Math.sqrt(s) || 1;
  }

  // Find the k nearest rows to the current query (skipping row `skip`).
  // Keeps a small sorted list instead of sorting all n distances.
  nearest(skip = -1) {
    const k = Math.min(this.k, this.n - (skip >= 0 ? 1 : 0));
    const kd = this.kd, ki = this.ki;
    let filled = 0;
    for (let i = 0; i < this.n; i++) {
      if (i === skip) continue;
      const d = this.dist(i);
      if (filled < k) {
        let p = filled++;
        while (p > 0 && kd[p - 1] > d) { kd[p] = kd[p - 1]; ki[p] = ki[p - 1]; p--; }
        kd[p] = d; ki[p] = i;
      } else if (d < kd[k - 1]) {
        let p = k - 1;
        while (p > 0 && kd[p - 1] > d) { kd[p] = kd[p - 1]; ki[p] = ki[p - 1]; p--; }
        kd[p] = d; ki[p] = i;
      }
    }
    return filled;
  }

  // x: raw feature vector. probs: Float32Array(numClasses), filled with the
  // weighted votes. Returns { cls, conf, nearest } (nearest = closest distance).
  predict(x, probs) {
    probs.fill(0);
    if (!this.n) return { cls: -1, conf: 0, nearest: Infinity };
    this.setQuery(x);
    const found = this.nearest();
    let total = 0;
    for (let p = 0; p < found; p++) {
      const w = 1 / (this.kd[p] + 1e-3);   // closer neighbours count more
      probs[this.y[this.ki[p]]] += w;
      total += w;
    }
    let cls = 0;
    for (let c = 0; c < this.numClasses; c++) {
      probs[c] /= total || 1;
      if (probs[c] > probs[cls]) cls = c;
    }
    return { cls, conf: probs[cls], nearest: this.kd[0] };
  }

  // Distance from x to its single closest training sample.
  nearestDistance(x) {
    if (!this.n) return Infinity;
    this.setQuery(x);
    let best = Infinity;
    for (let i = 0; i < this.n; i++) { const d = this.dist(i); if (d < best) best = d; }
    return best;
  }

  // 95th percentile of "distance to nearest OTHER training sample", times a
  // safety factor. Measured on up to 300 random rows to keep fit() fast.
  computeNoveltyThreshold() {
    if (this.n < 3) return Infinity;
    const rand = mulberry32(1234);
    const m = Math.min(300, this.n);
    const ds = [];
    const savedK = this.k;
    this.k = 1;
    for (let s = 0; s < m; s++) {
      const i = Math.floor(rand() * this.n);
      this.q.set(this.Xs.subarray(i * this.D, i * this.D + this.D));
      this.qNorm = this.norms[i];
      this.nearest(i);
      ds.push(this.kd[0]);
    }
    this.k = savedK;
    ds.sort((a, b) => a - b);
    return ds[Math.floor(NOVELTY_PERCENTILE * (ds.length - 1))] * NOVELTY_FACTOR;
  }

  // 1 when x is within the usual neighbourhood of the data, falling toward 0
  // as it gets farther away (threshold / distance).
  noveltyGate(nearest) {
    if (!(nearest > this.threshold)) return 1;
    return this.threshold / nearest;
  }
}
