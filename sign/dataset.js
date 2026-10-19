// =============================================================================
//  sign/dataset.js: recorded training samples (IndexedDB + JSON export/import)
// =============================================================================
//
//  One SAMPLE = one frame of one hand making one letter:
//    {
//      uid:    "k3x9…",        unique id (so merged files don't duplicate)
//      letter: "A",            the label (the class)
//      signer: "Sophia",       who recorded it (for leave-one-signer-out tests)
//      hand:   "Right",        handedness; left hands are mirrored in preprocessing
//      t:      1760000000000,  when it was recorded (ms since 1970)
//      lm:     [63 numbers],   RAW landmarks (x, y, z) x 21, in canvas pixels
//    }
//  We store raw landmarks rather than finished feature vectors, so changing
//  the preprocessing later (sign/features.js) never invalidates old data.
//
//  All samples are also kept in memory (`samples`) for fast counting and
//  training. IndexedDB is the persistent copy; if it's unavailable (private
//  window, blocked storage) the app still works and warns you to Export.
// =============================================================================

import { STATIC_LETTERS } from "./letters.js";

const STORE = "samples";
const FORMAT = "handlab-sign-samples";
const FORMAT_VERSION = 1;

let db = null;
let samples = [];
let dbName = "handlab-signs";

// ---- Small promise wrappers around the IndexedDB callback API ----
function req(r) {
  return new Promise((resolve, reject) => {
    r.onsuccess = () => resolve(r.result);
    r.onerror = () => reject(r.error);
  });
}

function done(tx) {
  return new Promise((resolve, reject) => {
    tx.oncomplete = () => resolve();
    tx.onerror = tx.onabort = () => reject(tx.error);
  });
}

// Open (or create) the database and load every sample into memory.
// Returns { persistent: true/false, count }.
export async function openDataset(name = dbName) {
  dbName = name;
  try {
    const open = indexedDB.open(dbName, 1);
    open.onupgradeneeded = () => {
      const store = open.result.createObjectStore(STORE, { keyPath: "uid" });
      store.createIndex("letter", "letter");
    };
    db = await req(open);
    samples = await req(db.transaction(STORE).objectStore(STORE).getAll());
  } catch (err) {
    console.warn("[HandLab] IndexedDB unavailable, samples will only live in memory:", err);
    db = null;
  }
  return { persistent: !!db, count: samples.length };
}

export const isPersistent = () => !!db;
export const allSamples = () => samples;

function makeUid() {
  try {
    return crypto.randomUUID();
  } catch {
    return Date.now().toString(36) + Math.random().toString(36).slice(2, 10);
  }
}

// Add one recorded sample (written to IndexedDB in the background).
export function addSample({ letter, signer, hand, lm }) {
  const s = { uid: makeUid(), letter, signer: signer || "anonymous", hand, t: Date.now(), lm };
  samples.push(s);
  if (db) {
    try {
      db.transaction(STORE, "readwrite").objectStore(STORE).put(s).onerror = (e) =>
        console.warn("[HandLab] Failed to save sample:", e.target.error);
    } catch (err) {
      console.warn("[HandLab] Failed to save sample:", err);
    }
  }
  return s;
}

// Delete every sample of one letter. Returns how many were removed.
export async function deleteLetter(letter) {
  const before = samples.length;
  const removed = samples.filter((s) => s.letter === letter);
  samples = samples.filter((s) => s.letter !== letter);
  if (db) {
    try {
      const tx = db.transaction(STORE, "readwrite");
      const store = tx.objectStore(STORE);
      for (const s of removed) store.delete(s.uid);
      await done(tx);
    } catch (err) {
      console.warn("[HandLab] Failed to delete from IndexedDB:", err);
    }
  }
  return before - samples.length;
}

// Delete everything (used by the test page; not exposed in the UI).
export async function clearAll() {
  samples = [];
  if (db) {
    try {
      const tx = db.transaction(STORE, "readwrite");
      tx.objectStore(STORE).clear();
      await done(tx);
    } catch (err) {
      console.warn("[HandLab] Failed to clear IndexedDB:", err);
    }
  }
}

// Counts for the overview: per letter, per signer, and per signer+letter.
export function counts() {
  const byLetter = Object.fromEntries(STATIC_LETTERS.map((l) => [l, 0]));
  const bySigner = {};
  for (const s of samples) {
    byLetter[s.letter] = (byLetter[s.letter] || 0) + 1;
    bySigner[s.signer] = (bySigner[s.signer] || 0) + 1;
  }
  return { byLetter, bySigner, total: samples.length };
}

// ---- JSON export / import ----

export function exportJSON() {
  return JSON.stringify({ format: FORMAT, version: FORMAT_VERSION, exported: new Date().toISOString(), samples });
}

function isValidSample(s) {
  return s && typeof s.uid === "string" && STATIC_LETTERS.includes(s.letter) &&
    (s.hand === "Left" || s.hand === "Right") && Array.isArray(s.lm) && s.lm.length === 63 &&
    s.lm.every((v) => typeof v === "number" && Number.isFinite(v));
}

// Merge samples from an exported file. Samples already present (same uid)
// are skipped, so importing the same file twice is harmless.
// Returns { added, duplicates, invalid }.
export async function importJSON(text) {
  const data = JSON.parse(text);
  if (!data || data.format !== FORMAT || !Array.isArray(data.samples)) {
    throw new Error("Not a Hand Lab sample file.");
  }
  const have = new Set(samples.map((s) => s.uid));
  const fresh = [];
  let duplicates = 0, invalid = 0;
  for (const s of data.samples) {
    if (!isValidSample(s)) invalid++;
    else if (have.has(s.uid)) duplicates++;
    else {
      const clean = { uid: s.uid, letter: s.letter, signer: String(s.signer || "anonymous").slice(0, 40),
        hand: s.hand, t: Number(s.t) || 0, lm: s.lm };
      fresh.push(clean);
      have.add(s.uid);
    }
  }
  samples.push(...fresh);
  if (db && fresh.length) {
    try {
      const tx = db.transaction(STORE, "readwrite");
      const store = tx.objectStore(STORE);
      for (const s of fresh) store.put(s);
      await done(tx);
    } catch (err) {
      console.warn("[HandLab] Imported samples could not be saved to IndexedDB:", err);
    }
  }
  return { added: fresh.length, duplicates, invalid };
}
