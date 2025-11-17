import React, { useState, useRef, useEffect } from "react";

/**
* Sonar Label Verification — High-Security Implementation (A)
* ---------------------------------------------------------
* Flow:
* 1. Play a short chirp (2kHz -> 18kHz)
* 2. Record ~CAPTURE_MS ms of mic audio via MediaRecorder
* 3. Decode audio, split into SEGMENTS time-slices
* 4. For each time-slice compute FFT magnitude and band energies
* 5. Normalize per-segment and compute delta ratios between consecutive segments
* 6. Flatten delta ratios into a single vector and compare via cosine similarity
*
* Notes:
* - Designed for single-phone or cross-phone with normalization; A-level uses 50 segments
* - Uses an internal FFT implementation (radix-2) for portability (no deps)
*/

export default function SonarVerifierApp() {
 const [recording, setRecording] = useState(false);
 const [bandsMatrix, setBandsMatrix] = useState(null);
 const [baseline, setBaseline] = useState(null);
 const [similarity, setSimilarity] = useState(null);
 const [status, setStatus] = useState("idle");

 const mediaRecorderRef = useRef(null);
 const audioChunksRef = useRef([]);

 const SEGMENTS = 50;
 const BAND_RANGES = [
   [2000, 4000],
   [4000, 6000],
   [6000, 8000],
   [8000, 10000],
   [10000, 12000],
 ];
 const CAPTURE_MS = 700;

 // FFT implementation (iterative radix-2)
 function fftMag(segment, fftSize = 512) {
   const N = fftSize;
   const re = new Float64Array(N);
   const im = new Float64Array(N);
   for (let i = 0; i < Math.min(segment.length, N); i++) re[i] = segment[i];

   // bit-reversal
   let j = 0;
   for (let i = 1; i < N - 1; i++) {
     let bit = N >> 1;
     for (; j & bit; bit >>= 1) j ^= bit;
     j ^= bit;
     if (i < j) {
       const tmp = re[i];
       re[i] = re[j];
       re[j] = tmp;
     }
   }

   for (let len = 2; len <= N; len <<= 1) {
     const half = len >> 1;
     const theta = (-2 * Math.PI) / len;
     const wmulRe = Math.cos(theta);
     const wmulIm = Math.sin(theta);
     for (let i = 0; i < N; i += len) {
       let wr = 1;
       let wi = 0;
       for (let j2 = 0; j2 < half; j2++) {
         const idx = i + j2;
         const idx2 = idx + half;
         const xr = re[idx2] * wr - im[idx2] * wi;
         const xi = re[idx2] * wi + im[idx2] * wr;

         re[idx2] = re[idx] - xr;
         im[idx2] = im[idx] - xi;
         re[idx] = re[idx] + xr;
         im[idx] = im[idx] + xi;

         const tmpwr = wr * wmulRe - wi * wmulIm;
         wi = wr * wmulIm + wi * wmulRe;
         wr = tmpwr;
       }
     }
   }

   const mag = new Float64Array(N / 2);
   for (let i = 0; i < N / 2; i++) mag[i] = Math.hypot(re[i], im[i]);
   return mag;
 }

 function computeBandEnergies(mag, sampleRate, fftSize) {
   const binHz = sampleRate / fftSize;
   const energies = BAND_RANGES.map(([low, high]) => {
     let sum = 0;
     const start = Math.max(0, Math.floor(low / binHz));
     const end = Math.min(mag.length - 1, Math.floor(high / binHz));
     for (let i = start; i <= end; i++) sum += mag[i];
     return sum;
   });
   return energies;
 }

 // helper: cosine similarity
 const cosineSim = (a, b) => {
   if (!a || !b || a.length !== b.length) return 0;
   let num = 0;
   let na = 0;
   let nb = 0;
   for (let i = 0; i < a.length; i++) {
     num += a[i] * b[i];
     na += a[i] * a[i];
     nb += b[i] * b[i];
   }
   return num / (Math.sqrt(na) * Math.sqrt(nb) + 1e-12);
 };

 // capture flow: record audio blob while playing chirp, then process
 const captureAndProcess = async () => {
   setStatus("requesting-mic");
   audioChunksRef.current = [];

   let stream;
   try {
     stream = await navigator.mediaDevices.getUserMedia({ audio: true });
   } catch (e) {
     console.error("getUserMedia failed", e);
     setStatus("mic-error");
     return null;
   }

   const mimeType = MediaRecorder.isTypeSupported("audio/webm") ? "audio/webm" : "audio/wav";
   const mr = new MediaRecorder(stream, { mimeType });
   mediaRecorderRef.current = mr;

   const chunks = [];
   mr.ondataavailable = (ev) => chunks.push(ev.data);

   const stopPromise = new Promise((resolve) => {
     mr.onstop = () => resolve(new Blob(chunks, { type: mimeType }));
   });

   mr.start();
   setRecording(true);
   setStatus("playing-chirp");

   // play chirp via AudioContext
   const ctx = new (window.AudioContext || window.webkitAudioContext)();
   try { await ctx.resume(); } catch (e) {}
   const osc = ctx.createOscillator();
   const gain = ctx.createGain();
   const duration = 0.25;
   const startFreq = 2000;
   const endFreq = 18000;

   osc.type = "sine";
   osc.frequency.setValueAtTime(startFreq, ctx.currentTime + 0.01);
   osc.frequency.exponentialRampToValueAtTime(endFreq, ctx.currentTime + duration);
   gain.gain.setValueAtTime(0.6, ctx.currentTime + 0.01);
   osc.connect(gain).connect(ctx.destination);
   osc.start(ctx.currentTime + 0.01);
   osc.stop(ctx.currentTime + duration + 0.01);

   // stop recording after CAPTURE_MS
   setTimeout(() => {
     try { mr.stop(); } catch (e) { console.warn(e); }
     setRecording(false);
     setStatus("processing");
   }, CAPTURE_MS);

   const blob = await stopPromise;

   // stop tracks
   try { stream.getTracks().forEach((t) => t.stop()); } catch (e) {}

   // decode audio and compute bands matrix
   const arrayBuffer = await blob.arrayBuffer();
   const decodeCtx = new (window.AudioContext || window.webkitAudioContext)();
   const audioBuffer = await decodeCtx.decodeAudioData(arrayBuffer);
   const sampleRate = audioBuffer.sampleRate;
   const data = audioBuffer.getChannelData(0);

   const totalSamples = data.length;
   const segLen = Math.floor(totalSamples / SEGMENTS) || 1;
   const fftSize = 512;

   const bandsMat = [];
   for (let s = 0; s < SEGMENTS; s++) {
     const start = s * segLen;
     const end = Math.min(start + segLen, totalSamples);
     const segment = data.slice(start, end);

     // window & zero-pad to fftSize
     const windowed = new Float64Array(fftSize);
     for (let i = 0; i < fftSize; i++) {
       const x = i < segment.length ? segment[i] : 0;
       const w = 0.5 * (1 - Math.cos((2 * Math.PI * i) / (fftSize - 1)));
       windowed[i] = x * w;
     }

     const mag = fftMag(windowed, fftSize);
     const energies = computeBandEnergies(mag, sampleRate, fftSize);
     bandsMat.push(energies);
   }

   // normalize each segment's bands
   const normMat = bandsMat.map((arr) => {
     const sum = arr.reduce((a, b) => a + b, 1e-12);
     return arr.map((v) => v / sum);
   });

   setBandsMatrix(normMat);

   // compute deltas (ratio cur/prev) flatten
   const deltas = [];
   for (let i = 1; i < normMat.length; i++) {
     for (let j = 0; j < normMat[0].length; j++) {
       const prev = normMat[i - 1][j] + 1e-12;
       const cur = normMat[i][j] + 1e-12;
       deltas.push(cur / prev);
     }
   }

   setStatus("captured");
   return deltas;
 };

 const handleSaveBaseline = async () => {
   setStatus("saving-baseline");
   const deltas = await captureAndProcess();
   if (deltas) {
     setBaseline(deltas);
     setStatus("baseline-saved");
   } else {
     setStatus("baseline-failed");
   }
 };

 const handleCompare = async () => {
   if (!baseline) {
     setStatus("no-baseline");
     return;
   }
   setStatus("capturing-compare");
   const deltas = await captureAndProcess();
   if (!deltas) {
     setStatus("capture-failed");
     return;
   }
   const sim = cosineSim(baseline, deltas);
   setSimilarity(sim);
   setStatus(sim > 0.92 ? "match" : "no-match");
 };

 // simple capture for UI (capture and display bands only)
 const handleCaptureOnly = async () => {
   setStatus("capturing-only");
   await captureAndProcess();
 };

 useEffect(() => {
   return () => {
     if (mediaRecorderRef.current && mediaRecorderRef.current.state !== "inactive") {
       try { mediaRecorderRef.current.stop(); } catch (e) {}
     }
   };
 }, []);

 return (
   <div style={{ maxWidth: 760, margin: "18px auto", fontFamily: "system-ui, sans-serif" }}>
     <h1 style={{ fontSize: 20, marginBottom: 8 }}>Sonar Label Verification — High Security (A)</h1>
     <p style={{ marginTop: 0 }}>Protocol: play a short chirp, record ~{CAPTURE_MS} ms, slice into {SEGMENTS} segments, compute {BAND_RANGES.length} bands per segment, compute delta ratios, compare via cosine similarity.</p>

     <div style={{ display: "flex", gap: 8, marginTop: 12 }}>
       <button onClick={handleCaptureOnly} disabled={recording} style={{ padding: "10px 14px", background: "#2563eb", color: "white", border: "none", borderRadius: 8 }}>
         {recording ? "Recording..." : "Capture (Play & Record)"}
       </button>
       <button onClick={handleSaveBaseline} style={{ padding: "10px 14px", background: "#059669", color: "white", border: "none", borderRadius: 8 }}>
         Capture & Save Baseline
       </button>
       <button onClick={handleCompare} style={{ padding: "10px 14px", background: "#0ea5e9", color: "white", border: "none", borderRadius: 8 }}>
         Capture & Compare
       </button>
     </div>

     <div style={{ marginTop: 18, padding: 12, background: "#fafafa", borderRadius: 8 }}>
       <div style={{ fontSize: 13, color: "#333", marginBottom: 8 }}>Status: <b>{status}</b></div>

       <div style={{ display: "flex", gap: 12 }}>
         <div style={{ flex: 1 }}>
           <div style={{ fontSize: 12, color: "#666" }}>Last capture (first 3 segments):</div>
           <pre style={{ fontSize: 12, marginTop: 6 }}>{bandsMatrix ? JSON.stringify(bandsMatrix.slice(0,3), null, 2) : "-"}</pre>
         </div>

         <div style={{ flex: 1 }}>
           <div style={{ fontSize: 12, color: "#666" }}>Baseline sample (first 10 values):</div>
           <pre style={{ fontSize: 12, marginTop: 6 }}>{baseline ? JSON.stringify(baseline.slice(0,10), null, 2) : "-"}</pre>
         </div>
       </div>

       <div style={{ marginTop: 12 }}>
         <div style={{ fontSize: 12, color: "#666" }}>Similarity</div>
         <div style={{ fontSize: 16, fontWeight: 600 }}>{similarity !== null ? similarity.toFixed(4) : "-"}</div>
         <div style={{ fontSize: 12, color: "#888", marginTop: 6 }}>Threshold: 0.92 (tune with your data)</div>
       </div>
     </div>
   </div>
 );
}