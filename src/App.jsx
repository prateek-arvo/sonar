import React, { useState, useRef, useEffect } from "react";

/**
* Sonar Label Verification — Fixed capture implementation
* ------------------------------------------------------
* Replaces MediaRecorder-based recording (which can hang on some browsers/Safari)
* with a ScriptProcessor-based direct-buffer capture using WebAudio. This is
* more reliable across browsers and avoids MIME/MediaRecorder incompatibility.
*
* Key changes:
* - Create AudioContext and call resume() inside the user gesture handler
* - Use getUserMedia to obtain mic stream, connect to an AudioContext MediaStreamSource
* - Use ScriptProcessorNode to capture float samples into buffers for CAPTURE_MS
* - Play chirp via the same AudioContext (guarantees audio is generated during the gesture)
* - Process the collected raw samples directly (no decode step)
*/

export default function SonarVerifierApp() {
 const [recording, setRecording] = useState(false);
 const [bandsMatrix, setBandsMatrix] = useState(null);
 const [baseline, setBaseline] = useState(null);
 const [similarity, setSimilarity] = useState(null);
 const [status, setStatus] = useState("idle");

 const audioCtxRef = useRef(null);
 const streamRef = useRef(null);
 const procRef = useRef(null);
 const capturedChunksRef = useRef([]);

 const SEGMENTS = 50;
 const BAND_RANGES = [
   [2000, 4000],
   [4000, 6000],
   [6000, 8000],
   [8000, 10000],
   [10000, 12000],
 ];
 const CAPTURE_MS = 700;

 // simple iterative FFT (radix-2) returning magnitude
 function fftMag(segment, fftSize = 512) {
   const N = fftSize;
   const re = new Float64Array(N);
   const im = new Float64Array(N);
   for (let i = 0; i < Math.min(segment.length, N); i++) re[i] = segment[i];

   // bit reversal
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
   return BAND_RANGES.map(([low, high]) => {
     let sum = 0;
     const start = Math.max(0, Math.floor(low / binHz));
     const end = Math.min(mag.length - 1, Math.floor(high / binHz));
     for (let i = start; i <= end; i++) sum += mag[i];
     return sum;
   });
 }

 const cosineSim = (a, b) => {
   if (!a || !b || a.length !== b.length) return 0;
   let num = 0,
     na = 0,
     nb = 0;
   for (let i = 0; i < a.length; i++) {
     num += a[i] * b[i];
     na += a[i] * a[i];
     nb += b[i] * b[i];
   }
   return num / (Math.sqrt(na) * Math.sqrt(nb) + 1e-12);
 };

 // play chirp on provided AudioContext (returns after ended)
 const playChirpViaCtx = (ctx, { duration = 0.25, startFreq = 2000, endFreq = 12000 } = {}) => {
   const osc = ctx.createOscillator();
   const gain = ctx.createGain();
   osc.type = "sine";
   osc.frequency.setValueAtTime(startFreq, ctx.currentTime + 0.005);
   try {
     osc.frequency.exponentialRampToValueAtTime(endFreq, ctx.currentTime + duration);
   } catch (e) {
     osc.frequency.linearRampToValueAtTime(endFreq, ctx.currentTime + duration);
   }
   gain.gain.setValueAtTime(0.6, ctx.currentTime + 0.005);
   osc.connect(gain).connect(ctx.destination);
   osc.start(ctx.currentTime + 0.005);
   osc.stop(ctx.currentTime + duration + 0.005);
   return new Promise((res) => {
     osc.onended = () => {
       try {
         osc.disconnect();
         gain.disconnect();
       } catch (e) {}
       res();
     };
   });
 };

 // Core: capture raw audio samples using ScriptProcessor and play chirp concurrently
 const captureAndProcess = async () => {
   setStatus("requesting-mic");
   setBandsMatrix(null);
   setSimilarity(null);
   capturedChunksRef.current = [];

   // create/reuse AudioContext synchronously (must be in user gesture)
   if (!audioCtxRef.current) {
     try {
       audioCtxRef.current = new (window.AudioContext || window.webkitAudioContext)();
     } catch (e) {
       console.error("AudioContext creation failed", e);
       setStatus("audioctx-fail");
       return null;
     }
   }
   const ctx = audioCtxRef.current;

   // resume if suspended
   if (ctx.state === "suspended") {
     try {
       await ctx.resume();
     } catch (e) {
       console.warn("AudioContext resume failed", e);
     }
   }

   // get mic stream (this will prompt the user)
   let stream;
   try {
     stream = await navigator.mediaDevices.getUserMedia({ audio: true });
   } catch (e) {
     console.error("getUserMedia failed", e);
     setStatus("mic-error");
     return null;
   }
   streamRef.current = stream;

   // connect stream to audio context
   const source = ctx.createMediaStreamSource(stream);

   // buffer capture using ScriptProcessorNode
   const bufferSize = 4096; // power of two, widely supported
   const proc = (ctx.createScriptProcessor || ctx.createJavaScriptNode).call(ctx, bufferSize, 1, 1);
   procRef.current = proc;

   proc.onaudioprocess = (e) => {
     try {
       const inData = e.inputBuffer.getChannelData(0);
       // copy data to a regular Float32Array and push
       capturedChunksRef.current.push(new Float32Array(inData));
     } catch (err) {
       console.warn("onaudioprocess error", err);
     }
   };

   source.connect(proc);
   proc.connect(ctx.destination); // connect to destination to keep node alive on some browsers

   setRecording(true);
   setStatus("playing-chirp");

   // play chirp and capture for CAPTURE_MS
   try {
     // start chirp (non-blocking)
     playChirpViaCtx(ctx, { duration: 0.25, startFreq: 2000, endFreq: 12000 }).catch((e) => console.warn(e));
   } catch (e) {
     console.warn("playChirp failed", e);
   }

   // wait CAPTURE_MS then stop
   await new Promise((res) => setTimeout(res, CAPTURE_MS));

   // stop capturing
   try {
     proc.disconnect();
     source.disconnect();
   } catch (e) {}
   try {
     // stop tracks
     stream.getTracks().forEach((t) => t.stop());
   } catch (e) {}

   setRecording(false);
   setStatus("processing");

   // concatenate captured chunks into a single Float32Array
   let totalLen = 0;
   for (const c of capturedChunksRef.current) totalLen += c.length;
   const all = new Float32Array(totalLen);
   let offset = 0;
   for (const c of capturedChunksRef.current) {
     all.set(c, offset);
     offset += c.length;
   }

   // now process `all` as raw samples at ctx.sampleRate
   const sampleRate = ctx.sampleRate;
   const totalSamples = all.length;
   const segLen = Math.floor(totalSamples / SEGMENTS) || 1;
   const fftSize = 512;

   const bandsMat = [];
   for (let s = 0; s < SEGMENTS; s++) {
     const start = s * segLen;
     const end = Math.min(start + segLen, totalSamples);
     const segment = all.subarray(start, end);

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

   const normMat = bandsMat.map((arr) => {
     const sum = arr.reduce((a, b) => a + b, 1e-12);
     return arr.map((v) => v / sum);
   });

   setBandsMatrix(normMat);

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

 const handleCaptureOnly = async () => {
   setStatus("capturing-only");
   await captureAndProcess();
 };

 useEffect(() => {
   return () => {
     // cleanup audio context + stream on unmount
     try {
       if (streamRef.current) streamRef.current.getTracks().forEach((t) => t.stop());
     } catch (e) {}
     try {
       if (procRef.current) procRef.current.disconnect();
     } catch (e) {}
     try {
       if (audioCtxRef.current) {
         audioCtxRef.current.close();
         audioCtxRef.current = null;
       }
     } catch (e) {}
   };
 }, []);

 return (
   <div style={{ maxWidth: 760, margin: "18px auto", fontFamily: "system-ui, sans-serif" }}>
     <h1 style={{ fontSize: 20, marginBottom: 8 }}>Sonar Label Verification — Fixed Capture</h1>
     <p style={{ marginTop: 0 }}>Protocol: play short chirp, record ~{CAPTURE_MS} ms, slice into {SEGMENTS} segments and compute band deltas.</p>

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