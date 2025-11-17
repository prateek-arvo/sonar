import React, { useState, useRef, useEffect } from "react";
/**
* Sonar Label Verification — Single React component
* ------------------------------------------------
* - Plays a short chirp (2 kHz → 12 kHz) using the page's AudioContext
* - Captures microphone input from the same AudioContext
* - Accumulates several analyser frames during the capture window
* - Computes band-energy ratios (robust across phones)
* - Lets you save a baseline signature and compare incoming signatures
*
* Notes:
* - This version is written as a single functional component (hooks used correctly)
* - It uses an analyser node on the mic media stream and averages FFT frames
* - No external libraries required. Works in modern browsers that expose WebAudio & getUserMedia
*/
export default function SonarVerifierApp() {
 // UI state
 const [recording, setRecording] = useState(false);
 const [bands, setBands] = useState(null); // normalized band vector from last capture
 const [baseline, setBaseline] = useState(null); // saved baseline vector
 const [similarity, setSimilarity] = useState(null);
 const [status, setStatus] = useState("idle");
 // refs to hold runtime objects so they persist across renders
 const audioCtxRef = useRef(null);
 const analyserRef = useRef(null);
 const micStreamRef = useRef(null);
 const rafRef = useRef(null);
 const energyAccRef = useRef(null);
 const frameCountRef = useRef(0);
 // band ranges (Hz) — adjust as needed
 const BAND_RANGES = [
   [2000, 4000],
   [4000, 6000],
   [6000, 8000],
   [8000, 10000],
   [10000, 12000],
 ];
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
 // Build and start the AudioContext + microphone analyser
 const initAudio = async () => {
   // resume existing context if suspended (user gesture required in many browsers)
   if (!audioCtxRef.current) {
     audioCtxRef.current = new (window.AudioContext || window.webkitAudioContext)();
   }
   const audioCtx = audioCtxRef.current;
   // request mic
   try {
     const stream = await navigator.mediaDevices.getUserMedia({ audio: true });
     micStreamRef.current = stream;
     const source = audioCtx.createMediaStreamSource(stream);
     const analyser = audioCtx.createAnalyser();
     analyser.fftSize = 2048; // gives frequencyBinCount = 1024
     analyser.smoothingTimeConstant = 0.2;
     source.connect(analyser);
     analyserRef.current = analyser;
     // prepare accumulator
     energyAccRef.current = new Array(BAND_RANGES.length).fill(0);
     frameCountRef.current = 0;
     return true;
   } catch (err) {
     console.error("getUserMedia failed:", err);
     setStatus("mic-error");
     return false;
   }
 };
 // play a short chirp using the same audio context
 const playChirp = async (duration = 0.4, startFreq = 2000, endFreq = 12000) => {
   if (!audioCtxRef.current) audioCtxRef.current = new (window.AudioContext || window.webkitAudioContext)();
   const ctx = audioCtxRef.current;
   // ensure context is resumed (user gesture may be required)
   if (ctx.state === "suspended") {
     try {
       await ctx.resume();
     } catch (e) {
       console.warn("AudioContext resume failed:", e);
     }
   }
   const osc = ctx.createOscillator();
   const gain = ctx.createGain();
   osc.type = "sine";
   osc.frequency.setValueAtTime(startFreq, ctx.currentTime);
   // ramp to endFreq over duration
   osc.frequency.exponentialRampToValueAtTime(endFreq, ctx.currentTime + duration);
   gain.gain.setValueAtTime(0.6, ctx.currentTime); // moderate volume
   osc.connect(gain).connect(ctx.destination);
   osc.start(ctx.currentTime + 0.02);
   osc.stop(ctx.currentTime + duration + 0.02);
 };
 // main capture function: play chirp AND record mic analyser frames
 const captureOnce = async (captureMs = 700) => {
   setStatus("initializing");
   const ok = await initAudio();
   if (!ok) return;
   // small delay to let analyser stabilize
   await new Promise((r) => setTimeout(r, 120));
   setStatus("capturing");
   setRecording(true);
   // start playing chirp (non-blocking)
   playChirp(0.45, 2000, 12000).catch((e) => console.warn(e));
   const analyser = analyserRef.current;
   const audioCtx = audioCtxRef.current;
   const freqCount = analyser.frequencyBinCount; // 1024 for fftSize=2048
   const floatBins = new Float32Array(freqCount);
   // clear accumulator
   energyAccRef.current = new Array(BAND_RANGES.length).fill(0);
   frameCountRef.current = 0;
   const binHz = audioCtx.sampleRate / analyser.fftSize;
   const start = performance.now();
   // sample loop using requestAnimationFrame for short duration
   const sampleFrame = () => {
     analyser.getFloatFrequencyData(floatBins); // decibels (dBFS)
     // convert from dB to linear magnitude
     const linear = new Float32Array(freqCount);
     for (let i = 0; i < freqCount; i++) linear[i] = Math.pow(10, floatBins[i] / 20);
     // accumulate band energies
     for (let b = 0; b < BAND_RANGES.length; b++) {
       const [low, high] = BAND_RANGES[b];
       let sum = 0;
       // iterate bins — bins correspond to freq = i * binHz
       for (let i = 0; i < freqCount; i++) {
         const freq = i * binHz;
         if (freq >= low && freq < high) sum += linear[i];
         if (freq >= high) break; // speed up
       }
       energyAccRef.current[b] += sum;
     }
     frameCountRef.current += 1;
     if (performance.now() - start < captureMs) {
       rafRef.current = requestAnimationFrame(sampleFrame);
     } else {
       // finish
       finalizeBands();
     }
   };
   rafRef.current = requestAnimationFrame(sampleFrame);
 };
 const finalizeBands = () => {
   // average energies across frames
   const counts = Math.max(frameCountRef.current, 1);
   const avg = energyAccRef.current.map((v) => v / counts);
   const total = avg.reduce((a, b) => a + b, 1e-12);
   const normalized = avg.map((v) => v / total);
   setBands(normalized);
   setRecording(false);
   setStatus("captured");
   // cleanup mic stream but keep audio context for reuse
   if (micStreamRef.current) {
     const tracks = micStreamRef.current.getTracks();
     tracks.forEach((t) => t.stop());
     micStreamRef.current = null;
   }
   if (rafRef.current) {
     cancelAnimationFrame(rafRef.current);
     rafRef.current = null;
   }
 };
 // UI actions
 const handleCapture = async () => {
   setBands(null);
   setSimilarity(null);
   await captureOnce(700);
 };
 const handleSaveBaseline = () => {
   if (!bands) return;
   setBaseline(bands);
   setSimilarity(null);
   setStatus("baseline-saved");
 };
 const handleCompare = () => {
   if (!baseline || !bands) return;
   const sim = cosineSim(baseline, bands);
   setSimilarity(sim);
   setStatus(sim > 0.85 ? "match" : "no-match");
 };
 // cleanup on unmount
 useEffect(() => {
   return () => {
     if (micStreamRef.current) {
       micStreamRef.current.getTracks().forEach((t) => t.stop());
     }
     if (audioCtxRef.current) {
       try {
         audioCtxRef.current.close();
       } catch (e) {
         /* ignore */
       }
       audioCtxRef.current = null;
     }
     if (rafRef.current) cancelAnimationFrame(rafRef.current);
   };
 }, []);
 return (
<div style={{ maxWidth: 720, margin: "24px auto", fontFamily: "system-ui, sans-serif" }}>
<h1 style={{ fontSize: 20, marginBottom: 8 }}>Sonar Label Verification (Prototype)</h1>
<p style={{ marginTop: 0, color: "#333" }}>
       This demo plays a short chirp and records the microphone while the chirp plays. Place the label on a flat surface near your phone's microphone (or on the phone screen if you want a strong coupling). Then press <b>Capture</b>.
</p>
<div style={{ display: "flex", gap: 8, marginTop: 12 }}>
<button
         onClick={handleCapture}
         disabled={recording}
         style={{ padding: "10px 14px", background: "#2563eb", color: "white", border: "none", borderRadius: 8 }}
>
         {recording ? "Capturing..." : "Capture (Play & Record)"}
</button>
<button
         onClick={handleSaveBaseline}
         disabled={!bands}
         style={{ padding: "10px 14px", background: "#059669", color: "white", border: "none", borderRadius: 8 }}
>
         Save Baseline
</button>
<button
         onClick={handleCompare}
         disabled={!bands || !baseline}
         style={{ padding: "10px 14px", background: "#0ea5e9", color: "white", border: "none", borderRadius: 8 }}
>
         Compare to Baseline
</button>
</div>
<div style={{ marginTop: 18, padding: 12, background: "#35bc00ff", borderRadius: 8 }}>
<div style={{ fontSize: 13, color: "#333", marginBottom: 8 }}>
         Status: <b>{status}</b>
</div>
<div style={{ display: "flex", gap: 12 }}>
<div style={{ flex: 1 }}>
<div style={{ fontSize: 12, color: "#666" }}>Last capture bands</div>
<pre style={{ fontSize: 12, marginTop: 6 }}>{bands ? JSON.stringify(bands, null, 2) : "-"}</pre>
</div>
<div style={{ flex: 1 }}>
<div style={{ fontSize: 12, color: "#666" }}>Baseline</div>
<pre style={{ fontSize: 12, marginTop: 6 }}>{baseline ? JSON.stringify(baseline, null, 2) : "-"}</pre>
</div>
</div>
<div style={{ marginTop: 12 }}>
<div style={{ fontSize: 12, color: "#666" }}>Similarity</div>
<div style={{ fontSize: 16, fontWeight: 600 }}>{similarity !== null ? similarity.toFixed(3) : "-"}</div>
<div style={{ fontSize: 12, color: "#888", marginTop: 6 }}>
           Tip: a similarity &gt; 0.85 typically indicates a match for the same phone & label; tune thresholds using your own test data.
</div>
</div>
</div>
</div>
 );
}