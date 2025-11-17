import React, { useState, useRef, useEffect } from "react";

/**
 * Combined Contact-Mode + Vibration Sonar Verifier (React)
 * -------------------------------------------------------
 * Features:
 * - Contact-mode chirp (play sweep through AudioContext while label sits on screen)
 * - Optional vibration pulse pattern (navigator.vibrate) to excite mechanical damping
 * - Raw-sample capture via ScriptProcessorNode for max browser compatibility
 * - Time-frequency analysis split into SEGMENTS x BANDS
 * - Delta-ratio features + vibration-modulation features fused into vector
 * - Baseline save & compare (cosine similarity)
 *
 * Notes/Caveats:
 * - Must be served over HTTPS (or localhost) to use getUserMedia on mobile browsers
 * - iOS Safari support for navigator.vibrate is limited; vibration is optional (app will still work without it)
 * - Encourage the user to place the printed label flat on the screen, press gently, then tap "Capture"
 */

export default function CombinedSonarVerifier() {
  const [status, setStatus] = useState("idle");
  const [recording, setRecording] = useState(false);
  const [bandsMatrix, setBandsMatrix] = useState(null);
  const [baseline, setBaseline] = useState(null);
  const [similarity, setSimilarity] = useState(null);

  const audioCtxRef = useRef(null);
  const procRef = useRef(null);
  const streamRef = useRef(null);
  const capturedChunksRef = useRef([]);

  // security/high-res parameters
  const SEGMENTS = 50; // A-level
  const FFT_SIZE = 512;
  const BAND_RANGES = [
    [2000, 4000],
    [4000, 6000],
    [6000, 8000],
    [8000, 10000],
    [10000, 12000],
  ];
  const CAPTURE_MS = 900; // a bit longer to capture pre/post chirp + vibration

  // small FFT (radix-2) returning magnitude
  function fftMag(segment, fftSize = FFT_SIZE) {
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

  // play chirp on ctx and return when complete
  const playChirp = (ctx, { duration = 0.3, startFreq = 2000, endFreq = 14000 } = {}) => {
    const osc = ctx.createOscillator();
    const gain = ctx.createGain();
    osc.type = "sine";
    osc.frequency.setValueAtTime(startFreq, ctx.currentTime + 0.01);
    try {
      osc.frequency.exponentialRampToValueAtTime(endFreq, ctx.currentTime + duration);
    } catch (e) {
      osc.frequency.linearRampToValueAtTime(endFreq, ctx.currentTime + duration);
    }
    gain.gain.setValueAtTime(0.55, ctx.currentTime + 0.01);
    osc.connect(gain).connect(ctx.destination);
    osc.start(ctx.currentTime + 0.01);
    osc.stop(ctx.currentTime + duration + 0.01);
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

  // optional vibration pattern (best-effort). Some browsers/OS ignore vibrate; that's fine.
  const doVibratePattern = (pattern = [60, 40, 60, 40]) => {
    if (navigator.vibrate) {
      try {
        navigator.vibrate(pattern);
      } catch (e) {
        // ignore
      }
    }
  };

  // main capture: user gesture must call this handler
  const captureCombined = async () => {
    setStatus("requesting-mic");
    setBandsMatrix(null);
    setSimilarity(null);
    capturedChunksRef.current = [];

    // create/resume AudioContext in user gesture
    if (!audioCtxRef.current) {
      try {
        audioCtxRef.current = new (window.AudioContext || window.webkitAudioContext)();
      } catch (e) {
        console.error("AudioContext failed", e);
        setStatus("audioctx-fail");
        return;
      }
    }
    const ctx = audioCtxRef.current;
    try {
      if (ctx.state === "suspended") await ctx.resume();
    } catch (e) {}

    // request mic
    let stream;
    try {
      stream = await navigator.mediaDevices.getUserMedia({ audio: true });
    } catch (e) {
      console.error("getUserMedia failed", e);
      setStatus("mic-error");
      return;
    }
    streamRef.current = stream;

    // capture via ScriptProcessor for broad browser support
    const source = ctx.createMediaStreamSource(stream);
    const bufferSize = 4096;
    const proc = (ctx.createScriptProcessor || ctx.createJavaScriptNode).call(ctx, bufferSize, 1, 1);
    procRef.current = proc;
    proc.onaudioprocess = (e) => {
      try {
        const inData = e.inputBuffer.getChannelData(0);
        capturedChunksRef.current.push(new Float32Array(inData));
      } catch (err) {
        console.warn("onaudioprocess error", err);
      }
    };

    source.connect(proc);
    // connect to destination lightly so it stays alive on some browsers
    try { proc.connect(ctx.destination); } catch (e) {}

    setRecording(true);
    setStatus("playing-chirp-and-vibrate");

    // start chirp and vibration concurrently
    playChirp(ctx, { duration: 0.3, startFreq: 2000, endFreq: 14000 }).catch((e) => console.warn(e));
    // vibrate pattern timed to overlap chirp and post-chirp
    doVibratePattern([80, 40, 80]);

    // wait capture window
    await new Promise((r) => setTimeout(r, CAPTURE_MS));

    // stop capture
    try { proc.disconnect(); } catch (e) {}
    try { source.disconnect(); } catch (e) {}
    try { stream.getTracks().forEach((t) => t.stop()); } catch (e) {}

    setRecording(false);
    setStatus("processing");

    // assemble raw samples
    let total = 0;
    for (const c of capturedChunksRef.current) total += c.length;
    const all = new Float32Array(total);
    let offset = 0;
    for (const c of capturedChunksRef.current) {
      all.set(c, offset);
      offset += c.length;
    }

    // core processing: STFT-like slicing then band energies
    const sampleRate = ctx.sampleRate;
    const totalSamples = all.length;
    const segLen = Math.max(1, Math.floor(totalSamples / SEGMENTS));

    const bandsMat = [];
    // also compute short-time RMS envelope (for vibration modulation features)
    const rmsWindow = Math.max(32, Math.floor(sampleRate * 0.005)); // 5 ms
    const envelope = new Float32Array(Math.max(1, Math.floor(totalSamples / rmsWindow)));
    for (let i = 0; i < envelope.length; i++) {
      let sum = 0;
      const startE = i * rmsWindow;
      const endE = Math.min(startE + rmsWindow, totalSamples);
      for (let k = startE; k < endE; k++) sum += all[k] * all[k];
      envelope[i] = Math.sqrt(sum / Math.max(1, endE - startE));
    }

    for (let s = 0; s < SEGMENTS; s++) {
      const start = s * segLen;
      const end = Math.min(start + segLen, totalSamples);
      const segment = all.subarray(start, end);

      // window & zero-pad
      const windowed = new Float64Array(FFT_SIZE);
      for (let i = 0; i < FFT_SIZE; i++) {
        const x = i < segment.length ? segment[i] : 0;
        const w = 0.5 * (1 - Math.cos((2 * Math.PI * i) / (FFT_SIZE - 1)));
        windowed[i] = x * w;
      }

      const mag = fftMag(windowed, FFT_SIZE);
      const energies = computeBandEnergies(mag, sampleRate, FFT_SIZE);
      bandsMat.push(energies);
    }

    // normalize per-segment
    const normMat = bandsMat.map((arr) => {
      const sum = arr.reduce((a, b) => a + b, 1e-12);
      return arr.map((v) => v / sum);
    });

    setBandsMatrix(normMat);

    // compute delta ratios between consecutive segments (as before)
    const deltas = [];
    for (let i = 1; i < normMat.length; i++) {
      for (let j = 0; j < normMat[0].length; j++) {
        const prev = normMat[i - 1][j] + 1e-12;
        const cur = normMat[i][j] + 1e-12;
        deltas.push(cur / prev);
      }
    }

    // compute vibration-modulation features from envelope: mean, std, skew, spectral centroid of envelope
    // envelope is short-time RMS; compute simple stats
    let mean = 0;
    for (let i = 0; i < envelope.length; i++) mean += envelope[i];
    mean /= envelope.length || 1;
    let variance = 0;
    for (let i = 0; i < envelope.length; i++) variance += (envelope[i] - mean) ** 2;
    variance /= envelope.length || 1;
    const std = Math.sqrt(variance);

    // small spectral centroid of envelope
    let centroidNum = 0;
    let centroidDen = 1e-12;
    for (let i = 0; i < envelope.length; i++) {
      centroidNum += i * envelope[i];
      centroidDen += envelope[i];
    }
    const centroid = centroidNum / centroidDen;

    // fuse features: deltas + [mean, std, centroid]
    const featureVector = deltas.concat([mean, std, centroid]);

    setStatus("captured");
    return featureVector;
  };

  // wrapper actions
  const handleCaptureOnly = async () => {
    setStatus("capturing");
    await captureCombined();
  };

  const handleSaveBaseline = async () => {
    setStatus("saving-baseline");
    const fv = await captureCombined();
    if (fv) {
      setBaseline(fv);
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
    const fv = await captureCombined();
    if (!fv) {
      setStatus("capture-failed");
      return;
    }
    const sim = cosineSim(baseline, fv);
    setSimilarity(sim);
    setStatus(sim > 0.92 ? "match" : "no-match");
  };

  useEffect(() => {
    return () => {
      try { if (streamRef.current) streamRef.current.getTracks().forEach((t) => t.stop()); } catch (e) {}
      try { if (procRef.current) procRef.current.disconnect(); } catch (e) {}
      try { if (audioCtxRef.current) { audioCtxRef.current.close(); audioCtxRef.current = null; } } catch (e) {}
    };
  }, []);

  return (
    <div style={{ maxWidth: 760, margin: "18px auto", fontFamily: "system-ui, sans-serif" }}>
      <h1 style={{ fontSize: 20, marginBottom: 8 }}>Combined Contact + Vibration Sonar Verifier</h1>

      <div style={{ background: "#fff8e6", padding: 12, borderRadius: 8, marginBottom: 12 }}>
        <strong>Instructions:</strong>
        <ol>
          <li>Place the printed label flat on the phone screen (center) and press gently.</li>
          <li>Tap <em>Capture</em>. Allow microphone permission if prompted.</li>
          <li>The app will play a short chirp and optionally vibrate the phone briefly — hold steady.</li>
          <li>Results: save baseline once per label and use <em>Compare</em> to verify later.</li>
        </ol>
      </div>

      <div style={{ display: "flex", gap: 8, marginTop: 12 }}>
        <button onClick={handleCaptureOnly} disabled={recording} style={{ padding: "10px 14px", background: "#2563eb", color: "white", border: "none", borderRadius: 8 }}>
          {recording ? "Recording..." : "Capture (Contact+Vibrate)"}
        </button>
        <button onClick={handleSaveBaseline} disabled={recording} style={{ padding: "10px 14px", background: "#059669", color: "white", border: "none", borderRadius: 8 }}>
          Capture & Save Baseline
        </button>
        <button onClick={handleCompare} disabled={recording} style={{ padding: "10px 14px", background: "#0ea5e9", color: "white", border: "none", borderRadius: 8 }}>
          Capture & Compare
        </button>
      </div>

      <div style={{ marginTop: 18, padding: 12, background: "#372c2cff", borderRadius: 8 }}>
        <div style={{ fontSize: 13, color: "#333", marginBottom: 8 }}>Status: <b>{status}</b></div>

        <div style={{ display: "flex", gap: 12 }}>
          <div style={{ flex: 1 }}>
            <div style={{ fontSize: 12, color: "#666" }}>Last capture (first 3 segments):</div>
            <pre style={{ fontSize: 12, marginTop: 6 }}>{bandsMatrix ? JSON.stringify(bandsMatrix.slice(0, 3), null, 2) : "-"}</pre>
          </div>

          <div style={{ flex: 1 }}>
            <div style={{ fontSize: 12, color: "#666" }}>Baseline sample (first 10 values):</div>
            <pre style={{ fontSize: 12, marginTop: 6 }}>{baseline ? JSON.stringify(baseline.slice(0, 10), null, 2) : "-"}</pre>
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