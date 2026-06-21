"""
J.A.R.V.I.S — Just A Rather Very Intelligent Sound-identifier
EE200: Signals, Systems & Networks · Course Project

Architecture:
  db_paired.pkl        → hash((f1, f2, dt)) → [(song, anchor_frame), ...]
  db_constellation.pkl → song_name          → [(frame, freq_bin), ...]

Fingerprinting pipeline (matches DB-builder in 3A_final.ipynb exactly):
  audio → mono @ SR → scipy spectrogram (nperseg=1102, noverlap=551) →
  magnitude dB → constellation peaks (20×20 neighborhood, dual threshold) →
  paired hashes (fan=10, max_dt=200) → DB lookup → offset histogram → winner
"""

import streamlit as st
import numpy as np
import pickle
import io
import os
import csv
import time
import subprocess
import collections
import warnings

warnings.filterwarnings("ignore")

from scipy.signal import spectrogram as scipy_spectrogram
from scipy.ndimage import maximum_filter
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import imageio_ffmpeg

# ─── Page Config ─────────────────────────────────────────────────────────────

st.set_page_config(
    page_title="J.A.R.V.I.S · Audio ID",
    page_icon="⚡",
    layout="wide",
    initial_sidebar_state="collapsed",
)

# ─── Global CSS — Iron Man HUD theme ─────────────────────────────────────────

st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Orbitron:wght@500;700;900&family=Rajdhani:wght@400;500;600;700&family=Share+Tech+Mono&family=Exo+2:wght@400;600;700&family=Bebas+Neue&display=swap');

html, body, [class*="css"] {
    font-family: 'Rajdhani', sans-serif;
}

.stApp {
    background:
        radial-gradient(ellipse 90% 40% at 50% 0%, rgba(220,38,38,0.08), transparent),
        radial-gradient(ellipse 50% 30% at 100% 100%, rgba(245,158,11,0.04), transparent),
        radial-gradient(ellipse 50% 30% at 0% 80%, rgba(0,229,255,0.03), transparent),
        linear-gradient(180deg, #08080f 0%, #0a0a14 100%);
}

#MainMenu, footer, header { visibility: hidden; }
.block-container { padding-top: 1.2rem; padding-bottom: 3rem; max-width: 1140px; }

/* ── Hero ── */
.hero {
    text-align: center;
    padding: 2.8rem 1rem 2rem;
    margin-bottom: 2rem;
    position: relative;
}
.hero-divider {
    height: 1px;
    background: linear-gradient(90deg, transparent, #dc2626, #f59e0b, #dc2626, transparent);
    margin-top: 2rem;
    opacity: 0.5;
}
.reactor-wrap {
    display: flex; justify-content: center; margin-bottom: 1.2rem;
}
.reactor-ring {
    width: 72px; height: 72px;
    border-radius: 50%;
    border: 2px solid #dc2626;
    box-shadow: 0 0 22px 3px rgba(220,38,38,0.55),
                inset 0 0 18px rgba(220,38,38,0.4),
                0 0 60px rgba(220,38,38,0.15);
    display: flex; align-items: center; justify-content: center;
    background: radial-gradient(circle, #fca5a5 0%, #dc2626 40%, #1a0505 100%);
    animation: pulse-reactor 3s ease-in-out infinite;
}
@keyframes pulse-reactor {
    0%, 100% { box-shadow: 0 0 22px 3px rgba(220,38,38,0.55), inset 0 0 18px rgba(220,38,38,0.4); }
    50%       { box-shadow: 0 0 34px 6px rgba(220,38,38,0.8),  inset 0 0 26px rgba(220,38,38,0.6); }
}
.reactor-core {
    width: 26px; height: 26px; border-radius: 50%;
    background: radial-gradient(circle, #ffffff 0%, #fef08a 50%, #f59e0b 100%);
    box-shadow: 0 0 16px 6px rgba(254,240,138,0.7);
}

.hero-eyebrow {
    font-family: 'Share Tech Mono', monospace;
    font-size: 0.65rem; letter-spacing: 0.38em;
    color: #f59e0b; text-transform: uppercase;
    margin-bottom: 0.5rem; opacity: 0.85;
}
.hero-title {
    font-family: 'Orbitron', sans-serif;
    font-size: clamp(2.2rem, 5.5vw, 4rem); font-weight: 900;
    color: #fef2f2; line-height: 1.1; margin: 0 0 0.4rem;
    letter-spacing: 0.06em;
    text-shadow: 0 0 32px rgba(220,38,38,0.5), 0 0 80px rgba(220,38,38,0.2);
}
.hero-title .j { color: #ef4444; }
.hero-acronym {
    font-family: 'Share Tech Mono', monospace;
    font-size: 0.72rem; letter-spacing: 0.18em;
    color: #00e5ff; opacity: 0.8; margin-bottom: 0.8rem;
}
.hero-tagline {
    font-family: 'Rajdhani', sans-serif;
    font-size: 1.1rem; font-weight: 500;
    color: #a8a29e; font-style: italic;
}

/* ── Tabs ── */
button[data-baseweb="tab"] {
    font-family: 'Bebas Neue', sans-serif !important;
    font-size: 1rem !important; letter-spacing: 0.18em !important;
    color: #78716c !important;
}
button[data-baseweb="tab"][aria-selected="true"] { color: #f59e0b !important; }
[data-baseweb="tab-highlight"] { background-color: #dc2626 !important; }
[data-baseweb="tab-border"]    { background-color: rgba(220,38,38,0.12) !important; }

/* ── Section header ── */
.sec-hdr {
    display: flex; align-items: center; gap: 0.75rem;
    margin: 2rem 0 1rem;
}
.sec-hdr-line {
    flex: 1; height: 1px;
    background: linear-gradient(90deg, rgba(220,38,38,0.4), transparent);
}
.sec-hdr-text {
    font-family: 'Share Tech Mono', monospace;
    font-size: 0.62rem; letter-spacing: 0.28em;
    color: #00e5ff; text-transform: uppercase; font-weight: 700;
    white-space: nowrap;
}

/* ── Metric cards ── */
.metric-card {
    background: linear-gradient(135deg, #0f0f1a 0%, #0a0a12 100%);
    border: 1px solid rgba(220,38,38,0.22);
    border-radius: 4px; padding: 0.9rem 1.1rem;
    position: relative; overflow: hidden;
}
.metric-card::before {
    content: ''; position: absolute; top: 0; left: 0;
    width: 3px; height: 100%;
    background: linear-gradient(180deg, #f59e0b, #dc2626);
}
.metric-val {
    font-family: 'Orbitron', sans-serif; font-size: 1.6rem;
    font-weight: 700; color: #fef2f2; line-height: 1;
}
.metric-lbl {
    font-family: 'Share Tech Mono', monospace; font-size: 0.6rem;
    color: #b45309; margin-top: 0.35rem;
    text-transform: uppercase; letter-spacing: 0.12em;
}

/* ── Tab page headers (Identify / Batch) ── */
.page-hdr {
    margin-bottom: 1.4rem; padding-bottom: 0.8rem;
    border-bottom: 1px solid rgba(220,38,38,0.15);
}
.page-hdr-title {
    font-family: 'Orbitron', sans-serif; font-size: 1.4rem;
    font-weight: 700; color: #f59e0b; margin-bottom: 0.2rem;
}
.page-hdr-sub {
    font-family: 'Share Tech Mono', monospace; font-size: 0.68rem;
    color: #00e5ff; letter-spacing: 0.12em; opacity: 0.75;
}

/* ── Upload area ── */
[data-testid="stFileUploader"] {
    background: rgba(0,229,255,0.02) !important;
    border: 2px dashed rgba(0,229,255,0.25) !important;
    border-radius: 8px !important;
}
[data-testid="stFileUploader"]:hover {
    border-color: rgba(0,229,255,0.5) !important;
}

/* ── Buttons ── */
.stButton button, .stDownloadButton button {
    font-family: 'Bebas Neue', sans-serif !important;
    font-size: 1.05rem !important; letter-spacing: 0.14em !important;
    background: linear-gradient(135deg, #dc2626, #991b1b) !important;
    border: 1px solid #f59e0b !important; color: #fef2f2 !important;
    box-shadow: 0 0 16px rgba(220,38,38,0.3) !important;
    border-radius: 4px !important; transition: all 0.2s !important;
}
.stButton button:hover, .stDownloadButton button:hover {
    box-shadow: 0 0 28px rgba(220,38,38,0.6) !important;
    border-color: #fbbf24 !important;
}

/* ── Candidate scores ── */
.cand-row {
    display: flex; align-items: center; gap: 0.9rem;
    margin-bottom: 0.65rem; padding: 0.4rem 0;
    border-bottom: 1px solid rgba(255,255,255,0.03);
}
.cand-rank {
    font-family: 'Orbitron', sans-serif; font-size: 0.72rem;
    font-weight: 700; min-width: 24px; color: #57534e;
}
.cand-name {
    font-family: 'Exo 2', sans-serif; font-size: 0.9rem;
    font-weight: 600; color: #e7e5e4; min-width: 200px;
}
.cand-bar-wrap { flex: 1; height: 10px; background: rgba(255,255,255,0.04);
    border-radius: 99px; overflow: hidden; }
.cand-bar { height: 100%; border-radius: 99px; transition: width 0.4s ease; }
.cand-score {
    font-family: 'Share Tech Mono', monospace; font-size: 0.78rem;
    color: #a8a29e; min-width: 52px; text-align: right;
}

/* ── Match / No-match box ── */
.match-box {
    background: linear-gradient(135deg, #0d0a1a 0%, #080812 100%);
    border: 1px solid #dc2626; border-radius: 8px;
    padding: 1.8rem 2.2rem; margin: 1.2rem 0;
    box-shadow: 0 0 32px rgba(220,38,38,0.15), inset 0 0 50px rgba(220,38,38,0.03);
    position: relative;
}
.match-box::before {
    content: ''; position: absolute; top: -1px; left: -1px; right: -1px; height: 3px;
    background: linear-gradient(90deg, #dc2626, #f59e0b, #00e5ff, #f59e0b, #dc2626);
    border-radius: 8px 8px 0 0;
}
.match-jarvis {
    font-family: 'Share Tech Mono', monospace; font-size: 0.7rem;
    letter-spacing: 0.3em; color: #00e5ff; text-transform: uppercase;
    margin-bottom: 0.4rem; opacity: 0.9;
}
.match-phrase {
    font-family: 'Orbitron', sans-serif; font-size: 1.5rem;
    font-weight: 700; color: #f59e0b; margin-bottom: 0.5rem;
    text-shadow: 0 0 20px rgba(245,158,11,0.4);
}
.match-song {
    font-family: 'Orbitron', sans-serif;
    font-size: clamp(1.6rem, 3.5vw, 2.6rem); font-weight: 900;
    color: #fef2f2; line-height: 1.15;
    text-shadow: 0 0 24px rgba(220,38,38,0.4);
}
.match-meta {
    font-family: 'Share Tech Mono', monospace; font-size: 0.78rem;
    color: #fca5a5; margin-top: 0.7rem; opacity: 0.85;
}

.no-match-box {
    background: linear-gradient(135deg, #0d0a0a 0%, #080808 100%);
    border: 1px solid #3f3f46; border-radius: 8px;
    padding: 1.8rem 2.2rem; margin: 1.2rem 0; text-align: center;
}
.no-match-jarvis {
    font-family: 'Share Tech Mono', monospace; font-size: 0.7rem;
    letter-spacing: 0.3em; color: #57534e; text-transform: uppercase;
    margin-bottom: 0.4rem;
}
.no-match-phrase {
    font-family: 'Orbitron', sans-serif; font-size: 1.5rem;
    font-weight: 700; color: #78716c; margin-bottom: 0.5rem;
}
.no-match-sub {
    font-family: 'Rajdhani', sans-serif; font-size: 0.9rem;
    color: #57534e;
}

/* ── Step headers ── */
.step-hdr {
    display: flex; align-items: flex-start; gap: 1rem;
    margin: 2.4rem 0 0.8rem;
}
.step-num {
    width: 34px; height: 34px; border-radius: 4px; flex-shrink: 0;
    background: linear-gradient(135deg, #dc2626, #7f1d1d);
    border: 1px solid rgba(245,158,11,0.6);
    display: flex; align-items: center; justify-content: center;
    font-family: 'Orbitron', sans-serif; font-size: 0.82rem;
    font-weight: 900; color: #fef2f2;
    box-shadow: 0 0 14px rgba(220,38,38,0.35);
}
.step-info-title {
    font-family: 'Bebas Neue', sans-serif; font-size: 1.1rem;
    color: #fde68a; letter-spacing: 0.08em;
}
.step-info-sub {
    font-family: 'Rajdhani', sans-serif; font-size: 0.82rem;
    color: #78716c; margin-top: 0.1rem;
}

/* ── Timing chips ── */
.timing-chip {
    background: rgba(0,229,255,0.04);
    border: 1px solid rgba(0,229,255,0.18);
    border-radius: 4px; padding: 0.45rem 0.8rem;
    font-family: 'Share Tech Mono', monospace;
}
.chip-val { font-size: 1rem; font-weight: 700; color: #fef2f2; }
.chip-lbl { font-size: 0.56rem; color: #0891b2; text-transform: uppercase; letter-spacing: 0.1em; }
.chip-total .chip-val { color: #f59e0b; }
.chip-total { border-color: rgba(245,158,11,0.3); background: rgba(245,158,11,0.04); }

/* ── Hash info line ── */
.hash-info {
    font-family: 'Share Tech Mono', monospace; font-size: 0.75rem;
    color: #00e5ff; margin: 0.5rem 0; opacity: 0.75;
}

/* ── Batch table ── */
.batch-table { width: 100%; border-collapse: collapse; }
.batch-table th {
    font-family: 'Share Tech Mono', monospace;
    font-size: 0.6rem; letter-spacing: 0.18em; text-transform: uppercase;
    color: #0891b2; border-bottom: 1px solid rgba(0,229,255,0.15);
    padding: 0.5rem 0.8rem; text-align: left;
}
.batch-table td {
    padding: 0.6rem 0.8rem; font-size: 0.82rem; color: #e7e5e4;
    border-bottom: 1px solid rgba(255,255,255,0.03);
    font-family: 'Share Tech Mono', monospace;
}
.badge-match {
    background: rgba(245,158,11,0.1); color: #fbbf24;
    padding: 0.2rem 0.5rem; border-radius: 3px; font-size: 0.72rem;
    border: 1px solid rgba(245,158,11,0.2);
}
.badge-none {
    background: rgba(120,113,108,0.1); color: #78716c;
    padding: 0.2rem 0.5rem; border-radius: 3px; font-size: 0.72rem;
}
.conf-bar-wrap { display: flex; align-items: center; gap: 0.5rem; }
.conf-bar-bg { width: 90px; height: 6px; background: rgba(255,255,255,0.05);
    border-radius: 99px; overflow: hidden; }
.conf-bar-fill { height: 100%; border-radius: 99px;
    background: linear-gradient(90deg, #00e5ff, #f59e0b); }

/* ── Expander ── */
[data-testid="stExpander"] {
    background: rgba(0,229,255,0.01) !important;
    border: 1px solid rgba(0,229,255,0.1) !important;
    border-radius: 6px !important;
}

/* ── Song meta in library ── */
.song-meta {
    font-family: 'Share Tech Mono', monospace;
    font-size: 0.65rem; margin-bottom: 0.6rem;
}
</style>
""", unsafe_allow_html=True)


# ─── Constants — must match 3A_final.ipynb DB builder exactly ────────────────

SR               = 9600
NPERSEG          = int(0.05 * SR)          # 480 samples  (~50 ms window)
NOVERLAP         = NPERSEG // 2            # 240
FREQ_CUTOFF      = 4000                    # Hz
NEIGHBORHOOD     = (20, 20)
PEAK_PERCENTILE  = 90
MIN_AMPLITUDE_DB = -60.0
FAN_VALUE        = 20
MAX_DT           = 100
MIN_DT           = 1
CONFIDENCE_THRESHOLD = 10

FFMPEG_EXE = imageio_ffmpeg.get_ffmpeg_exe()

# Palette — cycles per song index (sorted order)
PALETTE = [
    "#00e5ff",   # teal
    "#f59e0b",   # gold
    "#e040fb",   # magenta
    "#76ff03",   # lime
    "#ff6d00",   # orange
    "#448aff",   # blue
    "#ff1744",   # red
    "#d500f9",   # purple
]

DARK_BG  = "#08080f"
PLOT_BG  = "#0d0d18"
AXIS_COL = "#2a2a3a"
TICK_COL = "#4a4a6a"
LBL_COL  = "#6b6b8a"


# ─── Audio I/O ────────────────────────────────────────────────────────────────

def read_audio_bytes(file_bytes: bytes, filename: str) -> np.ndarray:
    """
    Decode any audio format to mono float32 @ SR via bundled ffmpeg.
    Uses a temp file so M4A / container formats work correctly.
    """
    import tempfile
    ext    = os.path.splitext(filename)[1] or ".bin"
    tmp_in = None
    try:
        with tempfile.NamedTemporaryFile(suffix=ext, delete=False) as tmp:
            tmp.write(file_bytes)
            tmp_in = tmp.name
        cmd = [
            FFMPEG_EXE, "-v", "quiet",
            "-i", tmp_in,
            "-f", "f32le", "-ac", "1", "-ar", str(SR),
            "pipe:1",
        ]
        proc = subprocess.run(cmd, capture_output=True)
        if proc.returncode != 0 or len(proc.stdout) == 0:
            raise ValueError(
                f"Could not decode '{filename}'. "
                "Try WAV, MP3, FLAC, OGG, or M4A."
            )
        return np.frombuffer(proc.stdout, dtype=np.float32)
    finally:
        if tmp_in and os.path.exists(tmp_in):
            os.unlink(tmp_in)


# ─── Fingerprinting engine — matches 3A_final.ipynb exactly ──────────────────

def compute_spectrogram(audio: np.ndarray):
    """Return (freqs, times, Sxx, Sxx_dB) — parameters identical to notebook."""
    freqs, times, Sxx = scipy_spectrogram(
        audio, fs=SR,
        nperseg=NPERSEG, noverlap=NOVERLAP,
        window="hann", mode="magnitude",
    )
    mask   = freqs <= FREQ_CUTOFF
    freqs  = freqs[mask]
    Sxx    = Sxx[mask, :]
    Sxx_dB = 20 * np.log10(Sxx + 1e-6)
    return freqs, times, Sxx, Sxx_dB


def find_constellation(Sxx_dB: np.ndarray):
    """
    Dual-threshold peak picking — matches notebook get_constellation exactly:
      1. Local maximum in (20×20) neighborhood
      2. Above absolute dB floor  (MIN_AMPLITUDE_DB = -60)
      3. Above relative percentile floor (PEAK_PERCENTILE = 90)
    Returns [(time_idx, freq_idx), ...] sorted by time_idx.
    """
    local_max  = maximum_filter(Sxx_dB, size=NEIGHBORHOOD) == Sxx_dB
    abs_mask   = Sxx_dB > MIN_AMPLITUDE_DB
    rel_mask   = Sxx_dB > np.percentile(Sxx_dB, PEAK_PERCENTILE)
    peak_mask  = local_max & abs_mask & rel_mask
    freq_idxs, time_idxs = np.where(peak_mask)
    peaks = list(zip(time_idxs.tolist(), freq_idxs.tolist()))
    peaks.sort(key=lambda p: p[0])
    return peaks


def peaks_to_hashes(peaks):
    """Pair each peak with up to FAN_VALUE future peaks within [MIN_DT, MAX_DT]."""
    n = len(peaks)
    for i in range(n):
        t1, f1 = peaks[i]
        for j in range(1, FAN_VALUE + 1):
            if i + j < n:
                t2, f2 = peaks[i + j]
                dt = t2 - t1
                if MIN_DT <= dt <= MAX_DT:
                    yield hash((int(f1), int(f2), int(dt))), t1


def query_paired_db(peaks, db_paired):
    """Vote for (song, offset = db_frame − query_frame) for every matching hash."""
    votes = collections.Counter()
    for h, q_frame in peaks_to_hashes(peaks):
        if h in db_paired:
            for song, db_frame in db_paired[h]:
                votes[(song, db_frame - q_frame)] += 1
    return votes


def cluster_votes(votes: collections.Counter):
    """Per-song cluster score = height of tallest offset bin."""
    per_song = collections.defaultdict(collections.Counter)
    for (song, offset), cnt in votes.items():
        per_song[song][offset] += cnt
    results = [
        (song, off_counts.most_common(1)[0][1])
        for song, off_counts in per_song.items()
    ]
    return sorted(results, key=lambda x: -x[1])


def identify(audio: np.ndarray, db_paired):
    """Full pipeline: audio → result dict."""
    t0 = time.time()
    freqs, times, Sxx, Sxx_dB = compute_spectrogram(audio)
    t_spec = time.time()

    peaks  = find_constellation(Sxx_dB)
    t_const = time.time()

    votes  = query_paired_db(peaks, db_paired)
    ranked = cluster_votes(votes)
    t_lookup = time.time()

    offset_hist    = None
    matched_offset = None
    if ranked:
        top_song = ranked[0][0]
        per_song = collections.defaultdict(int)
        for (song, offset), cnt in votes.items():
            if song == top_song:
                per_song[offset] += cnt
        offset_hist = dict(per_song)
        if per_song:
            matched_offset = max(per_song, key=per_song.get)

    t_end = time.time()

    scores_out = []
    if ranked:
        top_score = ranked[0][1]
        for song, score in ranked[:10]:
            pct = (score / top_score * 100) if top_score > 0 else 0
            scores_out.append((song, score, pct))

    winner = ranked[0][0] if ranked and ranked[0][1] >= CONFIDENCE_THRESHOLD else None

    return {
        "winner":         winner,
        "scores":         scores_out,
        "peaks":          peaks,
        "freqs":          freqs,
        "times":          times,
        "Sxx_dB":         Sxx_dB,
        "offset_hist":    offset_hist,
        "matched_offset": matched_offset,
        "query_n_frames": Sxx.shape[1],
        "n_hashes":       sum(1 for _ in peaks_to_hashes(peaks)),
        "timing": {
            "spectrogram_ms":   int((t_spec   - t0)      * 1000),
            "constellation_ms": int((t_const  - t_spec)  * 1000),
            "lookup_ms":        int((t_lookup  - t_const) * 1000),
            "total_ms":         int((t_end    - t0)      * 1000),
        },
    }


# ─── DB loading ───────────────────────────────────────────────────────────────

@st.cache_resource(show_spinner=False)
def load_databases():
    """Load db_paired and db_constellation from same directory as app.py."""
    base         = os.path.dirname(os.path.abspath(__file__))
    paired_path  = os.path.join(base, "db_paired.pkl")
    constel_path = os.path.join(base, "db_constellation.pkl")

    if not os.path.exists(paired_path):
        return None, None

    with open(paired_path, "rb") as f:
        db_paired = pickle.load(f)

    db_constellation = {}
    if os.path.exists(constel_path):
        with open(constel_path, "rb") as f:
            db_constellation = pickle.load(f)

    return db_paired, db_constellation


@st.cache_data(show_spinner=False)
def get_song_hash_counts(_db_paired):
    counts = collections.Counter()
    for entries in _db_paired.values():
        for song, _ in entries:
            counts[song] += 1
    return counts


def song_color(song_name: str, sorted_names: list) -> str:
    """Return a consistent palette color for a song based on its sorted index."""
    try:
        idx = sorted_names.index(song_name)
    except ValueError:
        idx = 0
    return PALETTE[idx % len(PALETTE)]


# ─── Plot helpers ─────────────────────────────────────────────────────────────

def _style(fig, axes):
    fig.patch.set_facecolor(DARK_BG)
    for ax in axes:
        ax.set_facecolor(PLOT_BG)
        ax.tick_params(colors=TICK_COL, labelsize=8)
        for spine in ax.spines.values():
            spine.set_color(AXIS_COL)
        ax.xaxis.label.set_color(LBL_COL)
        ax.yaxis.label.set_color(LBL_COL)
        ax.title.set_color("#fbbf24")
        ax.title.set_fontsize(9)


def plot_query_analysis(freqs, times, Sxx_dB, peaks):
    """Side-by-side: spectrogram (inferno) + constellation scatter (teal dots)."""
    hop_s = (NPERSEG - NOVERLAP) / SR
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(11, 3.6), facecolor=DARK_BG)
    fig.subplots_adjust(wspace=0.3)

    ax1.pcolormesh(times, freqs / 1000, Sxx_dB,
                   cmap="inferno", vmin=-80, vmax=0, shading="auto")
    ax1.set_xlabel("time (s)")
    ax1.set_ylabel("freq (kHz)")
    ax1.set_title("Spectrogram")

    if peaks:
        pt = [p[0] * hop_s for p in peaks]
        pf = [freqs[p[1]] / 1000 if p[1] < len(freqs) else 0 for p in peaks]
        ax2.scatter(pt, pf, s=8, color="#00e5ff", alpha=0.8, linewidths=0)
    ax2.set_xlabel("time (s)")
    ax2.set_ylabel("freq (kHz)")
    ax2.set_title(f"Constellation  ({len(peaks)} peaks)")

    _style(fig, [ax1, ax2])
    return fig


def plot_library_constellation(peaks_list, title, color):
    """
    Real constellation scatter from db_constellation for library cards.
    peaks_list: [(frame, freq_bin), ...]
    """
    if not peaks_list:
        return None
    hop_s   = (NPERSEG - NOVERLAP) / SR
    frames  = np.array([p[0] for p in peaks_list], dtype=float)
    fbins   = np.array([p[1] for p in peaks_list], dtype=float)
    times_s = frames * hop_s

    fig, ax = plt.subplots(figsize=(5, 2.4), facecolor=DARK_BG)
    ax.scatter(times_s, fbins, s=2.5, color=color, alpha=0.7, linewidths=0)
    ax.set_xlabel("time (s)", fontsize=7)
    ax.set_ylabel("freq bin", fontsize=7)
    ax.set_title(title, fontsize=8)
    _style(fig, [ax])
    return fig


def plot_song_window(peaks_list, song_title, color,
                     highlight_start_s=None, highlight_end_s=None):
    """
    Full song constellation from db_constellation with query window highlighted.
    """
    if not peaks_list:
        return None
    hop_s   = (NPERSEG - NOVERLAP) / SR
    frames  = np.array([p[0] for p in peaks_list], dtype=float)
    fbins   = np.array([p[1] for p in peaks_list], dtype=float)
    times_s = frames * hop_s

    fig, ax = plt.subplots(figsize=(11, 3.2), facecolor=DARK_BG)

    if highlight_start_s is not None and highlight_end_s is not None:
        inside  = (times_s >= highlight_start_s) & (times_s <= highlight_end_s)
        outside = ~inside
        ax.scatter(times_s[outside], fbins[outside],
                   s=2, color=color, alpha=0.35, linewidths=0)
        ax.scatter(times_s[inside], fbins[inside],
                   s=5, color="#f59e0b", alpha=0.95, linewidths=0,
                   zorder=5, label="query clip")
        ax.axvspan(highlight_start_s, highlight_end_s,
                   color="#f59e0b", alpha=0.07)
        ax.axvline(highlight_start_s, color="#f59e0b", lw=1, alpha=0.6)
        ax.axvline(highlight_end_s,   color="#f59e0b", lw=1, alpha=0.6)
        ax.legend(fontsize=7, framealpha=0.1, labelcolor="#a8a29e",
                  loc="upper right")
    else:
        ax.scatter(times_s, fbins, s=2, color=color, alpha=0.55, linewidths=0)

    ax.set_xlabel("time (s)")
    ax.set_ylabel("freq bin")
    ax.set_title(f"{song_title}  ·  {len(peaks_list):,} constellation peaks")
    _style(fig, [ax])
    return fig


def plot_offset_histogram(offset_hist: dict, winner: str):
    """
    Offset histogram: teal bars for noise floor, gold bar for the alignment spike.
    X-axis in seconds (bucketed).
    """
    if not offset_hist:
        return None

    hop_s      = (NPERSEG - NOVERLAP) / SR
    sec_counts = collections.defaultdict(int)
    for offset_frame, cnt in offset_hist.items():
        sec = round(offset_frame * hop_s)
        sec_counts[sec] += cnt

    secs   = sorted(sec_counts.keys())
    counts = [sec_counts[s] for s in secs]
    if not counts:
        return None

    peak_idx = int(np.argmax(counts))
    peak_sec = secs[peak_idx]
    peak_val = counts[peak_idx]

    colors = ["#00e5ff" if i != peak_idx else "#f59e0b"
              for i in range(len(secs))]

    fig, ax = plt.subplots(figsize=(11, 3.2), facecolor=DARK_BG)
    ax.bar(secs, counts,
           width=max(0.6, len(secs) * 0.008),
           color=colors, alpha=0.85, edgecolor="none")

    noise = np.percentile(counts, 70) if len(counts) > 4 else 1
    ax.axhline(noise, color="#57534e", lw=0.9, ls="--", alpha=0.6,
               label="background noise")

    offset_x = peak_sec + max(2, (max(secs) - min(secs)) * 0.08) if secs else peak_sec
    ax.annotate(
        f"{peak_val:,} hashes\nalign here",
        xy=(peak_sec, peak_val),
        xytext=(offset_x, peak_val * 0.82),
        color="#fde68a", fontsize=8,
        arrowprops=dict(arrowstyle="->", color="#f59e0b", lw=1.2),
    )
    ax.legend(fontsize=7, framealpha=0.08, labelcolor="#a8a29e")
    ax.set_xlabel("time offset (seconds)  —  database time − query time")
    ax.set_ylabel("# hashes")
    ax.set_title(f"Alignment spike  ·  {winner.replace('_', ' ')}")
    _style(fig, [ax])
    return fig


# ─── UI helpers ───────────────────────────────────────────────────────────────

def sec_hdr(label: str):
    st.markdown(f"""
    <div class="sec-hdr">
        <div class="sec-hdr-text">{label}</div>
        <div class="sec-hdr-line"></div>
    </div>""", unsafe_allow_html=True)


def step_hdr(num: str, title: str, subtitle: str):
    st.markdown(f"""
    <div class="step-hdr">
        <div class="step-num">{num}</div>
        <div>
            <div class="step-info-title">{title}</div>
            <div class="step-info-sub">{subtitle}</div>
        </div>
    </div>""", unsafe_allow_html=True)


def render_candidates(scores, winner):
    if not scores:
        return
    sec_hdr("⟁ Candidate scores")
    top_score = scores[0][1] if scores else 1
    medal     = ["#f59e0b", "#94a3b8", "#b45309"]
    for i, (song, score, _) in enumerate(scores[:5]):
        is_win    = song == winner
        bar_color = "#f59e0b" if is_win else "#00e5ff"
        bar_w     = max(2, int(score / max(top_score, 1) * 100))
        rk_color  = medal[i] if i < 3 else "#57534e"
        st.markdown(f"""
        <div class="cand-row">
            <div class="cand-rank" style="color:{rk_color};">#{i+1}</div>
            <div class="cand-name">{song.replace('_', ' ')}</div>
            <div class="cand-bar-wrap">
                <div class="cand-bar" style="width:{bar_w}%;background:{bar_color};"></div>
            </div>
            <div class="cand-score">{score:,}</div>
        </div>""", unsafe_allow_html=True)


def render_match_box(winner, scores):
    if winner:
        top_score = scores[0][1]
        runner_up = scores[1][1] if len(scores) > 1 else 1
        ratio     = top_score / max(runner_up, 1)
        st.markdown(f"""
        <div class="match-box">
            <div class="match-jarvis">▲ J.A.R.V.I.S  RESPONSE</div>
            <div class="match-phrase">Found it, Boss.</div>
            <div class="match-song">{winner.replace('_', ' ')}</div>
            <div class="match-meta">
                alignment score {top_score:,} &nbsp;·&nbsp; {ratio:.0f}× the runner-up
            </div>
        </div>""", unsafe_allow_html=True)
    else:
        st.markdown("""
        <div class="no-match-box">
            <div class="no-match-jarvis">▲ J.A.R.V.I.S  RESPONSE</div>
            <div class="no-match-phrase">Even I have limits, Boss.</div>
            <div class="no-match-sub">Try a longer clip or check the library.</div>
        </div>""", unsafe_allow_html=True)


def render_timing(timing: dict):
    chips = [
        ("Spectrogram",   timing["spectrogram_ms"],   False),
        ("Constellation", timing["constellation_ms"],  False),
        ("DB Lookup",     timing["lookup_ms"],         False),
        ("Total",         timing["total_ms"],          True),
    ]
    cols = st.columns(len(chips))
    for col, (label, ms, is_total) in zip(cols, chips):
        extra = " chip-total" if is_total else ""
        with col:
            st.markdown(f"""
            <div class="timing-chip{extra}">
                <div class="chip-val">{ms} ms</div>
                <div class="chip-lbl">{label}</div>
            </div>""", unsafe_allow_html=True)


# ─── Tabs ─────────────────────────────────────────────────────────────────────

def tab_library(db_paired, db_constellation):
    # ── Hero (only on library tab) ──────────────────────────────────────────
    st.markdown("""
    <div class="hero">
        <div class="reactor-wrap">
            <div class="reactor-ring"><div class="reactor-core"></div></div>
        </div>
        <div class="hero-eyebrow">EE200 · Signals, Systems &amp; Networks · Course Project</div>
        <div class="hero-title"><span class="j">J</span>.A.R.V.I.S</div>
        <div class="hero-acronym">Just A Rather Very Intelligent Sound-identifier</div>
        <div class="hero-tagline">"JARVIS, identify that track."</div>
        <div class="hero-divider"></div>
    </div>""", unsafe_allow_html=True)

    counts       = get_song_hash_counts(db_paired)
    songs        = sorted(counts.items(), key=lambda x: x[0])
    sorted_names = [s[0] for s in songs]

    # ── System status ────────────────────────────────────────────────────────
    sec_hdr("⟁ System status")
    c1, c2, c3 = st.columns(3)
    with c1:
        st.markdown(
            f'<div class="metric-card"><div class="metric-val">{len(songs)}</div>'
            f'<div class="metric-lbl">Songs Indexed</div></div>',
            unsafe_allow_html=True,
        )
    with c2:
        st.markdown(
            f'<div class="metric-card"><div class="metric-val">{len(db_paired):,}</div>'
            f'<div class="metric-lbl">Unique Hash Keys</div></div>',
            unsafe_allow_html=True,
        )
    with c3:
        st.markdown(
            f'<div class="metric-card"><div class="metric-val">{SR//1000}k Hz</div>'
            f'<div class="metric-lbl">Sample Rate</div></div>',
            unsafe_allow_html=True,
        )

    # ── Indexed library ──────────────────────────────────────────────────────
    sec_hdr("⟁ Indexed library — expand a track to inspect its constellation")

    cols = st.columns(4)
    for i, (name, n_hash) in enumerate(songs):
        color = PALETTE[i % len(PALETTE)]
        with cols[i % 4]:
            with st.expander(f"⚡ {name.replace('_', ' ')}", expanded=False):
                st.markdown(
                    f'<div class="song-meta" style="color:{color};">'
                    f'{n_hash:,} hashes</div>',
                    unsafe_allow_html=True,
                )
                peaks_list = db_constellation.get(name, [])
                if peaks_list:
                    hop_s = (NPERSEG - NOVERLAP) / SR
                    dur_s = max(p[0] for p in peaks_list) * hop_s
                    st.markdown(
                        f'<div class="song-meta" style="color:{LBL_COL};">'
                        f'~{dur_s:.0f}s &nbsp;·&nbsp; {len(peaks_list):,} peaks</div>',
                        unsafe_allow_html=True,
                    )
                    fig = plot_library_constellation(peaks_list, "Constellation", color)
                    if fig:
                        st.pyplot(fig, use_container_width=True)
                        plt.close(fig)
                else:
                    st.markdown(
                        '<div class="song-meta" style="color:#57534e;">'
                        'Constellation data not available</div>',
                        unsafe_allow_html=True,
                    )


def tab_identify(db_paired, db_constellation):
    counts       = get_song_hash_counts(db_paired)
    sorted_names = sorted(counts.keys())

    st.markdown("""
    <div class="page-hdr">
        <div class="page-hdr-title">⚡ Identify a Track</div>
        <div class="page-hdr-sub">UPLOAD A CLIP · JARVIS WILL FIND IT</div>
    </div>""", unsafe_allow_html=True)

    uploaded = st.file_uploader(
        "Drop a clip here",
        type=["wav", "mp3", "flac", "ogg", "m4a"],
        label_visibility="collapsed",
        key="identify_uploader",
    )

    # ── Audio player — shown as soon as a file is dropped ───────────────────
    if uploaded:
        sec_hdr("⟁ Clip preview")
        st.audio(uploaded, format=f"audio/{uploaded.name.rsplit('.',1)[-1].lower()}")
        uploaded.seek(0)   # reset so read() below gets the full bytes

    run = st.button("⚡ IDENTIFY", type="primary", use_container_width=True)

    if not (uploaded and run):
        return

    with st.spinner("Scanning frequency signature…"):
        try:
            raw    = uploaded.read()
            audio  = read_audio_bytes(raw, uploaded.name)
            result = identify(audio, db_paired)
        except Exception as e:
            st.error(f"⚠ {e}")
            return

    winner = result["winner"]
    scores = result["scores"]

    # ── 1. Query Analysis ────────────────────────────────────────────────────
    step_hdr("1", "Spectrogram → Constellation",
             "Time-frequency map, then top peaks kept as fingerprint")
    fig1 = plot_query_analysis(
        result["freqs"], result["times"], result["Sxx_dB"], result["peaks"]
    )
    st.pyplot(fig1, use_container_width=True)
    plt.close(fig1)
    st.markdown(
        f'<div class="hash-info">→ {result["n_hashes"]:,} paired hashes '
        f'generated from {len(result["peaks"])} constellation peaks</div>',
        unsafe_allow_html=True,
    )

    # ── 2. Candidate Scores (above verdict) ──────────────────────────────────
    render_candidates(scores, winner)

    # ── 3. J.A.R.V.I.S verdict ───────────────────────────────────────────────
    sec_hdr("⟁ J.A.R.V.I.S verdict")
    render_match_box(winner, scores)

    if not winner:
        sec_hdr("⟁ Processing time")
        render_timing(result["timing"])
        return

    # ── 4. Where in the song ─────────────────────────────────────────────────
    hop_s      = (NPERSEG - NOVERLAP) / SR
    w_color    = song_color(winner, sorted_names)
    peaks_list = db_constellation.get(winner, [])

    step_hdr(
        "2", "Where in the Song?",
        f"Full constellation of '{winner.replace('_', ' ')}' — "
        "highlighted band shows where your clip sits",
    )

    if peaks_list and result["matched_offset"] is not None:
        win_start = result["matched_offset"] * hop_s
        win_end   = win_start + result["query_n_frames"] * hop_s
        fig2 = plot_song_window(
            peaks_list, winner.replace("_", " "), w_color,
            highlight_start_s=win_start, highlight_end_s=win_end,
        )
        if fig2:
            st.pyplot(fig2, use_container_width=True)
            plt.close(fig2)
    elif not peaks_list:
        st.markdown(
            '<div class="song-meta" style="color:#57534e;">'
            'Constellation data not available — place db_constellation.pkl next to app.py</div>',
            unsafe_allow_html=True,
        )

    # ── 5. Alignment Spike ───────────────────────────────────────────────────
    if result["offset_hist"]:
        step_hdr(
            "3", "Alignment Spike",
            "Every matched hash votes for a time offset — "
            "a genuine match converges on a single point",
        )
        fig3 = plot_offset_histogram(result["offset_hist"], winner)
        if fig3:
            st.pyplot(fig3, use_container_width=True)
            plt.close(fig3)

    # ── 6. Timing (bottom) ───────────────────────────────────────────────────
    sec_hdr("⟁ Processing time")
    render_timing(result["timing"])


def tab_batch(db_paired):
    st.markdown("""
    <div class="page-hdr">
        <div class="page-hdr-title">⚡ Batch Identification</div>
        <div class="page-hdr-sub">UPLOAD MULTIPLE CLIPS · GET results.csv</div>
    </div>""", unsafe_allow_html=True)

    st.markdown(
        '<p style="font-family:\'Rajdhani\',sans-serif;color:#78716c;'
        'font-size:0.9rem;margin-bottom:1.2rem;">'
        "Upload several clips. Each is matched independently against the indexed library. "
        "Results are exported as <code>results.csv</code> with columns "
        "<code>filename, prediction</code>.</p>",
        unsafe_allow_html=True,
    )

    files = st.file_uploader(
        "Drop clips here",
        type=["wav", "mp3", "flac", "ogg", "m4a"],
        accept_multiple_files=True,
        label_visibility="collapsed",
        key="batch_uploader",
    )
    run = st.button("▶ RUN BATCH", type="primary")

    if not (files and run):
        return

    rows = []
    prog = st.progress(0.0, text="Processing…")
    for i, f in enumerate(files):
        prog.progress((i + 1) / len(files), text=f"Identifying {f.name}…")
        try:
            raw   = f.read()
            audio = read_audio_bytes(raw, f.name)
            res   = identify(audio, db_paired)
            pred  = res["winner"] if res["winner"] else "none"
            score = res["scores"][0][1] if res["scores"] else 0
        except Exception:
            pred, score = "none", 0
        rows.append({"filename": f.name, "prediction": pred, "score": score})
    prog.empty()

    sec_hdr("⟁ Results")
    matched = sum(1 for r in rows if r["prediction"] != "none")
    st.markdown(
        f'<p style="font-family:\'Share Tech Mono\',monospace;font-size:0.78rem;'
        f'color:#00e5ff;margin-bottom:1rem;">'
        f'{matched}/{len(rows)} clips matched · '
        f'{len(rows)-matched} returned none</p>',
        unsafe_allow_html=True,
    )

    top_score_all = max((r["score"] for r in rows), default=1)
    html  = '<table class="batch-table"><thead><tr>'
    html += '<th>File</th><th>Prediction</th><th>Score</th></tr></thead><tbody>'
    for r in rows:
        badge = (
            f'<span class="badge-match">{r["prediction"].replace("_"," ")}</span>'
            if r["prediction"] != "none"
            else '<span class="badge-none">none</span>'
        )
        pct = int(r["score"] / max(top_score_all, 1) * 100)
        bar = (
            f'<div class="conf-bar-wrap">'
            f'<div class="conf-bar-bg">'
            f'<div class="conf-bar-fill" style="width:{pct}%;"></div></div>'
            f'<span style="font-size:0.7rem;color:#78716c;">{r["score"]:,}</span></div>'
        )
        html += f"<tr><td>{r['filename']}</td><td>{badge}</td><td>{bar}</td></tr>"
    html += "</tbody></table>"
    st.markdown(html, unsafe_allow_html=True)

    csv_buf = io.StringIO()
    writer  = csv.DictWriter(csv_buf, fieldnames=["filename", "prediction"])
    writer.writeheader()
    for r in rows:
        writer.writerow({"filename": r["filename"], "prediction": r["prediction"]})

    st.markdown("<br>", unsafe_allow_html=True)
    st.download_button(
        "⬇ DOWNLOAD results.csv",
        csv_buf.getvalue(),
        file_name="results.csv",
        mime="text/csv",
    )


# ─── Main ─────────────────────────────────────────────────────────────────────

def main():
    with st.spinner("Booting J.A.R.V.I.S…"):
        db_paired, db_constellation = load_databases()

    if db_paired is None:
        st.error(
            "⚠ Database not found. "
            "Place `db_paired.pkl` and `db_constellation.pkl` "
            "in the same directory as `app.py`."
        )
        return

    tab1, tab2, tab3 = st.tabs(["📡  LIBRARY", "🎯  IDENTIFY", "⚡  BATCH"])

    with tab1:
        tab_library(db_paired, db_constellation)
    with tab2:
        tab_identify(db_paired, db_constellation)
    with tab3:
        tab_batch(db_paired)


if __name__ == "__main__":
    main()
