"""
J.A.R.V.I.S — Just A Rather Very Intelligent Sound-identifier

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

/* ── Scanline overlay (subtle, static grid only) ── */
.stApp::before {
    content: '';
    position: fixed; top: 0; left: 0; width: 100%; height: 100%;
    background: linear-gradient(transparent 50%, rgba(0,0,0,0.025) 50%);
    background-size: 100% 4px;
    pointer-events: none; z-index: 9998;
}

.stApp {
    background:
        radial-gradient(ellipse 90% 40% at 50% 0%, rgba(220,38,38,0.08), transparent),
        radial-gradient(ellipse 50% 30% at 100% 100%, rgba(245,158,11,0.04), transparent),
        radial-gradient(ellipse 50% 30% at 0% 80%, rgba(0,229,255,0.03), transparent),
        linear-gradient(180deg, #08080f 0%, #0a0a14 100%);
}

/* ── Hex particle canvas ── */
#hex-canvas {
    position: fixed; top: 0; left: 0;
    width: 100%; height: 100%;
    pointer-events: none; z-index: 0; opacity: 0.18;
}

#MainMenu, footer, header { visibility: hidden; }
.block-container { padding-top: 1.2rem; padding-bottom: 3rem; max-width: 1140px; position: relative; z-index: 1; }

/* ── Viewport corner brackets ── */
.vp-corner {
    position: fixed; width: 28px; height: 28px;
    border-color: rgba(0,229,255,0.35); border-style: solid;
    pointer-events: none; z-index: 9997;
    animation: corner-pulse 4s ease-in-out infinite;
}
.vp-corner.tl { top: 12px; left: 12px; border-width: 2px 0 0 2px; }
.vp-corner.tr { top: 12px; right: 12px; border-width: 2px 2px 0 0; }
.vp-corner.bl { bottom: 28px; left: 12px; border-width: 0 0 2px 2px; }
.vp-corner.br { bottom: 28px; right: 12px; border-width: 0 2px 2px 0; }
@keyframes corner-pulse {
    0%, 100% { opacity: 0.35; }
    50%       { opacity: 0.7; }
}

/* ── Diagonal HUD grid overlay ── */
.hud-grid {
    position: fixed; top: 0; left: 0; width: 100%; height: 100%;
    pointer-events: none; z-index: 0; opacity: 0.03;
    background-image:
        repeating-linear-gradient(45deg, rgba(0,229,255,1) 0px, rgba(0,229,255,1) 1px, transparent 1px, transparent 60px),
        repeating-linear-gradient(-45deg, rgba(220,38,38,0.6) 0px, rgba(220,38,38,0.6) 1px, transparent 1px, transparent 60px);
}

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

/* ── Arc Reactor ── */
.reactor-wrap {
    display: flex; justify-content: center; margin-bottom: 1.4rem;
}
.reactor-outer {
    position: relative; width: 140px; height: 140px;
    display: flex; align-items: center; justify-content: center;
}

/* power ring */
.reactor-power-ring {
    position: absolute; width: 140px; height: 140px; border-radius: 50%;
    border: 2px solid rgba(220,38,38,0.25);
    box-shadow: 0 0 8px rgba(220,38,38,0.15);
}
.reactor-power-ring::before {
    content: '';
    position: absolute; top: -2px; left: -2px; right: -2px; bottom: -2px;
    border-radius: 50%;
    border: 2px solid transparent;
    border-top-color: #dc2626;
    border-right-color: #f59e0b;
    animation: spin-power 3s linear infinite;
}
@keyframes spin-power { to { transform: rotate(360deg); } }

/* radar sweep */
.reactor-radar {
    position: absolute; width: 110px; height: 110px; border-radius: 50%;
    border: 1px solid rgba(0,229,255,0.15);
    overflow: hidden;
}
.reactor-radar::after {
    content: '';
    position: absolute; top: 50%; left: 50%;
    width: 55px; height: 1px;
    background: linear-gradient(90deg, rgba(0,229,255,0.8), transparent);
    transform-origin: left center;
    animation: radar-sweep 2.5s linear infinite;
}
@keyframes radar-sweep { to { transform: rotate(360deg); } }

/* segment ring */
.reactor-segments {
    position: absolute; width: 90px; height: 90px; border-radius: 50%;
    animation: spin-segments 8s linear infinite reverse;
}
.reactor-segments svg {
    width: 100%; height: 100%;
}

/* inner ring */
.reactor-inner-ring {
    position: absolute; width: 68px; height: 68px; border-radius: 50%;
    border: 1.5px solid rgba(0,229,255,0.5);
    box-shadow: 0 0 12px rgba(0,229,255,0.3), inset 0 0 12px rgba(0,229,255,0.1);
    animation: pulse-inner 2s ease-in-out infinite;
}
@keyframes pulse-inner {
    0%, 100% { box-shadow: 0 0 12px rgba(0,229,255,0.3), inset 0 0 12px rgba(0,229,255,0.1); }
    50%       { box-shadow: 0 0 22px rgba(0,229,255,0.6), inset 0 0 20px rgba(0,229,255,0.25); }
}

/* core */
.reactor-core {
    position: absolute; width: 38px; height: 38px; border-radius: 50%;
    background: radial-gradient(circle, #ffffff 0%, #a5f3fc 35%, #00e5ff 65%, #0e7490 100%);
    box-shadow: 0 0 20px 6px rgba(0,229,255,0.7), 0 0 50px rgba(0,229,255,0.3);
    animation: pulse-core 2s ease-in-out infinite, surge 5s ease-in-out infinite;
    z-index: 2;
}
@keyframes pulse-core {
    0%, 100% { box-shadow: 0 0 20px 6px rgba(0,229,255,0.7), 0 0 50px rgba(0,229,255,0.3); }
    50%       { box-shadow: 0 0 30px 10px rgba(0,229,255,0.9), 0 0 80px rgba(0,229,255,0.5); }
}
@keyframes surge {
    0%, 80%, 100% { filter: brightness(1); }
    85%            { filter: brightness(2.5); }
    90%            { filter: brightness(1); }
    93%            { filter: brightness(2); }
}
@keyframes spin-segments { to { transform: rotate(360deg); } }

/* arc sparks */
.arc-spark {
    position: absolute; width: 2px; border-radius: 99px;
    background: linear-gradient(180deg, rgba(0,229,255,0.9), transparent);
    transform-origin: bottom center;
    animation: spark-flash 5s ease-in-out infinite;
    opacity: 0;
}
@keyframes spark-flash {
    0%, 70%, 100% { opacity: 0; height: 0px; }
    75%            { opacity: 1; height: 18px; }
    80%            { opacity: 0; height: 0px; }
    84%            { opacity: 0.7; height: 12px; }
    88%            { opacity: 0; height: 0px; }
}

/* readout */
.reactor-readout {
    font-family: 'Share Tech Mono', monospace;
    font-size: 0.55rem; letter-spacing: 0.2em;
    color: rgba(0,229,255,0.45); margin-top: 0.6rem;
    text-align: center; text-transform: uppercase;
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
    cursor: default;
    transition: text-shadow 0.2s;
}
.hero-title:hover {
    text-shadow: 0 0 8px rgba(220,38,38,0.9), 0 0 40px rgba(220,38,38,0.6),
                 2px 0 0 rgba(0,229,255,0.4), -2px 0 0 rgba(220,38,38,0.4);
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

/* ── Section header with corner brackets ── */
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
.sec-hdr-bracket {
    font-family: 'Share Tech Mono', monospace;
    font-size: 0.9rem; color: rgba(220,38,38,0.5);
    line-height: 1;
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

/* ── Upload area — HUD reticle ── */
[data-testid="stFileUploader"] {
    background: rgba(0,229,255,0.02) !important;
    border: none !important;
    border-radius: 0px !important;
    position: relative;
}
.hud-uploader-wrap {
    position: relative; padding: 1.2rem;
    background: rgba(0,229,255,0.015);
}
.hud-uploader-wrap::before,
.hud-uploader-wrap::after {
    content: '';
    position: absolute; width: 18px; height: 18px;
    border-color: #00e5ff; border-style: solid; opacity: 0.6;
}
.hud-uploader-wrap::before {
    top: 0; left: 0;
    border-width: 2px 0 0 2px;
}
.hud-uploader-wrap::after {
    bottom: 0; right: 0;
    border-width: 0 2px 2px 0;
}
.hud-corner-tr, .hud-corner-bl {
    position: absolute; width: 18px; height: 18px;
    border-color: #00e5ff; border-style: solid; opacity: 0.6;
    pointer-events: none;
}
.hud-corner-tr { top: 0; right: 0; border-width: 2px 2px 0 0; }
.hud-corner-bl { bottom: 0; left: 0; border-width: 0 0 2px 2px; }
.awaiting-input {
    font-family: 'Share Tech Mono', monospace;
    font-size: 0.65rem; letter-spacing: 0.3em;
    color: #00e5ff; text-align: center;
    margin-top: 0.5rem; text-transform: uppercase;
    animation: blink-await 1.4s step-end infinite;
}
@keyframes blink-await { 50% { opacity: 0; } }

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

/* ── Song cards — fixed height HUD panels ── */
.song-card {
    background: linear-gradient(135deg, #0d0d1a 0%, #0a0a12 100%);
    border: 1px solid rgba(220,38,38,0.18);
    border-left: 3px solid;
    border-radius: 4px;
    padding: 0.55rem 0.7rem 0.55rem 0.8rem;
    height: 54px;
    display: flex; align-items: center; gap: 0.5rem;
    cursor: pointer; margin-bottom: 0.5rem;
    transition: border-color 0.2s, box-shadow 0.2s;
    overflow: hidden; position: relative;
}
.song-card:hover {
    box-shadow: 0 0 14px rgba(220,38,38,0.25);
    border-color: rgba(245,158,11,0.5);
}
.song-card-idx {
    font-family: 'Share Tech Mono', monospace;
    font-size: 0.52rem; color: #57534e;
    min-width: 26px; letter-spacing: 0.05em;
}
.song-card-name {
    font-family: 'Exo 2', sans-serif; font-size: 0.82rem;
    font-weight: 600; color: #e7e5e4; flex: 1;
    white-space: nowrap; overflow: hidden; text-overflow: ellipsis;
}
.song-card-dot {
    width: 6px; height: 6px; border-radius: 50%; flex-shrink: 0;
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

/* ── Confidence power meter ── */
.power-meter-wrap {
    margin-top: 1rem;
}
.power-meter-label {
    font-family: 'Share Tech Mono', monospace; font-size: 0.6rem;
    letter-spacing: 0.22em; color: #00e5ff; text-transform: uppercase;
    margin-bottom: 0.4rem;
}
.power-meter-segments {
    display: flex; gap: 3px;
}
.power-seg {
    height: 14px; flex: 1; border-radius: 2px;
    background: rgba(255,255,255,0.06);
    transition: background 0.1s;
}
.power-seg.active-low    { background: #dc2626; box-shadow: 0 0 6px rgba(220,38,38,0.6); }
.power-seg.active-mid    { background: #f59e0b; box-shadow: 0 0 6px rgba(245,158,11,0.6); }
.power-seg.active-high   { background: #22c55e; box-shadow: 0 0 6px rgba(34,197,94,0.6); }
.power-pct {
    font-family: 'Orbitron', monospace; font-size: 0.75rem;
    color: #fef2f2; margin-top: 0.35rem; letter-spacing: 0.1em;
}

/* ── Match song name with glitch ── */
.match-song {
    font-family: 'Orbitron', sans-serif;
    font-size: clamp(1.6rem, 3.5vw, 2.6rem); font-weight: 900;
    color: #fef2f2; line-height: 1.15;
    text-shadow: 0 0 24px rgba(220,38,38,0.4);
    margin: 0.3rem 0;
    animation: glitch-song 6s ease-in-out infinite;
    position: relative;
}
@keyframes glitch-song {
    0%, 85%, 100% { text-shadow: 0 0 24px rgba(220,38,38,0.4); transform: none; }
    87%  { text-shadow: 3px 0 0 rgba(0,229,255,0.7), -3px 0 0 rgba(220,38,38,0.7); transform: skewX(-1deg); }
    89%  { text-shadow: -2px 0 0 rgba(0,229,255,0.7), 2px 0 0 rgba(220,38,38,0.7); transform: skewX(1deg); }
    91%  { text-shadow: 0 0 24px rgba(220,38,38,0.4); transform: none; }
}

/* ── Circular confidence gauge ── */
.gauge-wrap {
    display: flex; align-items: center; gap: 1.5rem; margin-top: 1rem;
}
.gauge-svg { flex-shrink: 0; }
.gauge-info { flex: 1; }
.gauge-label {
    font-family: 'Share Tech Mono', monospace; font-size: 0.6rem;
    letter-spacing: 0.22em; color: #00e5ff; text-transform: uppercase;
    margin-bottom: 0.3rem;
}
.gauge-pct {
    font-family: 'Orbitron', sans-serif; font-size: 1.8rem;
    font-weight: 900; color: #fef2f2; line-height: 1;
}
.gauge-sub {
    font-family: 'Share Tech Mono', monospace; font-size: 0.58rem;
    color: #57534e; margin-top: 0.2rem; letter-spacing: 0.1em;
}

/* ── Waveform equalizer ── */
.waveform-wrap {
    display: flex; align-items: center; justify-content: center;
    gap: 3px; height: 48px; margin: 1rem 0;
}
.waveform-bar {
    width: 4px; border-radius: 2px;
    background: linear-gradient(180deg, #00e5ff, #0891b2);
    box-shadow: 0 0 6px rgba(0,229,255,0.5);
    animation: eq-bounce var(--d, 0.8s) ease-in-out infinite alternate;
    transform-origin: bottom;
}
@keyframes eq-bounce {
    from { transform: scaleY(0.15); opacity: 0.4; }
    to   { transform: scaleY(1);    opacity: 1; }
}
.waveform-label {
    font-family: 'Share Tech Mono', monospace; font-size: 0.6rem;
    letter-spacing: 0.25em; color: #00e5ff; text-align: center;
    text-transform: uppercase; margin-bottom: 0.3rem;
    animation: blink-await 1s step-end infinite;
}

/* ── Scanning animation ── */
.scanning-wrap {
    font-family: 'Share Tech Mono', monospace; font-size: 0.75rem;
    color: #f59e0b; letter-spacing: 0.2em; text-transform: uppercase;
    margin: 0.5rem 0; display: flex; align-items: center; gap: 0.5rem;
}
.scanning-cursor {
    animation: blink-await 0.6s step-end infinite;
}

/* ── Status bar ── */
.status-bar {
    position: fixed; bottom: 0; left: 0; right: 0;
    background: rgba(4,4,10,0.95);
    border-top: 1px solid rgba(220,38,38,0.2);
    padding: 0.3rem 1.5rem;
    display: flex; align-items: center; gap: 1.5rem;
    font-family: 'Share Tech Mono', monospace;
    font-size: 0.55rem; letter-spacing: 0.15em;
    color: #57534e; text-transform: uppercase;
    z-index: 9000;
}
.status-bar-dot {
    width: 5px; height: 5px; border-radius: 50%;
    background: #22c55e;
    box-shadow: 0 0 6px rgba(34,197,94,0.8);
    animation: pulse-status 2s ease-in-out infinite;
    display: inline-block; margin-right: 0.3rem;
}
@keyframes pulse-status {
    0%, 100% { opacity: 1; } 50% { opacity: 0.4; }
}
.status-bar-item { color: #4a4a6a; }
.status-bar-item span { color: #00e5ff; }
</style>
""", unsafe_allow_html=True)


# ─── Ambient effects ─────────────────────────────────────────────────────────

st.markdown("""
<canvas id="hex-canvas"></canvas>
<div class="hud-grid"></div>
<div class="vp-corner tl"></div>
<div class="vp-corner tr"></div>
<div class="vp-corner bl"></div>
<div class="vp-corner br"></div>

<div class="status-bar">
    <span><span class="status-bar-dot"></span></span>
    <span class="status-bar-item">JARVIS <span>v1.0</span></span>
    <span class="status-bar-item">STATUS <span>ONLINE</span></span>
    <span class="status-bar-item">SYSTEM <span>NOMINAL</span></span>
    <span class="status-bar-item">STARK INDUSTRIES AUDIO ID</span>
</div>

<script>
(function() {
    // ── Hex particle canvas ──
    var canvas = document.getElementById('hex-canvas');
    if (!canvas) return;
    var ctx = canvas.getContext('2d');
    canvas.width  = window.innerWidth;
    canvas.height = window.innerHeight;

    var HEX_R = 14, particles = [];
    function hexW(r) { return Math.sqrt(3) * r; }
    function hexH(r) { return 2 * r; }

    function buildGrid() {
        particles = [];
        var COLS = Math.ceil(canvas.width  / (hexW(HEX_R) * 1.05)) + 2;
        var ROWS = Math.ceil(canvas.height / (hexH(HEX_R) * 0.78)) + 2;
        for (var r = 0; r < ROWS; r++) {
            for (var c = 0; c < COLS; c++) {
                var x = c * hexW(HEX_R) * 1.05 + (r % 2) * hexW(HEX_R) * 0.525;
                var y = r * hexH(HEX_R) * 0.78;
                particles.push({ x: x, y: y, alpha: Math.random() * 0.4, speed: 0.003 + Math.random() * 0.006, phase: Math.random() * Math.PI * 2 });
            }
        }
    }

    function drawHex(x, y, r, alpha) {
        ctx.beginPath();
        for (var i = 0; i < 6; i++) {
            var angle = Math.PI / 180 * (60 * i - 30);
            var px = x + r * Math.cos(angle);
            var py = y + r * Math.sin(angle);
            i === 0 ? ctx.moveTo(px, py) : ctx.lineTo(px, py);
        }
        ctx.closePath();
        ctx.strokeStyle = 'rgba(0,229,255,' + alpha + ')';
        ctx.lineWidth = 0.5;
        ctx.stroke();
    }

    var t = 0;
    function animate() {
        ctx.clearRect(0, 0, canvas.width, canvas.height);
        t += 0.016;
        particles.forEach(function(p) {
            var a = (Math.sin(t * p.speed * 60 + p.phase) + 1) / 2 * p.alpha;
            drawHex(p.x, p.y, HEX_R - 1, a);
        });
        requestAnimationFrame(animate);
    }

    buildGrid();
    animate();

    window.addEventListener('resize', function() {
        canvas.width  = window.innerWidth;
        canvas.height = window.innerHeight;
        buildGrid();
    });
})();
</script>
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
        <div class="sec-hdr-bracket">⌐</div>
        <div class="sec-hdr-text">{label}</div>
        <div class="sec-hdr-bracket">¬</div>
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
        conf_pct  = min(100, int((ratio / 10) * 100))
        # circular gauge: circumference of r=28 circle = ~175.9
        circ      = 175.9
        dash      = conf_pct / 100 * circ
        gap       = circ - dash
        # colour based on confidence
        if conf_pct >= 70:   gauge_col = "#22c55e"
        elif conf_pct >= 40: gauge_col = "#f59e0b"
        else:                gauge_col = "#dc2626"
        song_display = winner.replace('_', ' ')
        st.markdown(f"""
        <div class="match-box">
            <div class="match-jarvis">▲ J.A.R.V.I.S  RESPONSE</div>
            <div class="match-phrase">Found it, Boss.</div>
            <div class="match-song">{song_display}</div>
            <div class="match-meta">
                alignment score {top_score:,} &nbsp;·&nbsp; {ratio:.1f}× the runner-up
            </div>
            <div class="gauge-wrap">
                <svg class="gauge-svg" width="80" height="80" viewBox="0 0 80 80">
                    <circle cx="40" cy="40" r="28" fill="none"
                            stroke="rgba(255,255,255,0.06)" stroke-width="6"/>
                    <circle cx="40" cy="40" r="28" fill="none"
                            stroke="{gauge_col}" stroke-width="6"
                            stroke-dasharray="{dash:.1f} {gap:.1f}"
                            stroke-dashoffset="44"
                            stroke-linecap="round"
                            style="filter:drop-shadow(0 0 4px {gauge_col});transition:stroke-dasharray 0.6s ease;"/>
                    <text x="40" y="44" text-anchor="middle"
                          fill="{gauge_col}" font-family="Orbitron,sans-serif"
                          font-size="13" font-weight="900">{conf_pct}%</text>
                </svg>
                <div class="gauge-info">
                    <div class="gauge-label">Signal Confidence</div>
                    <div class="gauge-pct" style="color:{gauge_col};">{conf_pct}%</div>
                    <div class="gauge-sub">HASH ALIGNMENT RATIO {ratio:.1f}×</div>
                </div>
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
            <div class="reactor-outer">
                <div class="reactor-power-ring"></div>
                <div class="reactor-radar"></div>
                <div class="reactor-segments">
                    <svg viewBox="0 0 90 90" xmlns="http://www.w3.org/2000/svg">
                        <g transform="translate(45,45)">
                            <path d="M0,-38 L8,-20 L-8,-20 Z" fill="rgba(220,38,38,0.35)" stroke="#dc2626" stroke-width="0.8"/>
                            <path d="M0,-38 L8,-20 L-8,-20 Z" fill="rgba(220,38,38,0.35)" stroke="#dc2626" stroke-width="0.8" transform="rotate(60)"/>
                            <path d="M0,-38 L8,-20 L-8,-20 Z" fill="rgba(220,38,38,0.35)" stroke="#dc2626" stroke-width="0.8" transform="rotate(120)"/>
                            <path d="M0,-38 L8,-20 L-8,-20 Z" fill="rgba(220,38,38,0.35)" stroke="#dc2626" stroke-width="0.8" transform="rotate(180)"/>
                            <path d="M0,-38 L8,-20 L-8,-20 Z" fill="rgba(220,38,38,0.35)" stroke="#dc2626" stroke-width="0.8" transform="rotate(240)"/>
                            <path d="M0,-38 L8,-20 L-8,-20 Z" fill="rgba(220,38,38,0.35)" stroke="#dc2626" stroke-width="0.8" transform="rotate(300)"/>
                            <circle r="38" fill="none" stroke="rgba(220,38,38,0.4)" stroke-width="1"/>
                            <circle r="28" fill="none" stroke="rgba(245,158,11,0.2)" stroke-width="0.5" stroke-dasharray="4 4"/>
                        </g>
                    </svg>
                </div>
                <div class="reactor-inner-ring"></div>
                <div class="arc-spark" style="bottom:50%;left:calc(50% - 1px);transform-origin:bottom center;transform:translateY(50%) rotate(0deg) translateY(-34px);animation-delay:0s;"></div>
                <div class="arc-spark" style="bottom:50%;left:calc(50% - 1px);transform-origin:bottom center;transform:translateY(50%) rotate(90deg) translateY(-34px);animation-delay:1.2s;"></div>
                <div class="arc-spark" style="bottom:50%;left:calc(50% - 1px);transform-origin:bottom center;transform:translateY(50%) rotate(180deg) translateY(-34px);animation-delay:2.4s;"></div>
                <div class="arc-spark" style="bottom:50%;left:calc(50% - 1px);transform-origin:bottom center;transform:translateY(50%) rotate(270deg) translateY(-34px);animation-delay:3.6s;"></div>
                <div class="reactor-core"></div>
            </div>
        </div>
        <div class="reactor-readout">ARC REACTOR · OUTPUT: 3.00 GJ · STABLE</div>
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
            f'<div class="metric-card"><div class="metric-val">9.6 kHz</div>'
            f'<div class="metric-lbl">Sample Rate</div></div>',
            unsafe_allow_html=True,
        )

    # ── Indexed library ──────────────────────────────────────────────────────
    sec_hdr("⟁ Indexed library — expand a track to inspect its constellation")

    cols = st.columns(4)
    for i, (name, n_hash) in enumerate(songs):
        color      = PALETTE[i % len(PALETTE)]
        peaks_list = db_constellation.get(name, [])
        has_data   = bool(peaks_list)
        dot_color  = "#22c55e" if has_data else "#3f3f46"
        disp_name  = name.replace('_', ' ')
        with cols[i % 4]:
            with st.expander(f"#{i+1:03d}  {disp_name}", expanded=False):
                st.markdown(
                    f'<div class="song-meta" style="color:{color};">'
                    f'{n_hash:,} hashes &nbsp;·&nbsp; '
                    f'<span style="color:{dot_color};">{"● DATA" if has_data else "○ NO DATA"}</span>'
                    f'</div>',
                    unsafe_allow_html=True,
                )
                if has_data:
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
        <div class="page-hdr-title">Identify a Track</div>
        <div class="page-hdr-sub">UPLOAD AUDIO SIGNATURE · JARVIS WILL FIND IT</div>
    </div>""", unsafe_allow_html=True)

    st.markdown('<div style="position:relative;padding:0.8rem;background:rgba(0,229,255,0.015);border:1px solid rgba(0,229,255,0.12);border-radius:4px;"><div class="hud-corner-tr"></div><div class="hud-corner-bl"></div>', unsafe_allow_html=True)
    uploaded = st.file_uploader(
        "Upload Audio Signature",
        type=["wav", "mp3", "flac", "ogg", "m4a"],
        label_visibility="collapsed",
        key="identify_uploader",
    )
    if not uploaded:
        st.markdown('<div class="awaiting-input">⟁ AWAITING AUDIO SIGNATURE</div>', unsafe_allow_html=True)
    st.markdown('</div>', unsafe_allow_html=True)

    # ── Audio player ─────────────────────────────────────────────────────────
    if uploaded:
        sec_hdr("⟁ Clip preview")
        st.audio(uploaded, format=f"audio/{uploaded.name.rsplit('.',1)[-1].lower()}")
        uploaded.seek(0)

    run = st.button("IDENTIFY", type="primary", use_container_width=True)

    if not (uploaded and run):
        return

    # mark that we just ran identify so we stay on this tab
    st.session_state["_active_tab"] = "identify"

    # ── Waveform equalizer while scanning ────────────────────────────────────
    import random
    bars_html = ""
    for b in range(32):
        delay = round(random.uniform(0, 1.2), 2)
        dur   = round(random.uniform(0.4, 1.0), 2)
        h     = random.randint(20, 100)
        bars_html += f'<div class="waveform-bar" style="height:{h}%;--d:{dur}s;animation-delay:{delay}s;"></div>'

    st.markdown(f"""
    <div class="waveform-label">⟁ SCANNING AUDIO SIGNATURE</div>
    <div class="waveform-wrap">{bars_html}</div>
    """, unsafe_allow_html=True)

    with st.spinner(""):
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
        f'generated from {len(result["peaks"])} constellation peaks &nbsp;·&nbsp; '
        f'<span style="color:#f59e0b;">{len(result["peaks"])} peaks in query clip</span></div>',
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
        <div class="page-hdr-title">Batch Identification</div>
        <div class="page-hdr-sub">UPLOAD MULTIPLE TARGETS · GET results.csv</div>
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
        "Upload Multiple Targets",
        type=["wav", "mp3", "flac", "ogg", "m4a"],
        accept_multiple_files=True,
        label_visibility="collapsed",
        key="batch_uploader",
    )
    run = st.button("▶ RUN BATCH", type="primary")

    if not (files and run):
        return

    rows        = []
    all_results = []
    prog        = st.progress(0.0, text="Processing…")

    for i, f in enumerate(files):
        prog.progress((i + 1) / len(files), text=f"Identifying {f.name}…")
        try:
            raw   = f.read()
            audio = read_audio_bytes(raw, f.name)
            res   = identify(audio, db_paired)
            pred  = res["winner"] if res["winner"] else "none"
            score = res["scores"][0][1] if res["scores"] else 0
        except Exception as e:
            res, pred, score = None, "none", 0
        rows.append({"filename": f.name, "prediction": pred, "score": score})
        all_results.append((f.name, res))
    prog.empty()

    # ── Summary table ────────────────────────────────────────────────────────
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

    # ── CSV download ─────────────────────────────────────────────────────────
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

    # ── Per-clip intermediate steps ──────────────────────────────────────────
    sec_hdr("⟁ Per-clip analysis")
    counts       = get_song_hash_counts(db_paired)
    sorted_names = sorted(counts.keys())

    for fname, res in all_results:
        if res is None:
            st.markdown(
                f'<div class="hash-info" style="color:#dc2626;">⚠ {fname} — failed to process</div>',
                unsafe_allow_html=True,
            )
            continue

        winner = res["winner"]
        scores = res["scores"]
        label  = winner.replace("_", " ") if winner else "NO MATCH"
        color  = "#f59e0b" if winner else "#57534e"

        with st.expander(f"▶  {fname}  ·  {label}", expanded=False):

            # verdict chip
            render_match_box(winner, scores)

            # step 1 — spectrogram + constellation
            step_hdr("1", "Spectrogram → Constellation",
                     "Time-frequency map, then top peaks kept as fingerprint")
            fig1 = plot_query_analysis(
                res["freqs"], res["times"], res["Sxx_dB"], res["peaks"]
            )
            st.pyplot(fig1, use_container_width=True)
            plt.close(fig1)
            st.markdown(
                f'<div class="hash-info">→ {res["n_hashes"]:,} paired hashes '
                f'from {len(res["peaks"])} peaks &nbsp;·&nbsp; '
                f'<span style="color:#f59e0b;">{len(res["peaks"])} peaks in query clip</span></div>',
                unsafe_allow_html=True,
            )

            # step 2 — candidate scores
            render_candidates(scores, winner)

            # step 3 — where in the song
            if winner:
                hop_s      = (NPERSEG - NOVERLAP) / SR
                w_color    = song_color(winner, sorted_names)
                peaks_list = db_paired  # placeholder — we use db_constellation below

                step_hdr("2", "Where in the Song?",
                         f"Full constellation of '{label}' with query window highlighted")

                # we need db_constellation — pass it through session state
                db_constellation = st.session_state.get("_db_constellation", {})
                peaks_list = db_constellation.get(winner, [])

                if peaks_list and res["matched_offset"] is not None:
                    win_start = res["matched_offset"] * hop_s
                    win_end   = win_start + res["query_n_frames"] * hop_s
                    fig2 = plot_song_window(
                        peaks_list, label, w_color,
                        highlight_start_s=win_start, highlight_end_s=win_end,
                    )
                    if fig2:
                        st.pyplot(fig2, use_container_width=True)
                        plt.close(fig2)
                    st.markdown(
                        f'<div class="hash-info" style="color:#f59e0b;">'
                        f'⟁ Clip matched from <b>{label}</b>'
                        f' &nbsp;·&nbsp; {win_start:.2f}s → {win_end:.2f}s</div>',
                        unsafe_allow_html=True,
                    )

            # step 4 — offset histogram
            if res["offset_hist"]:
                step_hdr("3", "Alignment Spike",
                         "Hash votes converge on the matching time offset")
                fig3 = plot_offset_histogram(res["offset_hist"], winner)
                if fig3:
                    st.pyplot(fig3, use_container_width=True)
                    plt.close(fig3)

            # timing
            sec_hdr("⟁ Processing time")
            render_timing(res["timing"])


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

    # store for batch tab access
    st.session_state["_db_constellation"] = db_constellation

    # Tab persistence — default to 0 (DATABASE), switch to 1 (IDENTIFY) after a run
    active = 0
    if st.session_state.get("_active_tab") == "identify":
        active = 1
        st.session_state["_active_tab"] = None

    tab1, tab2, tab3 = st.tabs(["  DATABASE  ", "  IDENTIFY  ", "  BATCH  "])

    # Force the active tab via JS
    if active == 1:
        st.markdown("""
        <script>
        (function() {
            var tabs = window.parent.document.querySelectorAll('button[data-baseweb="tab"]');
            if (tabs && tabs[1]) { tabs[1].click(); }
        })();
        </script>""", unsafe_allow_html=True)

    with tab1:
        tab_library(db_paired, db_constellation)
    with tab2:
        tab_identify(db_paired, db_constellation)
    with tab3:
        tab_batch(db_paired)


if __name__ == "__main__":
    main()
