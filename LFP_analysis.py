
"""
Source-data script for the quantitative panels of Supplementary Fig. 9.
    - Registration of channels and regions
    - Match behaviour and ephys files
    - NOE extraction and delta statistics
    - HAB running band-coherence extraction and statistics
    - HAB running power spectra

"""

import os
import re
import glob
import itertools
from collections import defaultdict

import matplotlib
matplotlib.rcParams["pdf.fonttype"] = 42
matplotlib.rcParams["svg.fonttype"] = "none"
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from open_ephys.analysis import Session
from scipy import signal
from scipy.signal import hilbert, welch
from scipy.signal.windows import dpss
from scipy.stats import friedmanchisquare, wilcoxon


np.random.seed(1)


CHANNEL_TO_REGION = {
    "HCl": "HC",
    "HCr": "HC",
    "PFCl": "PFC",
    "PFCr": "PFC",
    "RE": "RE",
}
CHANNEL_TO_HEMI = {
    "HCl": "L",
    "HCr": "R",
    "PFCl": "L",
    "PFCr": "R",
    "RE": "M",
}

PAIR_POOL_MAP = {}
for hemi_letter, suffix in [("L", "l"), ("R", "r")]:
    PAIR_POOL_MAP[tuple(sorted([f"HC{suffix}", f"PFC{suffix}"]))] = ("HC-PFC", hemi_letter)
    PAIR_POOL_MAP[tuple(sorted([f"HC{suffix}", "RE"]))] = ("HC-RE", hemi_letter)
    PAIR_POOL_MAP[tuple(sorted([f"PFC{suffix}", "RE"]))] = ("PFC-RE", hemi_letter)

REGION_FIGURE_LABEL = {"HC": "vHIPP", "PFC": "mPFC", "RE": "RE"}
PAIR_FIGURE_LABEL = {
    "HC-PFC": "mPFC-vHIPP",
    "HC-RE": "RE-vHIPP",
    "PFC-RE": "mPFC-RE",
}
PAC_DIRECTION_LABEL = {
    "HC-PFC": ("vHIPP", "mPFC"),
    "HC-RE": ("vHIPP", "RE"),
    "PFC-RE": ("mPFC", "RE"),
}

CONDITION_GROUPS = {
    "naive": ["naive1", "naive2"],
    "CDM": ["CDM_1", "CDM_2"],
    "C21": ["C21_1"],
}
CONDITION_LABEL = {"naive": "naive", "CDM": "post-CDM", "C21": "post-C21"}
CONDITION_COLOR = {"naive": "#E69F00", "CDM": "#D55E00", "C21": "#56B4E9"}

PSD_REGIONS = ["PFC", "RE", "HC"]
COHERENCE_PAIRS = ["PFC-RE", "HC-RE", "HC-PFC"]

BAR_PANELS_RUNNING = [
    {
        "panel": "m",
        "target": "HC-PFC",
        "metric": "coh",
        "band": "beta",
        "title": "Running - mPFC-vHIPP coherence β",
    },
    {
        "panel": "n",
        "target": "HC-RE",
        "metric": "coh",
        "band": "lowGamma",
        "title": "Running - RE-vHIPP coherence low-γ",
    },
    {
        "panel": "o",
        "target": "HC-PFC",
        "metric": "coh",
        "band": "lowGamma",
        "title": "Running - mPFC-vHIPP coherence low-γ",
    },
]

BAR_PANELS_NOE = [
    {
        "panel": "p",
        "target": "PFC-RE",
        "metric": "pac_mi",
        "band": "theta_lowGamma",
        "title": "NOE - mPFC-RE θ-lowγ PAC-mi",
    },
    {
        "panel": "q",
        "target": "HC-RE",
        "metric": "coh",
        "band": "lowGamma",
        "title": "NOE - RE-vHIPP coherence lowγ",
    },
    {
        "panel": "r",
        "target": "HC-PFC",
        "metric": "pac_mi",
        "band": "theta_lowGamma",
        "title": "NOE - mPFC-vHIPP θ-lowγ PAC-mi",
    },
]

def make_cfg():
    cfg = {
        "rootFoldEphys": r"D:\LTP_analysis\ProcessedData",
        "rootFoldVideo": r"D:\LTP_analysis\ProcessedData\VideoRec",
        "resultsDir": r"D:\LTP_analysis\results",
        "miceAvail": ["10084", "10085", "male_7pins", "male_9pins", "10418", "100134"],
        "batch": [1, 1, 1, 1, 2, 2],
        "conditions": ["naive1", "naive2", "CDM_1", "CDM_2", "C21_1", "C21_2"],
        "test": "NOE",
        "speedThresh_cm_s": 4.0,
        "px_to_cm": 1 / 8,
        "minEpochDur_s_NOE": 1.0,
        "minEpochDur_s_HAB": 0.5,
        "doDownsample": True,
        "fsTarget": 1000,
        "doNotch": False,
        "lineFreq": 50,
        "notchQ": 35,
        "freqRange": (1, 150),
        "bands": {
            "theta": (4, 12),
            "beta": (13, 30),
            "lowGamma": (30, 55),
            "highGamma": (65, 100),
        },
    }
    cfg["metaXlsx"] = os.path.join(cfg["rootFoldEphys"], "LFP_organization_bothbatches.xlsx")
    return cfg
    
def pool_pair_name(a, b):
    return PAIR_POOL_MAP.get(tuple(sorted([a, b])), (None, None))


def pool_channel_name(ch):
    return CHANNEL_TO_REGION.get(ch, ch), CHANNEL_TO_HEMI.get(ch, "?")


def should_skip(mouse, cond, test):
    if mouse == "10418" and cond in ["C21_1", "C21_2"]:
        return True
    return False


def _norm_str(s):
    return str(s).strip()


def find_meta_row(meta, mouse, cond, test):
    m = meta.copy()
    m["mouse_id"] = m["mouse_id"].astype(str).str.strip()
    m["condition"] = m["condition"].astype(str).str.strip()
    m["test"] = m["test"].astype(str).str.strip()
    mask = (
        (m["mouse_id"] == _norm_str(mouse))
        & (m["condition"] == _norm_str(cond))
        & (m["test"] == _norm_str(test))
    )
    idx = np.where(mask.values)[0]
    if len(idx) == 0:
        return None, "no_metadata_match"
    return int(idx[0]), "ok"


def parse_nose_xy(nose_series):
    x = np.zeros(len(nose_series), dtype=float)
    y = np.zeros(len(nose_series), dtype=float)
    for i, s in enumerate(nose_series):
        nums = [float(v) for v in re.findall(r"[-+]?\d*\.?\d+(?:[eE][-+]?\d+)?", str(s))]
        if len(nums) >= 2:
            x[i] = nums[0]
            y[i] = nums[1]
    return x, y


def idx_to_epochs(time_vec, idx, min_dur):
    idx = np.asarray(idx, dtype=int)
    if idx.size == 0:
        return np.array([]), np.array([])
    d = np.diff(idx)
    split = np.where(d > 1)[0]
    starts = np.r_[0, split + 1]
    ends = np.r_[split, len(idx) - 1]
    ep_s = time_vec[idx[starts]]
    ep_e = time_vec[idx[ends]]
    keep = (ep_e - ep_s) >= min_dur
    return ep_s[keep], ep_e[keep]


def resolve_video_dir(video_root, video_folder, batch_id):
    vf = str(video_folder)
    if os.path.isabs(vf) and os.path.exists(vf):
        return vf
    candidate = os.path.join(video_root, vf)
    if os.path.exists(candidate):
        return candidate
    base = os.path.basename(vf.rstrip("\\/"))
    candidate2 = os.path.join(video_root, f"Batch{batch_id}", base)
    if os.path.exists(candidate2):
        return candidate2
    return candidate


def load_behavior_epochs(meta_row, test_name, cfg):
    video_dir = resolve_video_dir(cfg["rootFoldVideo"], meta_row["video_folder"], int(meta_row["batch_id"]))
    trial = str(meta_row["trial_number"])
    trial_suffix = trial[1:] if len(trial) > 1 else trial

    xls_name = f"processed_interpolated_T{trial_suffix}_V2.xlsx"
    xls_path = os.path.join(video_dir, xls_name)
    if not os.path.exists(xls_path):
        pattern = os.path.join(video_dir, f"processed_interpolated_T*{trial_suffix}*_V2.xlsx")
        cand = glob.glob(pattern)
        if cand:
            xls_path = cand[0]

    if not os.path.exists(xls_path):
        return None, None, None, f"behavior_file_missing:{xls_path}"

    b = pd.read_excel(xls_path)
    if "Timestamp_sec" not in b.columns:
        return None, None, None, "behavior_missing_Timestamp_sec"

    b.loc[b["Timestamp_sec"] <= 1, "Timestamp_sec"] = np.nan
    b = b.dropna(subset=["Timestamp_sec"]).copy()
    if len(b) == 0:
        return None, None, None, "behavior_empty_after_crop"

    b["Timestamp_sec"] = b["Timestamp_sec"] - b["Timestamp_sec"].iloc[0]

    if test_name == "NOE":
        if "region1_other" not in b.columns:
            return b, np.array([]), np.array([]), "missing_region1_other"
        idx = np.where(pd.to_numeric(b["region1_other"], errors="coerce").fillna(0).values > 0)[0]
        ep_s, ep_e = idx_to_epochs(b["Timestamp_sec"].values, idx, cfg["minEpochDur_s_NOE"])
        return b, ep_s, ep_e, "ok"

    stop_idx = np.where(b["Timestamp_sec"].values <= 600)[0]
    if len(stop_idx) == 0:
        return b, np.array([]), np.array([]), "hab_no_10min"
    stop10 = stop_idx[-1]

    if "nose" not in b.columns:
        return b, np.array([]), np.array([]), "missing_nose"

    noseX, noseY = parse_nose_xy(b["nose"].iloc[: stop10 + 1].values)
    dX = np.diff(noseX) * (40.0 / 680.0)
    dY = np.diff(noseY) * (30.0 / 481.0)
    dist = np.sqrt(dX ** 2 + dY ** 2)
    dt = np.diff(b["Timestamp_sec"].iloc[: stop10 + 1].values)
    dt[dt <= 0] = np.nan
    vel = dist / dt
    vel = np.nan_to_num(vel, nan=0.0, posinf=0.0, neginf=0.0)
    vel = np.r_[0, vel, 0]
    is_run = vel > cfg["speedThresh_cm_s"]

    idx = np.where(is_run)[0]
    ts = b["Timestamp_sec"].iloc[: len(vel)].values
    ep_s, ep_e = idx_to_epochs(ts, idx, cfg["minEpochDur_s_HAB"])
    return b, ep_s, ep_e, "ok"


def preprocess_signal(x, fs, cfg):
    y = np.asarray(x, dtype=float)
    fs_out = float(fs)

    if cfg["doDownsample"] and fs_out > cfg["fsTarget"]:
        down = int(round(fs_out / cfg["fsTarget"]))
        y = signal.resample_poly(y, 1, down)
        fs_out = fs_out / down

    if cfg["doNotch"]:
        f0 = cfg["lineFreq"]
        q = cfg["notchQ"]
        if fs_out > (2 * (f0 + 2)):
            b, a = signal.iirnotch(w0=f0, Q=q, fs=fs_out)
            y = signal.filtfilt(b, a, y)

    y = signal.detrend(y, type="constant")
    return y, fs_out


def reject_artifact_windows(x, centers, win_sec, fs, thresh_sd=4.0):
    if len(centers) == 0:
        return [], 0
    half = int(round(fs * win_sec / 2))
    n = len(x)
    amps = []
    valid_centers = []
    for c in centers:
        s, e = int(c - half), int(c + half)
        if s >= 0 and e <= n:
            amps.append(np.max(np.abs(x[s:e])))
            valid_centers.append(c)
    if len(amps) == 0:
        return [], 0
    amps = np.array(amps)
    med = np.median(amps)
    mad = np.median(np.abs(amps - med))
    robust_sd = 1.4826 * mad
    cutoff = med + thresh_sd * robust_sd
    keep = amps < cutoff
    clean = [c for c, k in zip(valid_centers, keep) if k]
    n_rejected = int(np.sum(~keep))
    return clean, n_rejected


def compute_speed_on_lfp_time(bdf, t_lfp, px_to_cm=None):
    if "nose" not in bdf.columns:
        return np.zeros(len(t_lfp), dtype=float)

    px_to_cm_x = 40.0 / 680.0
    px_to_cm_y = 30.0 / 481.0

    bts = bdf["Timestamp_sec"].values.astype(float)
    nX, nY = parse_nose_xy(bdf["nose"].values)

    dX = np.diff(nX) * px_to_cm_x
    dY = np.diff(nY) * px_to_cm_y
    dt = np.diff(bts)
    dt[dt <= 0] = np.nan
    vel = np.sqrt(dX ** 2 + dY ** 2) / dt
    vel = np.nan_to_num(vel, nan=0.0, posinf=0.0, neginf=0.0)

    vel_ts = (bts[:-1] + bts[1:]) / 2.0
    speed_lfp = np.interp(t_lfp, vel_ts, vel, left=0.0, right=0.0)
    return speed_lfp


def build_hab_speed_lfp_xy(bdf, t_lfp):
    px_to_cm_x = 40.0 / 680.0
    px_to_cm_y = 30.0 / 481.0

    bts = bdf["Timestamp_sec"].values.astype(float)
    nX, nY = parse_nose_xy(bdf["nose"].values)
    dX = np.diff(nX) * px_to_cm_x
    dY = np.diff(nY) * px_to_cm_y
    dt_b = np.diff(bts)
    dt_b[dt_b <= 0] = np.nan
    vel_b = np.sqrt(dX ** 2 + dY ** 2) / dt_b
    vel_b = np.nan_to_num(vel_b, nan=0.0, posinf=0.0, neginf=0.0)
    vel_ts = (bts[:-1] + bts[1:]) / 2.0
    return np.interp(t_lfp, vel_ts, vel_b, left=0.0, right=0.0)


def tile_bout_to_centers(s_event, e_event, fs, n, win_sec, overlap=0.5):
    half = int(round(fs * win_sec / 2))
    if half < 2 or (2 * half + 1) >= n:
        return []

    bout_dur = float(e_event) - float(s_event)
    if bout_dur < win_sec:
        return []

    step = win_sec * (1.0 - overlap)
    centers = []
    t_center = float(s_event) + win_sec / 2.0

    while (t_center + win_sec / 2.0) <= (float(e_event) + 1e-9):
        t_lo = t_center - win_sec / 2.0
        t_hi = t_center + win_sec / 2.0

        if t_lo < float(s_event) - 1e-9 or t_hi > float(e_event) + 1e-9:
            t_center += step
            continue

        c = int(round(t_center * fs))
        if c >= half and c < (n - half - 1):
            centers.append(c)

        t_center += step

    return centers


def _merge_close_epochs(ep_s, ep_e, max_gap_s, min_dur_s):
    if len(ep_s) == 0:
        return np.array([]), np.array([])
    order = np.argsort(ep_s)
    ep_s = np.asarray(ep_s[order], dtype=float)
    ep_e = np.asarray(ep_e[order], dtype=float)
    ms, me = [ep_s[0]], [ep_e[0]]
    for i in range(1, len(ep_s)):
        if ep_s[i] - me[-1] <= max_gap_s:
            me[-1] = max(me[-1], ep_e[i])
        else:
            ms.append(ep_s[i])
            me.append(ep_e[i])
    ms = np.array(ms)
    me = np.array(me)
    keep = (me - ms) >= min_dur_s
    return ms[keep], me[keep]


def _interval_to_mask(time_vec, starts, ends):
    m = np.zeros(len(time_vec), dtype=bool)
    for s, e in zip(np.asarray(starts, dtype=float), np.asarray(ends, dtype=float)):
        m |= (time_vec >= s) & (time_vec <= e)
    return m


def _object_mask_from_behavior(bdf):
    if "region1_other" in bdf.columns:
        return pd.to_numeric(bdf["region1_other"], errors="coerce").fillna(0).values > 0
    return np.zeros(len(bdf), dtype=bool)


def build_speed_matched_baseline_mask(t_lfp, speed_lfp, event_mask, obj_mask_lfp, fs, win_sec, n_sd=1.0):
    n = len(t_lfp)
    half = int(round(fs * win_sec / 2))

    exp_speeds = speed_lfp[event_mask]
    if len(exp_speeds) < 5:
        return np.zeros(n, dtype=bool)

    mu = float(np.nanmean(exp_speeds))
    sd = float(np.nanstd(exp_speeds))
    if sd == 0:
        sd = max(0.5, 0.05 * mu)

    speed_lo = max(0.0, mu - n_sd * sd)
    speed_hi = mu + n_sd * sd

    speed_ok = (speed_lfp >= speed_lo) & (speed_lfp <= speed_hi)
    not_event = ~event_mask
    not_obj = ~obj_mask_lfp if obj_mask_lfp is not None else np.ones(n, dtype=bool)

    edge_ok = np.zeros(n, dtype=bool)
    edge_ok[half : n - half - 1] = True

    return speed_ok & not_event & not_obj & edge_ok


def sample_baseline_centers(baseline_mask, n_needed, win_sec, fs, rng=None):
    if rng is None:
        rng = np.random.default_rng(seed=42)

    candidates = np.where(baseline_mask)[0]
    if len(candidates) == 0:
        return np.array([], dtype=int)

    min_sep = max(1, int(round(win_sec * fs)))
    order = rng.permutation(len(candidates))
    chosen = []
    for idx in order:
        c = int(candidates[idx])
        if all(abs(c - p) >= min_sep for p in chosen):
            chosen.append(c)
        if len(chosen) >= n_needed:
            break

    return np.array(sorted(chosen), dtype=int)


def butter_bandpass(fs, band, order=3):
    lo = max(0.5, float(band[0]))
    hi = min(float(band[1]), fs / 2 - 1)
    if hi <= lo:
        return None, None
    b, a = signal.butter(order, [lo, hi], btype="bandpass", fs=fs)
    return b, a


def window_slices(centers, half_win, n):
    out = []
    for c in centers:
        s = int(c - half_win)
        e = int(c + half_win + 1)
        if s >= 0 and e <= n:
            out.append((s, e))
    return out


def coherence_band_mean(x, y, fs, centers, win_sec, band, NW=2):
    n = len(x)
    half = int(round(fs * win_sec / 2))
    vals = []
    for s, e in window_slices(centers, half, n):
        segx, segy = x[s:e], y[s:e]
        N = len(segx)
        if N < 32:
            continue
        K = max(1, 2 * NW - 1)
        tapers = dpss(N, NW, K)
        nfft = int(2 ** np.ceil(np.log2(N)))
        freqs = np.fft.rfftfreq(nfft, 1.0 / fs)
        Sxx = np.zeros(len(freqs))
        Syy = np.zeros(len(freqs))
        Sxy = np.zeros(len(freqs), dtype=complex)
        for taper in tapers:
            fx = np.fft.rfft(segx * taper, n=nfft)
            fy = np.fft.rfft(segy * taper, n=nfft)
            Sxx += np.abs(fx) ** 2
            Syy += np.abs(fy) ** 2
            Sxy += fx * np.conj(fy)
        Sxx /= K
        Syy /= K
        Sxy /= K
        denom = Sxx * Syy
        coh = np.where(denom > 0, np.abs(Sxy) ** 2 / denom, 0.0)
        keep = (freqs >= band[0]) & (freqs <= band[1])
        if np.any(keep):
            vals.append(float(np.nanmean(coh[keep])))
    return float(np.nanmean(vals)) if vals else np.nan


def pac_mi(phase_sig, amp_sig, fs, centers, win_sec, phase_band, amp_band, n_bins=18):
    bp1 = butter_bandpass(fs, phase_band)
    bp2 = butter_bandpass(fs, amp_band)
    if bp1[0] is None or bp2[0] is None:
        return np.nan
    n = len(phase_sig)
    half = int(round(fs * win_sec / 2))
    bins = np.linspace(-np.pi, np.pi, n_bins + 1)
    vals = []
    for s, e in window_slices(centers, half, n):
        ph = np.angle(hilbert(signal.filtfilt(bp1[0], bp1[1], phase_sig[s:e])))
        amp = np.abs(hilbert(signal.filtfilt(bp2[0], bp2[1], amp_sig[s:e])))
        mean_amp = np.array(
            [
                np.mean(amp[(ph >= bins[bi]) & (ph < bins[bi + 1])])
                if np.any((ph >= bins[bi]) & (ph < bins[bi + 1]))
                else 0
                for bi in range(n_bins)
            ]
        )
        pa = mean_amp / (np.sum(mean_amp) + 1e-12)
        H = -np.sum(pa * np.log(pa + 1e-12))
        Hmax = np.log(n_bins)
        vals.append((Hmax - H) / (Hmax + 1e-12))
    return float(np.nanmean(vals)) if vals else np.nan


def fdr_bh(pvals):
    p = np.array(pvals, dtype=float)
    n = len(p)
    if n == 0:
        return p
    order = np.argsort(p)
    ranked = p[order]
    q = np.empty(n, dtype=float)
    prev = 1.0
    for i in range(n - 1, -1, -1):
        prev = min(prev, ranked[i] * n / (i + 1))
        q[i] = prev
    q_out = np.empty(n, dtype=float)
    q_out[order] = np.minimum(q, 1.0)
    return q_out


def load_channel_table(cfg):
    channel_data = pd.read_excel(cfg["metaXlsx"], sheet_name="electrodes")
    channel_data["animal"] = channel_data["animal"].astype(str).str.strip()
    return channel_data


def load_meta_table(cfg, batch_id):
    return pd.read_excel(cfg["metaXlsx"], sheet_name=f"batch{batch_id}")


def load_ephys_channels(cfg, meta_row, batch_id, channel_data, mouse):
    this_data_dir = os.path.join(
        cfg["rootFoldEphys"], "EphysRec", f"Batch{batch_id}", str(meta_row["ephy_folder"])
    )
    session_obj = Session(this_data_dir)
    rec = session_obj.recordnodes[0].recordings[0]
    stream_name = list(rec.continuous.keys())[0]
    cont = rec.continuous[stream_name]
    fs_raw = float(cont.metadata.sample_rate)
    t = np.asarray(cont.timestamps)

    eventline = rec.events.copy()
    st = pd.to_numeric(eventline["state"], errors="coerce").fillna(0).astype(int)
    smp = pd.to_numeric(eventline["sample_number"], errors="coerce")

    if np.sum(st == 1) < 1 or np.sum(st == 0) < 1:
        raise RuntimeError("bad_ttl_pattern")

    start_video = max(0, int(smp[st == 1].iloc[0] - np.min(t) * fs_raw + 1))
    stop_video = min(len(t) - 1, int(smp[st == 0].iloc[0] - np.min(t) * fs_raw + 1))
    if stop_video <= start_video:
        raise RuntimeError("empty_ttl_crop")

    ani_idx = np.where(channel_data["animal"].str.lower().values == str(mouse).strip().lower())[0]
    if len(ani_idx) == 0:
        raise RuntimeError("animal_not_in_electrode_sheet")
    ai = int(ani_idx[0])

    samples = np.asarray(cont.samples)
    time_first = samples.shape[0] == len(t)
    ref = (
        (samples[start_video:stop_video, 23] if time_first else samples[23, start_video:stop_video])
        if (batch_id == 1 and (samples.shape[1 if time_first else 0] > 23))
        else 0
    )

    channels = {}
    for name in ["HCl", "HCr", "RE", "PFCl", "PFCr"]:
        ch_id_raw = channel_data.loc[ai, name] if name in channel_data.columns else -1
        m_id = re.findall(r"[-+]?\d+", str(ch_id_raw))
        ch_id = int(m_id[0]) if m_id else -1
        if ch_id < 0:
            channels[name] = np.array([])
            continue
        ch_idx = ch_id - 1
        if time_first:
            channels[name] = (
                np.asarray(samples[start_video:stop_video, ch_idx], dtype=float) - ref
                if 0 <= ch_idx < samples.shape[1]
                else np.array([])
            )
        else:
            channels[name] = (
                np.asarray(samples[ch_idx, start_video:stop_video], dtype=float) - ref
                if 0 <= ch_idx < samples.shape[0]
                else np.array([])
            )
    return channels, fs_raw


def preprocess_channels(channels, fs_raw, cfg, max_time_s=None):
    valid_names = [k for k, v in channels.items() if isinstance(v, np.ndarray) and v.size > 0]
    proc = {}
    fs = None
    for name in valid_names:
        proc[name], fs_this = preprocess_signal(channels[name], fs_raw, cfg)
        if fs is None:
            fs = fs_this
    if not proc:
        return {}, np.nan, np.array([])
    n = min(len(v) for v in proc.values())
    if max_time_s is not None:
        n = min(n, int(max_time_s * fs))
    for name in proc:
        proc[name] = proc[name][:n]
    t_lfp = np.arange(n) / fs
    return proc, fs, t_lfp


def compute_group_summary(subject_df, value_col):
    rows = []
    group_cols = [c for c in subject_df.columns if c not in [value_col, "subject"]]
    for keys, g in subject_df.groupby(group_cols, sort=False):
        if not isinstance(keys, tuple):
            keys = (keys,)
        row = dict(zip(group_cols, keys))
        vals = g[value_col].values.astype(float)
        n = len(vals)
        row["n_subjects"] = n
        row["mean"] = float(np.nanmean(vals))
        row["sem"] = float(np.nanstd(vals, ddof=1) / np.sqrt(n)) if n > 1 else np.nan
        rows.append(row)
    return pd.DataFrame(rows)


def save_dataframe(df, path):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    df.to_csv(path, index=False)


def compute_triplet_stats(
    results_long,
    out_dir,
    prefix,
    epoch_keep,
    domain_keep,
    metric_keep,
    bands_keep,
    win_sec,
    pool_hemispheres=False,
    require_both_hemis=False,
    c21_conds=None,
    normalize_method="zscore",
    apply_fdr=False,
    min_n_event_windows=0,
):
    if c21_conds is None:
        c21_conds = ["C21_1"]

    R0 = results_long.copy()

    if len(domain_keep) > 0:
        R0 = R0[R0["domain"].isin(domain_keep)].copy()
    if len(epoch_keep) > 0:
        R0 = R0[R0["epoch"].isin(epoch_keep)].copy()
    if len(metric_keep) > 0:
        R0 = R0[R0["metric"].isin(metric_keep)].copy()
    if bands_keep != "ALL":
        R0 = R0[R0["band"].isin(bands_keep)].copy()
    if win_sec > 0:
        R0 = R0[np.isclose(R0["win_sec"].astype(float), float(win_sec))].copy()
    if min_n_event_windows > 0 and "n_event_windows" in R0.columns:
        R0 = R0[
            pd.to_numeric(R0["n_event_windows"], errors="coerce").fillna(0).astype(int) >= int(min_n_event_windows)
        ].copy()

    if len(R0) == 0:
        raise RuntimeError(f"{prefix}: no rows left after filtering.")

    base_keys = ["win_sec", "epoch", "domain", "target", "metric", "band"]

    if pool_hemispheres:
        side = R0.groupby(["mouse", "condition"] + base_keys, as_index=False).agg(
            value=("value", "mean"),
            n_hemi=("hemisphere", "nunique"),
        )
        if require_both_hemis:
            is_lr_target = side["target"].isin(["HC", "PFC"])
            side = side[(~is_lr_target) | (side["n_hemi"] == 2)].copy()
        side["subject"] = side["mouse"].astype(str)
    else:
        side = R0.groupby(["mouse", "hemisphere", "condition"] + base_keys, as_index=False).agg(
            value=("value", "mean")
        )
        side["subject"] = side["mouse"].astype(str) + "|" + side["hemisphere"].astype(str)

    merge_keys = ["subject"] + base_keys

    def group_cond(df, conds, label):
        x = df[df["condition"].isin(conds)].copy()
        return x.groupby(merge_keys, as_index=False)["value"].mean().rename(columns={"value": label})

    naive = group_cond(side, CONDITION_GROUPS["naive"], "naive")
    cdm = group_cond(side, CONDITION_GROUPS["CDM"], "cdm")
    c21 = group_cond(side, c21_conds, "c21")

    merged_raw = naive.merge(cdm, on=merge_keys, how="inner").merge(c21, on=merge_keys, how="inner")
    if len(merged_raw) == 0:
        raise RuntimeError(f"{prefix}: no matched Naive/CDM/C21 rows after merging.")

    merged_norm = merged_raw.copy()
    if normalize_method == "zscore":
        for feat_keys, g in merged_norm.groupby(base_keys, sort=False):
            feat_vals = feat_keys if isinstance(feat_keys, tuple) else (feat_keys,)
            for subj in g["subject"].unique():
                mask = merged_norm["subject"] == subj
                for bk, bv in zip(base_keys, feat_vals):
                    mask = mask & (merged_norm[bk] == bv)
                vals = merged_norm.loc[mask, ["naive", "cdm", "c21"]].values.flatten().astype(float)
                mu = np.nanmean(vals)
                sd = np.nanstd(vals)
                if sd > 0:
                    merged_norm.loc[mask, "naive"] = (merged_norm.loc[mask, "naive"] - mu) / sd
                    merged_norm.loc[mask, "cdm"] = (merged_norm.loc[mask, "cdm"] - mu) / sd
                    merged_norm.loc[mask, "c21"] = (merged_norm.loc[mask, "c21"] - mu) / sd
        norm_label = "z-score"
    elif normalize_method == "none":
        norm_label = "raw"
    else:
        raise RuntimeError(f"Unsupported normalization: {normalize_method}")

    rows = []
    for keys, g in merged_norm.groupby(base_keys, sort=False):
        if not isinstance(keys, tuple):
            keys = (keys,)
        win_sec_i, epoch_i, domain_i, target_i, metric_i, band_i = keys
        vn = g["naive"].values.astype(float)
        vc = g["cdm"].values.astype(float)
        v2 = g["c21"].values.astype(float)
        n_subjects = len(g)
        if n_subjects < 3:
            continue

        try:
            chi2, p_friedman = friedmanchisquare(vn, vc, v2)
        except Exception:
            chi2, p_friedman = np.nan, np.nan

        try:
            _, p_naive_cdm = wilcoxon(vn, vc, alternative="two-sided")
        except Exception:
            p_naive_cdm = np.nan

        try:
            cdm_direction = np.nanmean(vc) - np.nanmean(vn)
            if cdm_direction > 0:
                _, p_cdm_c21 = wilcoxon(vc, v2, alternative="greater")
            else:
                _, p_cdm_c21 = wilcoxon(vc, v2, alternative="less")
        except Exception:
            p_cdm_c21 = np.nan

        try:
            _, p_naive_c21 = wilcoxon(vn, v2, alternative="two-sided")
        except Exception:
            p_naive_c21 = np.nan

        d1 = vc - vn
        d2 = v2 - vc
        rows.append(
            {
                "win_sec": float(win_sec_i),
                "epoch": epoch_i,
                "domain": domain_i,
                "target": target_i,
                "metric": metric_i,
                "band": band_i,
                "n_subjects": int(n_subjects),
                "mean_naive": float(np.nanmean(vn)),
                "mean_cdm": float(np.nanmean(vc)),
                "mean_c21": float(np.nanmean(v2)),
                "chi2_friedman": chi2,
                "p_friedman": p_friedman,
                "p_naive_cdm": p_naive_cdm,
                "p_cdm_c21": p_cdm_c21,
                "p_naive_c21": p_naive_c21,
                "restores_frac": float(np.mean(np.sign(d1) == -np.sign(d2))),
                "opposite_sign_group": bool(np.sign(np.nanmean(d1)) == -np.sign(np.nanmean(d2))),
            }
        )

    stats_all = pd.DataFrame(rows)
    if len(stats_all) == 0:
        raise RuntimeError(f"{prefix}: no analyzable groups.")

    if apply_fdr:
        for pcol in ["p_friedman", "p_naive_cdm", "p_cdm_c21", "p_naive_c21"]:
            valid = stats_all[pcol].notna()
            if valid.sum() > 0:
                stats_all.loc[valid, f"{pcol}_fdr"] = fdr_bh(stats_all.loc[valid, pcol].values)

    stats_all = stats_all.sort_values("p_friedman", na_position="last").reset_index(drop=True)

    suffix_hemi = "pooledHemi" if pool_hemispheres else "splitHemi"
    suffix_norm = normalize_method

    save_dataframe(stats_all, os.path.join(out_dir, f"{prefix}_stats_{suffix_hemi}_{suffix_norm}.csv"))
    save_dataframe(merged_raw, os.path.join(out_dir, f"{prefix}_plot_data_raw_{suffix_hemi}_{suffix_norm}.csv"))
    save_dataframe(merged_norm, os.path.join(out_dir, f"{prefix}_plot_data_norm_{suffix_hemi}_{suffix_norm}.csv"))

    return {
        "stats": stats_all,
        "plot_data_raw": merged_raw,
        "plot_data_norm": merged_norm,
        "norm_label": norm_label,
        "suffix_hemi": suffix_hemi,
        "suffix_norm": suffix_norm,
    }


def run_noe_extraction(cfg, out_dir):
    channel_data = load_channel_table(cfg)
    win_sec = 0.5
    win_overlap = 0.5
    min_epoch_s = 0.5
    max_gap_s = 0.75
    min_exploration_s = 3.0
    min_bouts_per_session = 2
    baseline_n_sd = 1.0
    reject_baseline_if_object = True
    random_seed = 42
    artifact_thresh_sd = 4.0

    rng = np.random.default_rng(seed=random_seed)
    conn_bands = ["theta", "beta", "lowGamma"]
    pac_pairs = [("theta", "lowGamma")]

    all_rows = []
    logs = []
    bout_qc_rows = []

    orig_min_epoch = cfg["minEpochDur_s_NOE"]
    cfg["minEpochDur_s_NOE"] = 0.01

    for mi, mouse in enumerate(cfg["miceAvail"]):
        batch_id = int(cfg["batch"][mi])
        meta = load_meta_table(cfg, batch_id)

        for cond in cfg["conditions"]:
            test = "NOE"
            session_id = f"{mouse}|{cond}|{test}"

            if should_skip(mouse, cond, test):
                logs.append({"session_id": session_id, "mouse": mouse, "batch": batch_id, "condition": cond, "test": test, "status": "skipped_rule"})
                continue

            row_id, msg = find_meta_row(meta, mouse, cond, test)
            if row_id is None:
                logs.append({"session_id": session_id, "mouse": mouse, "batch": batch_id, "condition": cond, "test": test, "status": msg})
                continue

            md = meta.loc[row_id].copy()
            md["batch_id"] = batch_id

            try:
                channels, fs_raw = load_ephys_channels(cfg, md, batch_id, channel_data, mouse)
            except Exception as e:
                logs.append({"session_id": session_id, "mouse": mouse, "batch": batch_id, "condition": cond, "test": test, "status": f"load_ephys_failed:{e}"})
                continue

            bdf, ep_s, ep_e, bmsg = load_behavior_epochs(md, test, cfg)
            if bdf is None:
                logs.append({"session_id": session_id, "mouse": mouse, "batch": batch_id, "condition": cond, "test": test, "status": bmsg})
                continue

            n_raw = len(ep_s)
            ep_s, ep_e = _merge_close_epochs(ep_s, ep_e, max_gap_s, min_epoch_s)
            n_after = len(ep_s)
            total_exploration_s = float(np.sum(ep_e - ep_s)) if n_after > 0 else 0.0

            if n_after < min_bouts_per_session or total_exploration_s < min_exploration_s:
                reason = f"too_few_bouts:{n_after}" if n_after < min_bouts_per_session else f"too_little_exploration:{total_exploration_s:.1f}s"
                logs.append(
                    {
                        "session_id": session_id,
                        "mouse": mouse,
                        "batch": batch_id,
                        "condition": cond,
                        "test": test,
                        "status": reason,
                        "n_raw_bouts": n_raw,
                        "n_merged_bouts": n_after,
                        "total_exploration_s": total_exploration_s,
                    }
                )
                continue

            proc, fs, t_lfp = preprocess_channels(channels, fs_raw, cfg)
            if not proc or len(t_lfp) < int(fs):
                logs.append({"session_id": session_id, "mouse": mouse, "batch": batch_id, "condition": cond, "test": test, "status": "too_short"})
                continue

            n = len(t_lfp)
            event_mask = _interval_to_mask(t_lfp, ep_s, ep_e)
            speed_lfp = compute_speed_on_lfp_time(bdf, t_lfp, cfg["px_to_cm"])

            obj_mask_lfp = None
            if reject_baseline_if_object:
                bts = bdf["Timestamp_sec"].values.astype(float)
                obj_beh = _object_mask_from_behavior(bdf).astype(float)
                obj_mask_lfp = (
                    np.interp(t_lfp, bts, obj_beh, left=0.0, right=0.0) > 0.5
                    if len(bts) > 1
                    else np.zeros(n, dtype=bool)
                )

            event_centers_list = []
            per_bout_qc = []
            for s_evt, e_evt in zip(ep_s.astype(float), ep_e.astype(float)):
                centers_this_bout = tile_bout_to_centers(s_evt, e_evt, fs, n, win_sec=win_sec, overlap=win_overlap)
                event_centers_list.extend(centers_this_bout)
                per_bout_qc.append(
                    {
                        "session_id": session_id,
                        "mouse": mouse,
                        "condition": cond,
                        "bout_start_s": float(s_evt),
                        "bout_end_s": float(e_evt),
                        "bout_duration_s": float(e_evt - s_evt),
                        "n_windows_tiled": len(centers_this_bout),
                    }
                )

            event_centers = np.array(sorted(event_centers_list), dtype=int)
            if len(event_centers) == 0:
                logs.append({"session_id": session_id, "mouse": mouse, "batch": batch_id, "condition": cond, "test": test, "status": "no_event_windows"})
                continue

            artifact_keep = np.ones(len(event_centers), dtype=bool)
            for _, x_ch in proc.items():
                clean_centers_ch, _ = reject_artifact_windows(x_ch, list(event_centers), win_sec, fs, thresh_sd=artifact_thresh_sd)
                clean_set = set(clean_centers_ch)
                for ci, c in enumerate(event_centers):
                    if c not in clean_set:
                        artifact_keep[ci] = False
            event_centers = event_centers[artifact_keep]
            n_artifact_rejected = int(np.sum(~artifact_keep))
            n_event_windows = len(event_centers)
            if n_event_windows == 0:
                logs.append({"session_id": session_id, "mouse": mouse, "batch": batch_id, "condition": cond, "test": test, "status": "all_event_windows_rejected"})
                continue

            baseline_mask = build_speed_matched_baseline_mask(
                t_lfp,
                speed_lfp,
                event_mask,
                obj_mask_lfp,
                fs=fs,
                win_sec=win_sec,
                n_sd=baseline_n_sd,
            )
            n_pool = int(np.sum(baseline_mask))
            baseline_centers = sample_baseline_centers(baseline_mask, n_needed=n_event_windows, win_sec=win_sec, fs=fs, rng=rng)

            baseline_keep = np.ones(len(baseline_centers), dtype=bool)
            for _, x_ch in proc.items():
                clean_centers_ch, _ = reject_artifact_windows(x_ch, list(baseline_centers), win_sec, fs, thresh_sd=artifact_thresh_sd)
                clean_set = set(clean_centers_ch)
                for ci, c in enumerate(baseline_centers):
                    if c not in clean_set:
                        baseline_keep[ci] = False
            baseline_centers = baseline_centers[baseline_keep]
            n_baseline_windows = len(baseline_centers)
            n_valid_pairs = min(n_event_windows, n_baseline_windows)

            bout_qc_rows.extend(per_bout_qc)
            epoch_centers = {"event": event_centers, "baseline": baseline_centers}
            epoch_metrics = {}

            for epoch_name, centers in epoch_centers.items():
                m = {}
                if len(centers) == 0:
                    epoch_metrics[epoch_name] = m
                    continue

                for a, b in itertools.combinations(sorted(proc.keys()), 2):
                    canonical, pair_hemi = pool_pair_name(a, b)
                    if canonical is None:
                        continue
                    xa, xb = proc[a], proc[b]

                    for bname in conn_bands:
                        val = coherence_band_mean(xa, xb, fs, centers, win_sec, cfg["bands"][bname])
                        m[("pair", canonical, pair_hemi, "coh", bname)] = val

                    for phase_band_name, amp_band_name in pac_pairs:
                        mi_val = pac_mi(
                            xa,
                            xb,
                            fs,
                            centers,
                            win_sec,
                            cfg["bands"][phase_band_name],
                            cfg["bands"][amp_band_name],
                        )
                        m[("pair", canonical, pair_hemi, "pac_mi", f"{phase_band_name}_{amp_band_name}")] = mi_val

                epoch_metrics[epoch_name] = m

                for key, val in m.items():
                    domain, target, hemi, metric, band = key
                    phase_region, amp_region = PAC_DIRECTION_LABEL.get(target, ("", ""))
                    all_rows.append(
                        {
                            "session_id": session_id,
                            "mouse": mouse,
                            "batch": batch_id,
                            "condition": cond,
                            "test": test,
                            "epoch": epoch_name,
                            "win_sec": win_sec,
                            "domain": domain,
                            "target": target,
                            "target_figure_label": PAIR_FIGURE_LABEL.get(target, target),
                            "hemisphere": hemi,
                            "metric": metric,
                            "band": band,
                            "phase_region": phase_region if metric == "pac_mi" else "",
                            "amplitude_region": amp_region if metric == "pac_mi" else "",
                            "value": float(val) if np.isfinite(val) else np.nan,
                            "notes": "ok",
                            "n_raw_bouts": int(n_raw),
                            "n_merged_bouts": int(n_after),
                            "total_exploration_s": float(total_exploration_s),
                            "n_event_windows": int(n_event_windows),
                            "n_baseline_windows": int(n_baseline_windows),
                            "n_valid_pairs": int(n_valid_pairs),
                            "baseline_pool_size": int(n_pool),
                            "n_artifact_rejected": int(n_artifact_rejected),
                        }
                    )

            me = epoch_metrics.get("event", {})
            mb = epoch_metrics.get("baseline", {})
            for kk in set(me.keys()) & set(mb.keys()):
                ve, vb = me[kk], mb[kk]
                vd = ve - vb if np.isfinite(ve) and np.isfinite(vb) else np.nan
                domain, target, hemi, metric, band = kk
                phase_region, amp_region = PAC_DIRECTION_LABEL.get(target, ("", ""))
                all_rows.append(
                    {
                        "session_id": session_id,
                        "mouse": mouse,
                        "batch": batch_id,
                        "condition": cond,
                        "test": test,
                        "epoch": "delta",
                        "win_sec": win_sec,
                        "domain": domain,
                        "target": target,
                        "target_figure_label": PAIR_FIGURE_LABEL.get(target, target),
                        "hemisphere": hemi,
                        "metric": metric,
                        "band": band,
                        "phase_region": phase_region if metric == "pac_mi" else "",
                        "amplitude_region": amp_region if metric == "pac_mi" else "",
                        "value": float(vd) if np.isfinite(vd) else np.nan,
                        "notes": "event_minus_speedMatchedBaseline",
                        "n_raw_bouts": int(n_raw),
                        "n_merged_bouts": int(n_after),
                        "total_exploration_s": float(total_exploration_s),
                        "n_event_windows": int(n_event_windows),
                        "n_baseline_windows": int(n_baseline_windows),
                        "n_valid_pairs": int(n_valid_pairs),
                        "baseline_pool_size": int(n_pool),
                        "n_artifact_rejected": int(n_artifact_rejected),
                    }
                )

            logs.append(
                {
                    "session_id": session_id,
                    "mouse": mouse,
                    "batch": batch_id,
                    "condition": cond,
                    "test": test,
                    "status": "ok",
                    "n_raw_bouts": int(n_raw),
                    "n_merged_bouts": int(n_after),
                    "total_exploration_s": float(total_exploration_s),
                    "n_event_windows": int(n_event_windows),
                    "n_baseline_windows": int(n_baseline_windows),
                    "n_valid_pairs": int(n_valid_pairs),
                    "baseline_pool_size": int(n_pool),
                    "n_artifact_rejected": int(n_artifact_rejected),
                }
            )

    cfg["minEpochDur_s_NOE"] = orig_min_epoch

    results_long = pd.DataFrame(all_rows)
    session_log = pd.DataFrame(logs)
    bout_qc = pd.DataFrame(bout_qc_rows)

    save_dataframe(results_long, os.path.join(out_dir, "NOE_results_long.csv"))
    save_dataframe(session_log, os.path.join(out_dir, "NOE_session_log.csv"))
    save_dataframe(bout_qc, os.path.join(out_dir, "NOE_bout_qc.csv"))

    return results_long, session_log, bout_qc


def run_hab_running_band_coherence(cfg, out_dir):
    channel_data = load_channel_table(cfg)

    win_sec = 0.5
    win_overlap = 0.5
    max_time_s = 600
    speed_run_thresh = 5.0
    min_windows = 3
    artifact_thresh = 4.0

    long_rows = []
    session_log = []

    for mi, mouse in enumerate(cfg["miceAvail"]):
        batch_id = int(cfg["batch"][mi])
        meta = load_meta_table(cfg, batch_id)

        for cond in cfg["conditions"]:
            test = "HAB"
            session_id = f"{mouse}|{cond}|{test}"

            if should_skip(mouse, cond, test):
                session_log.append({"session_id": session_id, "status": "skipped_rule"})
                continue

            row_id, msg = find_meta_row(meta, mouse, cond, test)
            if row_id is None:
                session_log.append({"session_id": session_id, "status": msg})
                continue

            md = meta.loc[row_id].copy()
            md["batch_id"] = batch_id

            try:
                channels, fs_raw = load_ephys_channels(cfg, md, batch_id, channel_data, mouse)
            except Exception as e:
                session_log.append({"session_id": session_id, "status": f"ephys_failed:{e}"})
                continue

            video_dir = resolve_video_dir(cfg["rootFoldVideo"], md["video_folder"], batch_id)
            trial = str(md["trial_number"])
            trial_suffix = trial[1:] if len(trial) > 1 else trial
            xls_name = f"processed_interpolated_T{trial_suffix}_V2.xlsx"
            xls_path = os.path.join(video_dir, xls_name)
            if not os.path.exists(xls_path):
                pattern = os.path.join(video_dir, f"processed_interpolated_T*{trial_suffix}*_V2.xlsx")
                cand = glob.glob(pattern)
                if cand:
                    xls_path = cand[0]

            if not os.path.exists(xls_path):
                session_log.append({"session_id": session_id, "status": "behavior_missing"})
                continue

            bdf = pd.read_excel(xls_path)
            if "Timestamp_sec" not in bdf.columns or "nose" not in bdf.columns:
                session_log.append({"session_id": session_id, "status": "missing_columns"})
                continue

            bdf.loc[bdf["Timestamp_sec"] <= 1, "Timestamp_sec"] = np.nan
            bdf = bdf.dropna(subset=["Timestamp_sec"]).copy()
            if len(bdf) == 0:
                session_log.append({"session_id": session_id, "status": "empty_behavior"})
                continue
            bdf["Timestamp_sec"] = bdf["Timestamp_sec"] - bdf["Timestamp_sec"].iloc[0]
            bdf = bdf[bdf["Timestamp_sec"] <= max_time_s].copy()
            if len(bdf) < 100:
                session_log.append({"session_id": session_id, "status": "too_short"})
                continue

            proc, fs, t_lfp = preprocess_channels(channels, fs_raw, cfg, max_time_s=max_time_s)
            if not proc or len(t_lfp) < int(fs * 2):
                session_log.append({"session_id": session_id, "status": "too_short_lfp"})
                continue

            speed_lfp = build_hab_speed_lfp_xy(bdf, t_lfp)
            n = len(t_lfp)
            half = int(round(fs * win_sec / 2))
            step = int(win_sec * (1 - win_overlap) * fs)
            all_centers = np.arange(half, n - half, step, dtype=int)
            if len(all_centers) < 10:
                session_log.append({"session_id": session_id, "status": "too_few_windows"})
                continue

            win_speeds = np.array([np.mean(speed_lfp[max(0, c - half) : min(n, c + half)]) for c in all_centers])

            clean_mask = np.ones(len(all_centers), dtype=bool)
            for _, x_ch in proc.items():
                cl, _ = reject_artifact_windows(x_ch, list(all_centers), win_sec, fs, artifact_thresh)
                cl_set = set(cl)
                for j, c in enumerate(all_centers):
                    if c not in cl_set:
                        clean_mask[j] = False

            clean_centers = all_centers[clean_mask]
            clean_speeds = win_speeds[clean_mask]
            run_centers = clean_centers[clean_speeds > speed_run_thresh]

            if len(run_centers) < min_windows:
                session_log.append({"session_id": session_id, "status": "too_few_running_windows"})
                continue

            valid_names = [k for k, v in proc.items() if isinstance(v, np.ndarray) and v.size > 0]
            pair_list = list(itertools.combinations(valid_names, 2))

            for chA, chB in pair_list:
                pair_name, pair_hemi = pool_pair_name(chA, chB)
                if pair_name is None:
                    continue
                xA, xB = proc[chA], proc[chB]
                for bname in ["theta", "beta", "lowGamma"]:
                    val = coherence_band_mean(xA, xB, fs, run_centers, win_sec, cfg["bands"][bname])
                    if np.isfinite(val):
                        long_rows.append(
                            {
                                "session_id": session_id,
                                "mouse": mouse,
                                "condition": cond,
                                "hemisphere": pair_hemi,
                                "domain": "pair",
                                "target": pair_name,
                                "target_figure_label": PAIR_FIGURE_LABEL.get(pair_name, pair_name),
                                "epoch": "running",
                                "win_sec": win_sec,
                                "n_windows": len(run_centers),
                                "metric": "coh",
                                "band": bname,
                                "value": float(val),
                            }
                        )

            session_log.append(
                {
                    "session_id": session_id,
                    "mouse": mouse,
                    "condition": cond,
                    "status": "ok",
                    "n_run": len(run_centers),
                    "mean_speed": float(np.mean(clean_speeds)) if len(clean_speeds) else np.nan,
                }
            )

    results_long = pd.DataFrame(long_rows)
    session_log = pd.DataFrame(session_log)

    save_dataframe(results_long, os.path.join(out_dir, "HAB_running_band_coherence_results_long.csv"))
    save_dataframe(session_log, os.path.join(out_dir, "HAB_running_band_coherence_session_log.csv"))

    return results_long, session_log


def run_hab_psd_and_coherence_spectra(cfg, out_dir):
    channel_data = load_channel_table(cfg)

    win_sec_welch = 1.0
    overlap_welch = 0.5
    freq_range = (1, 80)
    speed_thresh = 5.0

    psd_store = defaultdict(lambda: defaultdict(lambda: defaultdict(list)))
    coh_store = defaultdict(lambda: defaultdict(lambda: defaultdict(list)))
    freqs_ref_psd = None
    freqs_ref_coh = None

    cond_map = {c: label for label, conds in CONDITION_GROUPS.items() for c in conds}

    for mi, mouse in enumerate(cfg["miceAvail"]):
        batch_id = int(cfg["batch"][mi])
        meta = load_meta_table(cfg, batch_id)

        for cond in cfg["conditions"]:
            if cond not in cond_map:
                continue

            cond_label = cond_map[cond]
            test = "HAB"
            session_id = f"{mouse}|{cond}|{test}"

            if should_skip(mouse, cond, test):
                continue

            row_id, _ = find_meta_row(meta, mouse, cond, test)
            if row_id is None:
                continue

            md = meta.loc[row_id].copy()
            md["batch_id"] = batch_id

            try:
                channels, fs_raw = load_ephys_channels(cfg, md, batch_id, channel_data, mouse)
            except Exception:
                continue

            bdf, _, _, bmsg = load_behavior_epochs(md, test, cfg)
            if bdf is None:
                continue

            proc, fs, t_lfp = preprocess_channels(channels, fs_raw, cfg)
            if not proc:
                continue

            speed_lfp = compute_speed_on_lfp_time(bdf, t_lfp, cfg["px_to_cm"])
            running_mask = speed_lfp >= speed_thresh
            if running_mask.sum() < int(win_sec_welch * fs):
                continue

            nperseg = int(win_sec_welch * fs)
            noverlap = int(nperseg * overlap_welch)

            for ch_name, x in proc.items():
                if ch_name not in CHANNEL_TO_REGION:
                    continue
                region, hemi = pool_channel_name(ch_name)
                if region not in PSD_REGIONS:
                    continue

                store_key = f"{mouse}|{hemi}"
                x_run = x[running_mask]
                if len(x_run) < nperseg:
                    continue

                freqs, psd = welch(
                    x_run,
                    fs=fs,
                    nperseg=nperseg,
                    noverlap=noverlap,
                    window="hann",
                    scaling="density",
                )
                if freqs_ref_psd is None:
                    freqs_ref_psd = freqs
                psd_store[cond_label][region][store_key].append(psd)

            for a, b in itertools.combinations(sorted(proc.keys()), 2):
                pair_name, pair_hemi = pool_pair_name(a, b)
                if pair_name is None:
                    continue
                xa = proc[a][running_mask]
                xb = proc[b][running_mask]
                if len(xa) < nperseg or len(xb) < nperseg:
                    continue
                freqs_c, coh = signal.coherence(
                    xa,
                    xb,
                    fs=fs,
                    nperseg=nperseg,
                    noverlap=noverlap,
                    window="hann",
                )
                if freqs_ref_coh is None:
                    freqs_ref_coh = freqs_c
                store_key = f"{mouse}|{pair_hemi}"
                coh_store[cond_label][pair_name][store_key].append(coh)

    if freqs_ref_psd is None:
        raise RuntimeError("No HAB running PSD could be computed.")
    if freqs_ref_coh is None:
        raise RuntimeError("No HAB running coherence spectrum could be computed.")

    freq_mask_psd = (freqs_ref_psd >= freq_range[0]) & (freqs_ref_psd <= freq_range[1])
    freq_mask_coh = (freqs_ref_coh >= freq_range[0]) & (freqs_ref_coh <= freq_range[1])

    psd_subject_rows = []
    for cond_label, reg_dict in psd_store.items():
        for region, subj_dict in reg_dict.items():
            for subj_key, psd_list in subj_dict.items():
                subj_mean = np.nanmean(np.stack(psd_list, axis=0), axis=0)
                for f, v in zip(freqs_ref_psd[freq_mask_psd], subj_mean[freq_mask_psd]):
                    psd_subject_rows.append(
                        {
                            "analysis": "running_psd",
                            "condition_group": cond_label,
                            "condition_label": CONDITION_LABEL[cond_label],
                            "region": region,
                            "region_figure_label": REGION_FIGURE_LABEL[region],
                            "subject": subj_key,
                            "frequency_hz": float(f),
                            "value": float(v),
                        }
                    )

    psd_subject = pd.DataFrame(psd_subject_rows)
    psd_summary = compute_group_summary(psd_subject, "value")

    coh_subject_rows = []
    for cond_label, pair_dict in coh_store.items():
        for pair_name, subj_dict in pair_dict.items():
            for subj_key, coh_list in subj_dict.items():
                subj_mean = np.nanmean(np.stack(coh_list, axis=0), axis=0)
                for f, v in zip(freqs_ref_coh[freq_mask_coh], subj_mean[freq_mask_coh]):
                    coh_subject_rows.append(
                        {
                            "analysis": "running_coherence_spectrum",
                            "condition_group": cond_label,
                            "condition_label": CONDITION_LABEL[cond_label],
                            "target": pair_name,
                            "target_figure_label": PAIR_FIGURE_LABEL[pair_name],
                            "subject": subj_key,
                            "frequency_hz": float(f),
                            "value": float(v),
                        }
                    )

    coh_subject = pd.DataFrame(coh_subject_rows)
    coh_summary = compute_group_summary(coh_subject, "value")

    save_dataframe(psd_subject, os.path.join(out_dir, "running_psd_subject_curves.csv"))
    save_dataframe(psd_summary, os.path.join(out_dir, "running_psd_group_summary.csv"))
    save_dataframe(coh_subject, os.path.join(out_dir, "running_coherence_spectrum_subject_curves.csv"))
    save_dataframe(coh_summary, os.path.join(out_dir, "running_coherence_spectrum_group_summary.csv"))

    return psd_subject, psd_summary, coh_subject, coh_summary


def _star(p):
    if not np.isfinite(p):
        return "ns"
    if p < 0.001:
        return "***"
    if p < 0.01:
        return "**"
    if p < 0.05:
        return "*"
    return "ns"


def export_selected_bar_panel_csvs(panel_specs, stats_dict, out_dir, prefix):
    stats_all = stats_dict["stats"]
    plot_data_norm = stats_dict["plot_data_norm"]
    plot_data_raw = stats_dict["plot_data_raw"]

    combined_rows = []

    for spec in panel_specs:
        mask = (
            (stats_all["domain"] == "pair")
            & (stats_all["target"] == spec["target"])
            & (stats_all["metric"] == spec["metric"])
            & (stats_all["band"] == spec["band"])
        )
        stat_row = stats_all.loc[mask].copy()
        if len(stat_row) == 0:
            continue
        stat_row = stat_row.iloc[[0]].copy()
        stat_row["panel"] = spec["panel"]
        stat_row["figure_title"] = spec["title"]

        plot_mask = (
            np.isclose(plot_data_norm["win_sec"].astype(float), float(stat_row["win_sec"].iloc[0]))
            & (plot_data_norm["epoch"] == stat_row["epoch"].iloc[0])
            & (plot_data_norm["domain"] == stat_row["domain"].iloc[0])
            & (plot_data_norm["target"] == stat_row["target"].iloc[0])
            & (plot_data_norm["metric"] == stat_row["metric"].iloc[0])
            & (plot_data_norm["band"] == stat_row["band"].iloc[0])
        )
        d_norm = plot_data_norm.loc[plot_mask].copy()
        d_raw = plot_data_raw.loc[plot_mask].copy()

        panel_rows = []
        for _, row in d_norm.iterrows():
            raw_row = d_raw.loc[d_raw["subject"] == row["subject"]]
            raw_vals = raw_row.iloc[0] if len(raw_row) else None
            for cond_key in ["naive", "cdm", "c21"]:
                out_row = {
                    "panel": spec["panel"],
                    "figure_title": spec["title"],
                    "subject": row["subject"],
                    "target": row["target"],
                    "target_figure_label": PAIR_FIGURE_LABEL.get(row["target"], row["target"]),
                    "metric": row["metric"],
                    "band": row["band"],
                    "condition_group": cond_key.upper() if cond_key != "naive" else "naive",
                    "condition_label": CONDITION_LABEL["naive"] if cond_key == "naive" else CONDITION_LABEL[cond_key.upper()],
                    "value_norm": float(row[cond_key]),
                    "value_raw": float(raw_vals[cond_key]) if raw_vals is not None else np.nan,
                }
                if row["metric"] == "pac_mi":
                    phase_region, amp_region = PAC_DIRECTION_LABEL.get(row["target"], ("", ""))
                    out_row["phase_region"] = phase_region
                    out_row["amplitude_region"] = amp_region
                panel_rows.append(out_row)

        panel_df = pd.DataFrame(panel_rows)
        save_dataframe(panel_df, os.path.join(out_dir, f"{prefix}_panel_{spec['panel']}.csv"))
        save_dataframe(stat_row, os.path.join(out_dir, f"{prefix}_panel_{spec['panel']}_stats.csv"))
        combined_rows.append(panel_df)

    if combined_rows:
        save_dataframe(pd.concat(combined_rows, ignore_index=True), os.path.join(out_dir, f"{prefix}_all_panels_long.csv"))


def export_curve_panel_csvs(panel_key, subject_df, summary_df, out_dir):
    save_dataframe(subject_df, os.path.join(out_dir, f"{panel_key}_subject_curves.csv"))
    save_dataframe(summary_df, os.path.join(out_dir, f"{panel_key}_group_summary.csv"))


def add_pairwise_brackets(ax, stat_row, y_top, y_step):
    p_nc = float(stat_row.get("p_naive_cdm", np.nan))
    p_cc = float(stat_row.get("p_cdm_c21", np.nan))
    p_n2 = float(stat_row.get("p_naive_c21", np.nan))

    level = y_top
    if np.isfinite(p_nc) and p_nc < 0.05:
        ax.plot([0, 0, 1, 1], [level, level + 0.04 * y_step, level + 0.04 * y_step, level], "k-", lw=0.8)
        ax.text(0.5, level + 0.05 * y_step, _star(p_nc), ha="center", va="bottom", fontsize=9)
        level += 0.12 * y_step

    if np.isfinite(p_cc) and p_cc < 0.05:
        ax.plot([1, 1, 2, 2], [level, level + 0.04 * y_step, level + 0.04 * y_step, level], "k-", lw=0.8)
        ax.text(1.5, level + 0.05 * y_step, _star(p_cc), ha="center", va="bottom", fontsize=9)
        level += 0.12 * y_step

    if np.isfinite(p_n2) and p_n2 < 0.05:
        ax.plot([0, 0, 2, 2], [level, level + 0.04 * y_step, level + 0.04 * y_step, level], "k-", lw=0.8)
        ax.text(1.0, level + 0.05 * y_step, _star(p_n2), ha="center", va="bottom", fontsize=9)


def plot_selected_bar_panel(ax, spec, stats_dict):
    stats_all = stats_dict["stats"]
    plot_data_norm = stats_dict["plot_data_norm"]

    mask = (
        (stats_all["domain"] == "pair")
        & (stats_all["target"] == spec["target"])
        & (stats_all["metric"] == spec["metric"])
        & (stats_all["band"] == spec["band"])
    )
    if mask.sum() == 0:
        ax.set_axis_off()
        ax.text(0.5, 0.5, f"Missing panel {spec['panel']}", ha="center", va="center")
        return

    stat_row = stats_all.loc[mask].iloc[0]
    plot_mask = (
        np.isclose(plot_data_norm["win_sec"].astype(float), float(stat_row["win_sec"]))
        & (plot_data_norm["epoch"] == stat_row["epoch"])
        & (plot_data_norm["domain"] == stat_row["domain"])
        & (plot_data_norm["target"] == stat_row["target"])
        & (plot_data_norm["metric"] == stat_row["metric"])
        & (plot_data_norm["band"] == stat_row["band"])
    )
    d = plot_data_norm.loc[plot_mask].copy()
    if len(d) == 0:
        ax.set_axis_off()
        ax.text(0.5, 0.5, f"No data for panel {spec['panel']}", ha="center", va="center")
        return

    x = np.array([0, 1, 2], dtype=float)
    values = np.c_[d["naive"].values.astype(float), d["cdm"].values.astype(float), d["c21"].values.astype(float)]

    means = np.nanmean(values, axis=0)
    ax.bar(x, means, width=0.55, color=[CONDITION_COLOR["naive"], CONDITION_COLOR["CDM"], CONDITION_COLOR["C21"]], edgecolor="black", linewidth=0.8, zorder=1)

    for row in values:
        ax.plot(x, row, "-o", color="0.35", lw=0.8, ms=3.5, mfc="white", zorder=2)
        ax.scatter([0], [row[0]], s=12, color=CONDITION_COLOR["naive"], zorder=3, edgecolor="black", linewidth=0.2)
        ax.scatter([1], [row[1]], s=12, color=CONDITION_COLOR["CDM"], zorder=3, edgecolor="black", linewidth=0.2)
        ax.scatter([2], [row[2]], s=12, color=CONDITION_COLOR["C21"], zorder=3, edgecolor="black", linewidth=0.2)

    all_vals = values.flatten()
    ymax = float(np.nanmax(all_vals))
    ymin = float(np.nanmin(all_vals))
    dy = (ymax - ymin) if ymax > ymin else (abs(ymax) * 0.2 + 1e-3)
    add_pairwise_brackets(ax, stat_row, ymax + 0.18 * dy, dy)

    ax.axhline(0, color="black", lw=0.9)
    ax.set_xlim(-0.6, 2.6)
    ax.set_xticks([0, 1, 2])
    ax.set_xticklabels([CONDITION_LABEL["naive"], CONDITION_LABEL["CDM"], CONDITION_LABEL["C21"]], rotation=0, fontsize=8)
    ax.set_ylabel("z-score", fontsize=8)
    ax.set_title(spec["title"], fontsize=9, fontweight="bold")
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.tick_params(labelsize=8)


def add_band_guides(ax, freq_max):
    guides = [(4, "θ"), (12, "β"), (30, "low-γ"), (60, "high-γ")]
    for x, label in guides:
        if x <= freq_max:
            ax.axvline(x, ls=":", lw=0.8, color="0.55", zorder=0)
    labels = [(2.5, "δ"), (8, "θ"), (21, "β"), (45, "low-γ"), (70, "high-γ")]
    for x, label in labels:
        if x <= freq_max:
            ax.text(x, 1.01, label, transform=ax.get_xaxis_transform(), ha="center", va="bottom", fontsize=8)


def plot_curve_panel(ax, summary_df, value_col, x_col, title, ylabel, ylim=None, log_y=False):
    for cond_key in ["naive", "CDM", "C21"]:
        cond_data = summary_df[summary_df["condition_group"] == cond_key].copy()
        if len(cond_data) == 0:
            continue
        x = cond_data[x_col].values.astype(float)
        y = cond_data["mean"].values.astype(float)
        sem = cond_data["sem"].values.astype(float)
        order = np.argsort(x)
        x = x[order]
        y = y[order]
        sem = sem[order]
        if log_y:
            ax.semilogy(x, y, color=CONDITION_COLOR[cond_key], lw=1.8, label=CONDITION_LABEL[cond_key])
            ax.fill_between(x, np.clip(y - sem, 1e-20, None), y + sem, color=CONDITION_COLOR[cond_key], alpha=0.22)
        else:
            ax.plot(x, y, color=CONDITION_COLOR[cond_key], lw=1.8, label=CONDITION_LABEL[cond_key])
            ax.fill_between(x, y - sem, y + sem, color=CONDITION_COLOR[cond_key], alpha=0.22)

    add_band_guides(ax, float(np.nanmax(summary_df[x_col].values.astype(float))))
    ax.set_xlabel("Frequency (Hz)", fontsize=8)
    ax.set_ylabel(ylabel, fontsize=8)
    ax.set_title(title, fontsize=9, fontweight="bold")
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.tick_params(labelsize=8)
    if ylim is not None:
        ax.set_ylim(ylim)


def make_quantitative_figure(psd_summary, coh_summary, hab_stats, noe_stats, out_dir):
    fig, axes = plt.subplots(4, 3, figsize=(12.5, 14.5), constrained_layout=True)

    for j, region in enumerate(["PFC", "RE", "HC"]):
        d = psd_summary[psd_summary["region"] == region].copy()
        plot_curve_panel(
            axes[0, j],
            d,
            value_col="mean",
            x_col="frequency_hz",
            title=f"Power spectrum - {REGION_FIGURE_LABEL[region]}",
            ylabel="PSD (V²/Hz)",
            log_y=True,
        )
        axes[0, j].set_xlim(1, 80)
        axes[0, j].legend(frameon=False, fontsize=7)

    for j, pair_name in enumerate(["PFC-RE", "HC-RE", "HC-PFC"]):
        d = coh_summary[coh_summary["target"] == pair_name].copy()
        plot_curve_panel(
            axes[1, j],
            d,
            value_col="mean",
            x_col="frequency_hz",
            title=f"Coherence - {PAIR_FIGURE_LABEL[pair_name]}",
            ylabel="Coherence",
            log_y=False,
        )
        axes[1, j].set_xlim(1, 80)
        axes[1, j].set_ylim(0, 1)
        axes[1, j].legend(frameon=False, fontsize=7)

    for j, spec in enumerate(BAR_PANELS_RUNNING):
        plot_selected_bar_panel(axes[2, j], spec, hab_stats)

    for j, spec in enumerate(BAR_PANELS_NOE):
        plot_selected_bar_panel(axes[3, j], spec, noe_stats)

    panel_letters = [
        ["g", "h", "i"],
        ["j", "k", "l"],
        ["m", "n", "o"],
        ["p", "q", "r"],
    ]
    for i in range(4):
        for j in range(3):
            axes[i, j].text(-0.15, 1.06, panel_letters[i][j], transform=axes[i, j].transAxes, fontsize=12, fontweight="bold")

    for ext in ["png", "pdf", "svg"]:
        path = os.path.join(out_dir, f"Figure5_quantitative_panels.{ext}")
        fig.savefig(path, dpi=300 if ext == "png" else None, bbox_inches="tight")

    plt.close(fig)


def main():
    cfg = make_cfg()
    output_root = os.path.join(cfg["resultsDir"], "figure5_source_data")
    noe_dir = os.path.join(output_root, "NOE")
    hab_dir = os.path.join(output_root, "HAB_running")
    panel_dir = os.path.join(output_root, "panels")
    os.makedirs(noe_dir, exist_ok=True)
    os.makedirs(hab_dir, exist_ok=True)
    os.makedirs(panel_dir, exist_ok=True)

    noe_results_long, noe_session_log, noe_bout_qc = run_noe_extraction(cfg, noe_dir)
    noe_stats = compute_triplet_stats(
        results_long=noe_results_long,
        out_dir=noe_dir,
        prefix="NOE_delta_pair_coh_pac",
        epoch_keep=["delta"],
        domain_keep=["pair"],
        metric_keep=["coh", "pac_mi"],
        bands_keep=["theta", "beta", "lowGamma", "theta_lowGamma"],
        win_sec=0.5,
        pool_hemispheres=False,
        require_both_hemis=False,
        c21_conds=["C21_1"],
        normalize_method="zscore",
        apply_fdr=False,
        min_n_event_windows=2,
    )

    hab_running_results_long, hab_running_session_log = run_hab_running_band_coherence(cfg, hab_dir)
    hab_stats = compute_triplet_stats(
        results_long=hab_running_results_long,
        out_dir=hab_dir,
        prefix="HAB_running_pair_coherence",
        epoch_keep=["running"],
        domain_keep=["pair"],
        metric_keep=["coh"],
        bands_keep=["theta", "beta", "lowGamma"],
        win_sec=0.5,
        pool_hemispheres=False,
        require_both_hemis=False,
        c21_conds=["C21_1"],
        normalize_method="zscore",
        apply_fdr=False,
        min_n_event_windows=0,
    )

    psd_subject, psd_summary, coh_subject, coh_summary = run_hab_psd_and_coherence_spectra(cfg, hab_dir)

    export_selected_bar_panel_csvs(BAR_PANELS_RUNNING, hab_stats, panel_dir, "running")
    export_selected_bar_panel_csvs(BAR_PANELS_NOE, noe_stats, panel_dir, "noe")

    export_curve_panel_csvs("power_spectrum", psd_subject, psd_summary, panel_dir)
    export_curve_panel_csvs("coherence_spectrum", coh_subject, coh_summary, panel_dir)

    make_quantitative_figure(psd_summary, coh_summary, hab_stats, noe_stats, panel_dir)

    print(f"NOE rows: {len(noe_results_long)}")
    print(f"HAB running rows: {len(hab_running_results_long)}")
    print(f"PSD subject rows: {len(psd_subject)}")
    print(f"Coherence spectrum subject rows: {len(coh_subject)}")
    print(f"Output root: {output_root}")


if __name__ == "__main__":
    main()
