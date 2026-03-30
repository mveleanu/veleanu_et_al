
#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
TST fiberphotometry vs EthoVision Activity (Supplementary Fig. 4b-g)
"""

import os
import re
import json
import glob
import math
import warnings
from collections import defaultdict

import numpy as np
import pandas as pd

from scipy.stats import pearsonr, spearmanr, linregress, wilcoxon, ranksums, ttest_rel, ttest_ind

import matplotlib
import matplotlib.pyplot as plt


# -----------------------------
# Pick data from ethovision output
# -----------------------------

def _to_float(x):
    try:
        return float(str(x).replace(",", "."))
    except Exception:
        return np.nan

def _parse_event_time_any(s):
    """Parse a variety of time formats into seconds (float). Return np.nan if not parseable.
     examples:
      - "2.0, 46.0" or "2, 46" or "2;46" or "2 46"  -> 166.0
      - "2:46" or "02:46" or "0:02:46"            -> 166.0
      - "166", "166s", "166 sec"                  -> 166.0
      - "3,49" (ambiguous) -> 3.49 (seconds)  [single number is seconds]
    """
    if s is None:
        return np.nan
    ss = str(s).strip().strip('"').strip("'")
    if ss == "" or ss.lower() in ("nan", "none"):
        return np.nan

    # 1) minutes + seconds separated by comma/semicolon/space
    m = re.match(r"^\s*([+-]?\d+(?:[.,]\d+)?)\s*[,; ]\s*([+-]?\d+(?:[.,]\d+)?)\s*$", ss)
    if m:
        m_val = _to_float(m.group(1))
        s_val = _to_float(m.group(2))
        if np.isfinite(m_val) and np.isfinite(s_val):
            return m_val * 60.0 + s_val

    # 2) mm:ss or hh:mm:ss
    m = re.match(r"^\s*(\d+):(\d+)(?::(\d+(?:\.\d+)?))?\s*$", ss)
    if m:
        h = m.group(3) is not None and ss.count(":") == 2
        if h:
            hh = float(m.group(1))
            mm = float(m.group(2))
            ss_val = float(m.group(3))
            return hh * 3600.0 + mm * 60.0 + ss_val
        else:
            mm = float(m.group(1))
            ss_val = float(m.group(2))
            return mm * 60.0 + ss_val

    # 3) plain seconds with optional suffix
    m = re.match(r"^\s*([+-]?\d+(?:[.,]\d+)?)\s*(?:s|sec|secs|second|seconds)?\s*$", ss, re.I)
    if m:
        val = _to_float(m.group(1))
        if np.isfinite(val):
            return val

    return np.nan


def _find_header_row_with_events(raw, max_rows_to_scan=5):
    """Return (header_row_index, parseable_cols) where parseable_cols are columns with event timestamps."""
    nrows, ncols = raw.shape
    best = (-1, [])
    for r in range(min(max_rows_to_scan, nrows)):
        parseable = []
        for c in range(ncols):
            t = _parse_event_time_any(raw.iloc[r, c])
            if np.isfinite(t):
                parseable.append(c)
        if len(parseable) > len(best[1]):
            best = (r, parseable)
    return best  # may be (-1, [])


def _find_rel_time_column(raw_after_header):
    nrows, ncols = raw_after_header.shape
    candidates = []
    for c in range(ncols):
        col = pd.to_numeric(raw_after_header.iloc[:, c], errors="coerce")
        if col.notna().sum() < max(10, int(0.1 * nrows)):
            continue
        arr = col.to_numpy()
        arr = arr[np.isfinite(arr)]
        if arr.size < 10:
            continue
        diffs = np.diff(arr)
        # tolerate tiny noise
        if np.all(diffs >= -1e-6):
            span = float(np.nanmax(arr) - np.nanmin(arr))
            candidates.append((c, span, arr.size))
    if candidates:
        candidates.sort(key=lambda x: (x[1], x[2]), reverse=True)
        return candidates[0][0]
    return 0


# -----------------------------
# EthoVision loader 
# -----------------------------

def _find_header_row_for_ethovision(xlsx_path, search_limit=100):
    raw = pd.read_excel(xlsx_path, sheet_name=0, header=None, nrows=search_limit)
    for i in range(min(search_limit, len(raw))):
        row_vals = raw.iloc[i].astype(str).str.lower().tolist()
        if any("recording time" in v for v in row_vals) and any("activity" in v.strip() for v in row_vals):
            return i
    return 32


def load_ethovision_activity(xlsx_path):
    hdr = _find_header_row_for_ethovision(xlsx_path)
    df = pd.read_excel(xlsx_path, sheet_name=0, header=hdr).dropna(how="all")
    # Normalize column names
    cols_norm = {c: str(c).strip().lower() for c in df.columns}
    inv = {v: k for k, v in cols_norm.items()}
    # Accept 'activity' variants (e.g., 'activity (%)')
    act_key = next((k for k, v in cols_norm.items() if v.startswith("activity")), None)
    time_key = next((k for k, v in cols_norm.items() if "recording time" in v), None)
    if time_key is None or act_key is None:
        raise ValueError(f"EthoVision: couldn't find 'Recording time' and 'Activity' columns in: {xlsx_path}")
    def safe(x):
        try:
            return float(str(x).replace(",", "."))
        except Exception:
            return np.nan
    out = pd.DataFrame({
        "recording_time_s": df[time_key].map(safe),
        "activity_pct": df[act_key].map(safe)
    })
    out = out.dropna(subset=["recording_time_s"]).sort_values("recording_time_s").reset_index(drop=True)
    return out


# -----------------------------
# PSTH loader 
# -----------------------------

def load_psth_csv(psth_path, verbose=True):
    """Load PSTH where a row contains event timestamps and a column contains the relative-time vector.
    Returns: rel_time (N,), fiber_mat (N,E), event_times_sec (list of E floats).
    """
    try:
        raw = pd.read_csv(psth_path, header=None, engine="python")
    except Exception:
        raw = pd.read_csv(psth_path, header=None)  # fallback

    nrows, ncols = raw.shape
    if nrows < 2 or ncols < 2:
        raise ValueError(f"PSTH file looks malformed or has too few columns/rows: {psth_path} (shape {raw.shape})")

    hdr_row, parseable_cols = _find_header_row_with_events(raw, max_rows_to_scan=5)
    if hdr_row < 0 or len(parseable_cols) == 0:
        # As an extra fallback: assume row 0 holds events in cols 1..
        hdr_row = 0
        parseable_cols = list(range(1, ncols))

    # Build event times list, skipping non-parsable columns
    event_cols = []
    event_times_sec = []
    for c in parseable_cols:
        t = _parse_event_time_any(raw.iloc[hdr_row, c])
        if np.isfinite(t):
            event_cols.append(c)
            event_times_sec.append(float(t))

    # Slice data rows below the header
    data_rows = raw.iloc[hdr_row+1:, :].copy()

    # Detect relative-time column
    rel_col = _find_rel_time_column(data_rows)

    rel_time = pd.to_numeric(data_rows.iloc[:, rel_col], errors="coerce").to_numpy()

    # Exclude the rel_time column from event columns if present
    event_cols = [c for c in event_cols if c != rel_col]

    # Build fiber matrix
    fiber_df = data_rows.iloc[:, event_cols].apply(pd.to_numeric, errors="coerce")
    # Align by valid rel_time
    valid_mask = np.isfinite(rel_time)
    rel_time = rel_time[valid_mask]
    fiber_df = fiber_df[valid_mask]

    # If lengths mismatch (shouldn't), trim to min length
    if fiber_df.shape[0] != rel_time.shape[0]:
        m = min(fiber_df.shape[0], rel_time.shape[0])
        rel_time = rel_time[:m]
        fiber_df = fiber_df.iloc[:m, :]

    if len(event_cols) == 0 or fiber_df.shape[1] == 0:
        raise ValueError(
            "PSTH: No event columns were parsed.\n"
            f"- header_row={hdr_row}, parseable_in_header={len(parseable_cols)}\n"
            f"- Try inspecting the first row of your file to see how timestamps are written."
        )

    if verbose:
        print(f"[PSTH] Detected header row: {hdr_row}")
        print(f"[PSTH] Detected rel_time column: {rel_col} (min={np.nanmin(rel_time):.3f}, max={np.nanmax(rel_time):.3f}, n={len(rel_time)})")
        print(f"[PSTH] Parsed {len(event_cols)} event columns. First few event times (s): {event_times_sec[:6]}")

    fiber_mat = fiber_df.to_numpy()
    return rel_time, fiber_mat, event_times_sec


# -----------------------------
# Interpolation & plotting
# -----------------------------

def interpolate_activity(etho_df, abs_times):
    t = etho_df["recording_time_s"].to_numpy()
    y = etho_df["activity_pct"].to_numpy()
    times = np.asarray(abs_times, dtype=float)
    times = np.clip(times, np.nanmin(t), np.nanmax(t))
    return np.interp(times, t, y)

def resample_to_grid(x, y, grid):
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)
    grid = np.asarray(grid, dtype=float)
    return np.interp(grid, x, y)

def compute_event_stats(df_evt):
    x = df_evt["ethovision_activity_pct"].to_numpy()
    y = df_evt["fiber_signal"].to_numpy()
    pr, pp = pearsonr(y, x)
    sr, sp = spearmanr(y, x)
    slope, intercept, r_value, p_value, std_err = linregress(x, y)
    f_idx = int(np.nanargmax(y))
    a_idx = int(np.nanargmax(x))
    return {
        "n_points": int(len(df_evt)),
        "pearson_r": float(pr), "pearson_p": float(pp),
        "spearman_rho": float(sr), "spearman_p": float(sp),
        "linear_slope_fiber_vs_activity": float(slope),
        "linear_intercept": float(intercept),
        "linear_r": float(r_value),
        "linear_p": float(p_value),
        "fiber_mean": float(np.nanmean(y)),
        "fiber_std": float(np.nanstd(y, ddof=1)),
        "activity_mean_pct": float(np.nanmean(x)),
        "activity_std_pct": float(np.nanstd(x, ddof=1)),
        "fiber_peak_value": float(y[f_idx]),
        "fiber_peak_rel_time_s": float(df_evt["rel_time_s"].iloc[f_idx]),
        "activity_peak_pct": float(x[a_idx]),
        "activity_peak_rel_time_s": float(df_evt["rel_time_s"].iloc[a_idx]),
    }

def plot_event_overlay(df_evt, title, out_png):
    plt.figure()
    ax1 = plt.gca()
    ax1.plot(df_evt["rel_time_s"], df_evt["fiber_signal"], label="Fiber signal")
    ax1.axvline(0, linestyle="--")
    ax1.set_xlabel("Time relative to event (s)")
    ax1.set_ylabel("Fiber signal (a.u.)")
    ax2 = ax1.twinx()
    ax2.plot(df_evt["rel_time_s"], df_evt["ethovision_activity_pct"], label="EthoVision activity (%)")
    ax2.set_ylabel("Activity (%)")
    plt.title(title)
    plt.tight_layout()
    plt.savefig(out_png, dpi=200)
    plt.close()

def plot_event_scatter(df_evt, title, out_png, pearson_r=None, pearson_p=None):
    plt.figure()
    plt.scatter(df_evt["ethovision_activity_pct"], df_evt["fiber_signal"])
    plt.xlabel("EthoVision activity (%)")
    plt.ylabel("Fiber signal (a.u.)")
    if pearson_r is not None and pearson_p is not None:
        plt.title(f"{title}\nPearson r={pearson_r:.3f}, p={pearson_p:.3g}")
    else:
        plt.title(title)
    plt.tight_layout()
    plt.savefig(out_png, dpi=200)
    plt.close()

def plot_mean_overlay_with_shading(rel_grid, fiber_mean, fiber_sem, act_mean, act_sem, title, out_png):
    rel_grid = np.asarray(rel_grid, dtype=float)
    plt.figure()
    ax1 = plt.gca()
    ax1.plot(rel_grid, fiber_mean, label="Fiber mean")
    ax1.fill_between(rel_grid, fiber_mean - fiber_sem, fiber_mean + fiber_sem, alpha=0.25)
    ax1.axvline(0, linestyle="--")
    ax1.set_xlabel("Time relative to event (s)")
    ax1.set_ylabel("Fiber signal (a.u.)")
    ax2 = ax1.twinx()
    ax2.plot(rel_grid, act_mean, label="Activity mean (%)")
    ax2.fill_between(rel_grid, act_mean - act_sem, act_mean + act_sem, alpha=0.25)
    ax2.set_ylabel("Activity (%)")
    plt.title(title)
    plt.tight_layout()
    plt.savefig(out_png, dpi=200)
    plt.close()


# -----------------------------
#  analysis 
# -----------------------------

def _normalize_condition(token, condition_aliases):
    if token is None:
        return None
    t = token.lower().replace("_","").replace("-","")
    for k, v in condition_aliases.items():
        kk = k.lower().replace("_","").replace("-","")
        if kk in t or t in kk:
            return v
    if "cdm" in t: return "CDM"
    if "c21" in t: return "C21"
    return token

def discover_pairs(input_dir, psth_glob, ethovision_glob, mouse_regex, condition_aliases, manifest_csv=None):
    pairs = []
    if manifest_csv is not None:
        man = pd.read_csv(manifest_csv)
        required = {"mouse_id", "condition", "psth_path", "ethovision_path"}
        if not required.issubset(set(man.columns)):
            raise ValueError(f"Manifest must have columns: {required}")
        man = man.copy()
        man["condition_norm"] = man["condition"].apply(lambda c: _normalize_condition(str(c), condition_aliases) or str(c))
        for _, row in man.iterrows():
            pairs.append({
                "mouse_id": str(row["mouse_id"]),
                "condition": str(row["condition_norm"]),
                "psth_path": str(row["psth_path"]),
                "ethovision_path": str(row["ethovision_path"]),
                "session_id": str(row.get("session_id", "")),
                "source": "manifest"
            })
        return pairs
   
    return pairs

def analyze_pair(pair_dict, output_dir, window_s=5.0, global_rel_grid=None, skip_edge_events=False):
    mouse = pair_dict.get("mouse_id", "")
    cond  = pair_dict.get("condition", "")
    psth_path = pair_dict["psth_path"]
    ethopath  = pair_dict["ethovision_path"]

    etho = load_ethovision_activity(ethopath)
    rel_time, fiber_mat, event_times = load_psth_csv(psth_path, verbose=True)

    etho_tmin, etho_tmax = float(np.nanmin(etho["recording_time_s"])), float(np.nanmax(etho["recording_time_s"]))
    valid_event_mask = np.ones(len(event_times), dtype=bool)
    if skip_edge_events:
        for i, et in enumerate(event_times):
            if not np.isfinite(et):
                valid_event_mask[i] = False
                continue
            if (et - window_s) < etho_tmin or (et + window_s) > etho_tmax:
                valid_event_mask[i] = False

    tag = f"mouse_{mouse}_cond_{cond}".strip("_").replace(" ", "")
    pair_out = os.path.join(output_dir, "per_pair", tag)
    os.makedirs(pair_out, exist_ok=True)

    # Use a canonical grid (first PSTH grid passed by the caller), otherwise the file's own
    rel_grid = np.asarray(global_rel_grid if global_rel_grid is not None else rel_time, dtype=float)

    eventwise_summary = []
    all_evt_traces_fiber, all_evt_traces_act = [], []

    n_events_used = 0

    for col_idx, evt_time in enumerate(event_times):
        if not np.isfinite(evt_time):
            continue
        if not valid_event_mask[col_idx]:
            continue
        n_events_used += 1

        # Compose per-event DF
        fiber_vec = fiber_mat[:, col_idx]
        # If a canonical grid was provided, resample
        if global_rel_grid is not None and not np.array_equal(np.round(rel_time, 6), np.round(rel_grid, 6)):
            fiber_vec = resample_to_grid(rel_time, fiber_vec, rel_grid)
            abs_times = rel_grid + float(evt_time)
        else:
            abs_times = rel_time + float(evt_time)

        act_vec = interpolate_activity(etho, abs_times)

        df_evt = pd.DataFrame({
            "event_time_s": float(evt_time),
            "rel_time_s": rel_grid if global_rel_grid is not None else rel_time,
            "abs_time_s": abs_times,
            "fiber_signal": fiber_vec,
            "ethovision_activity_pct": act_vec
        })

        stats = compute_event_stats(df_evt)

        evt_lbl = f"{int(round(evt_time))}s" if abs(evt_time - round(evt_time)) < 1e-6 else f"{evt_time:.2f}s"
        csv_evt = os.path.join(pair_out, f"event_window_{evt_lbl}.csv")
        df_evt.to_csv(csv_evt, index=False)

        overlay_png = os.path.join(pair_out, f"overlay_{evt_lbl}.png")
        plot_event_overlay(df_evt, f"Mouse {mouse} | {cond} | Event {evt_lbl} (±{window_s}s)", overlay_png)

        scatter_png = os.path.join(pair_out, f"scatter_{evt_lbl}.png")
        plot_event_scatter(df_evt, f"Mouse {mouse} | {cond} | Event {evt_lbl}", scatter_png,
                           pearson_r=stats["pearson_r"], pearson_p=stats["pearson_p"])

        row = {"mouse_id": mouse, "condition": cond, "psth_path": psth_path, "ethovision_path": ethopath,
               "event_time_s": float(evt_time), "csv": csv_evt, "overlay_png": overlay_png, "scatter_png": scatter_png}
        row.update(stats)
        eventwise_summary.append(row)

        all_evt_traces_fiber.append(df_evt["fiber_signal"].to_numpy())
        all_evt_traces_act.append(df_evt["ethovision_activity_pct"].to_numpy())

    if n_events_used == 0:
        warnings.warn(f"No usable events found in {psth_path}. Check the header row and timestamp format.")

    eventwise_df = pd.DataFrame(eventwise_summary)
    eventwise_csv = os.path.join(pair_out, "eventwise_summary.csv")
    eventwise_df.to_csv(eventwise_csv, index=False)

    per_mouse = {}
    if len(all_evt_traces_fiber) > 0:
        fiber_stack = np.vstack(all_evt_traces_fiber)
        act_stack = np.vstack(all_evt_traces_act)
        fiber_mean = np.nanmean(fiber_stack, axis=0)
        fiber_sem = np.nanstd(fiber_stack, axis=0, ddof=1) / math.sqrt(max(1, fiber_stack.shape[0]))
        act_mean = np.nanmean(act_stack, axis=0)
        act_sem = np.nanstd(act_stack, axis=0, ddof=1) / math.sqrt(max(1, act_stack.shape[0]))

        rel_for_plot = rel_grid

        mouse_mean_png = os.path.join(pair_out, "mouse_mean_overlay.png")
        plot_mean_overlay_with_shading(rel_for_plot, fiber_mean, fiber_sem, act_mean, act_sem,
            f"Mouse {mouse} | {cond} — mean across {len(all_evt_traces_fiber)} events (±SEM)",
            mouse_mean_png)

        mouse_mean_csv = os.path.join(pair_out, "mouse_mean_traces.csv")
        pd.DataFrame({
            "rel_time_s": rel_for_plot,
            "fiber_mean": fiber_mean, "fiber_sem": fiber_sem,
            "activity_mean_pct": act_mean, "activity_sem_pct": act_sem
        }).to_csv(mouse_mean_csv, index=False)

        per_mouse = {
            "rel_time_s": rel_for_plot.tolist(),
            "fiber_mean": fiber_mean.tolist(),
            "fiber_sem": fiber_sem.tolist(),
            "activity_mean_pct": act_mean.tolist(),
            "activity_sem_pct": act_sem.tolist(),
            "mouse_mean_png": mouse_mean_png,
            "mouse_mean_csv": mouse_mean_csv,
            "n_events": int(len(all_evt_traces_fiber))
        }

    return {
        "pair_outdir": pair_out,
        "eventwise_csv": eventwise_csv,
        "eventwise_df": eventwise_df,
        "mouse_mean": per_mouse,
        "rel_grid": rel_grid
    }


def aggregate_by_condition(per_mouse_summaries, output_dir):
    from math import sqrt
    results = {}
    from collections import defaultdict
    cond_groups = defaultdict(list)
    for row in per_mouse_summaries:
        cond_groups[row["condition"]].append(row)
    for cond, rows in cond_groups.items():
        base_grid = np.asarray(rows[0]["rel_time_s"], dtype=float)
        fiber_means, act_means = [], []
        for r in rows:
            rg = np.asarray(r["rel_time_s"], dtype=float)
            if not np.array_equal(np.round(rg, 6), np.round(base_grid, 6)):
                fm = resample_to_grid(rg, np.asarray(r["fiber_mean"]), base_grid)
                am = resample_to_grid(rg, np.asarray(r["activity_mean_pct"]), base_grid)
            else:
                fm = np.asarray(r["fiber_mean"], dtype=float)
                am = np.asarray(r["activity_mean_pct"], dtype=float)
            fiber_means.append(fm)
            act_means.append(am)
        fiber_stack = np.vstack(fiber_means)
        act_stack = np.vstack(act_means)
        fiber_mean = np.nanmean(fiber_stack, axis=0)
        fiber_sem = np.nanstd(fiber_stack, axis=0, ddof=1) / math.sqrt(max(1, fiber_stack.shape[0]))
        act_mean = np.nanmean(act_stack, axis=0)
        act_sem = np.nanstd(act_stack, axis=0, ddof=1) / math.sqrt(max(1, act_stack.shape[0]))

        group_dir = os.path.join(output_dir, "per_group", cond or "UNKNOWN")
        os.makedirs(group_dir, exist_ok=True)
        group_png = os.path.join(group_dir, f"group_mean_overlay_{cond or 'UNKNOWN'}.png")
        plot_mean_overlay_with_shading(base_grid, fiber_mean, fiber_sem, act_mean, act_sem,
            f"Group mean (±SEM across mice) — {cond}",
            group_png)
        group_csv = os.path.join(group_dir, f"group_mean_traces_{cond or 'UNKNOWN'}.csv")
        pd.DataFrame({
            "rel_time_s": base_grid,
            "fiber_group_mean": fiber_mean, "fiber_group_sem": fiber_sem,
            "activity_group_mean_pct": act_mean, "activity_group_sem_pct": act_sem
        }).to_csv(group_csv, index=False)

        results[cond] = {
            "group_png": group_png,
            "group_csv": group_csv,
            "n_mice": int(fiber_stack.shape[0]),
            "rel_time_s": base_grid.tolist(),
            "fiber_group_mean": fiber_mean.tolist(),
            "fiber_group_sem": fiber_sem.tolist(),
            "activity_group_mean_pct": act_mean.tolist(),
            "activity_group_sem_pct": act_sem.tolist()
        }
    return results


def compare_activity_peaks(eventwise_all_df, output_dir):
    comp_dir = os.path.join(output_dir, "condition_comparison")
    os.makedirs(comp_dir, exist_ok=True)

    ev_cols = ["mouse_id", "condition", "event_time_s", "activity_peak_pct", "fiber_peak_value"]
    ev_df = eventwise_all_df.loc[:, [c for c in ev_cols if c in eventwise_all_df.columns]].copy()
    ev_csv = os.path.join(comp_dir, "peak_activity_per_event.csv")
    ev_df.to_csv(ev_csv, index=False)

    mouse_agg = (ev_df.groupby(["mouse_id", "condition"], as_index=False)
                    .agg(activity_peak_pct_mean=("activity_peak_pct", "mean"),
                         activity_peak_pct_median=("activity_peak_pct", "median"),
                         n_events=("activity_peak_pct", "size")))
    mouse_csv = os.path.join(comp_dir, "peak_activity_per_mouse.csv")
    mouse_agg.to_csv(mouse_csv, index=False)

    stats = {"conditions": sorted(ev_df["condition"].dropna().unique().tolist())}

    if len(stats["conditions"]) >= 2:
        c1, c2 = stats["conditions"][:2]
        x = ev_df.loc[ev_df["condition"] == c1, "activity_peak_pct"].to_numpy()
        y = ev_df.loc[ev_df["condition"] == c2, "activity_peak_pct"].to_numpy()
        try: u_stat, u_p = ranksums(x, y)
        except Exception: u_stat, u_p = np.nan, np.nan
        try: t_stat, t_p = ttest_ind(x, y, equal_var=False)
        except Exception: t_stat, t_p = np.nan, np.nan
        stats["event_level"] = {
            "conds": [c1, c2],
            "n_events": [int(np.isfinite(x).sum()), int(np.isfinite(y).sum())],
            "ranksums_stat": float(u_stat), "ranksums_p": float(u_p),
            "ttest_ind_stat": float(t_stat), "ttest_ind_p": float(t_p),
            "mean_activity_peak": [float(np.nanmean(x)), float(np.nanmean(y))],
            "median_activity_peak": [float(np.nanmedian(x)), float(np.nanmedian(y))]
        }

    mice_both = (mouse_agg.groupby("mouse_id").filter(lambda df: df["condition"].nunique() >= 2)["mouse_id"].unique())
    if len(mice_both) > 0 and len(stats["conditions"]) >= 2:
        c1, c2 = stats["conditions"][:2]
        mx, my = [], []
        for m in mice_both:
            row_c1 = mouse_agg[(mouse_agg["mouse_id"] == m) & (mouse_agg["condition"] == c1)]
            row_c2 = mouse_agg[(mouse_agg["mouse_id"] == m) & (mouse_agg["condition"] == c2)]
            if len(row_c1) == 1 and len(row_c2) == 1:
                mx.append(float(row_c1["activity_peak_pct_mean"].values[0]))
                my.append(float(row_c2["activity_peak_pct_mean"].values[0]))
        mx = np.asarray(mx, dtype=float)
        my = np.asarray(my, dtype=float)
        try: w_stat, w_p = wilcoxon(mx, my, zero_method="wilcox", alternative="two-sided")
        except Exception: w_stat, w_p = np.nan, np.nan
        try: t_stat, t_p = ttest_rel(mx, my)
        except Exception: t_stat, t_p = np.nan, np.nan
        stats["paired_mouse_level"] = {
            "conds": [c1, c2],
            "n_mice": int(len(mx)),
            "wilcoxon_stat": float(w_stat), "wilcoxon_p": float(w_p),
            "ttest_rel_stat": float(t_stat), "ttest_rel_p": float(t_p),
            "mean_activity_peak": [float(np.nanmean(mx)), float(np.nanmean(my))],
            "median_activity_peak": [float(np.nanmedian(mx)), float(np.nanmedian(my))]
        }

    stats_json = os.path.join(comp_dir, "peak_activity_stats.json")
    with open(stats_json, "w", encoding="utf-8") as f:
        json.dump(stats, f, indent=2)

    # Simple visuals
    if len(stats.get("conditions", [])) >= 2:
        c1, c2 = stats["conditions"][:2]
        plt.figure()
        data = [ev_df.loc[ev_df["condition"] == c1, "activity_peak_pct"].to_numpy(),
                ev_df.loc[ev_df["condition"] == c2, "activity_peak_pct"].to_numpy()]
        plt.boxplot(data, labels=[c1, c2], showfliers=False)
        plt.ylabel("Activity peak within window (%)")
        plt.title("Event-level Activity peaks by condition")
        plt.tight_layout()
        plt.savefig(os.path.join(comp_dir, "box_event_level_activity_peak.png"), dpi=200)
        plt.close()

    return {
        "event_level_csv": ev_csv,
        "mouse_level_csv": mouse_csv,
        "stats_json": stats_json
    }


def run_pipeline(
    input_dir,
    output_dir,
    manifest_csv=None,
    window_s=5.0,
    condition_aliases=None,
    mouse_regex=r"(?<!\d)(\d{2,4})(?!\d)",
    psth_glob="**/*.csv",
    ethovision_glob="**/*.xlsx",
    skip_edge_events=False
):
    os.makedirs(output_dir, exist_ok=True)
    if condition_aliases is None:
        condition_aliases = {"post-cdm":"CDM", "cdm":"CDM", "postc21":"C21", "post-c21":"C21", "c21":"C21"}

    pairs = discover_pairs(input_dir, psth_glob, ethovision_glob, mouse_regex, condition_aliases, manifest_csv=manifest_csv)
    if not pairs:
        raise RuntimeError("No (PSTH, EthoVision) pairs discovered. Check input_dir or provide a manifest.")

    # Detect canonical rel_time grid from first usable PSTH
    first_rel_grid = None
    for pr in pairs:
        try:
            rel_time, _, _ = load_psth_csv(pr["psth_path"], verbose=False)
            first_rel_grid = rel_time
            break
        except Exception as e:
            warnings.warn(f"Skipping {pr['psth_path']} for grid detection: {e}")
    if first_rel_grid is None:
        raise RuntimeError("Failed to detect a canonical relative-time grid from any PSTH.")

    all_eventwise = []
    per_mouse_summaries = []

    for pr in pairs:
        try:
            res = analyze_pair(pr, output_dir=output_dir, window_s=window_s,
                               global_rel_grid=first_rel_grid, skip_edge_events=skip_edge_events)
        except Exception as e:
            warnings.warn(f"Error analyzing pair {pr}: {e}")
            continue

        df = res["eventwise_df"].copy()
        df["mouse_id"] = pr.get("mouse_id", "")
        df["condition"] = pr.get("condition", "")
        df["psth_path"] = pr.get("psth_path", "")
        df["ethovision_path"] = pr.get("ethovision_path", "")
        all_eventwise.append(df)

        if res["mouse_mean"]:
            per_mouse_summaries.append({
                "mouse_id": pr.get("mouse_id", ""),
                "condition": pr.get("condition", ""),
                "rel_time_s": res["mouse_mean"]["rel_time_s"],
                "fiber_mean": res["mouse_mean"]["fiber_mean"],
                "activity_mean_pct": res["mouse_mean"]["activity_mean_pct"],
                "n_events": res["mouse_mean"]["n_events"],
                "mouse_mean_png": res["mouse_mean"]["mouse_mean_png"],
                "mouse_mean_csv": res["mouse_mean"]["mouse_mean_csv"]
            })

    eventwise_all_df = pd.concat(all_eventwise, ignore_index=True) if all_eventwise else pd.DataFrame()
    eventwise_all_csv = os.path.join(output_dir, "ALL_eventwise_summary.csv")
    eventwise_all_df.to_csv(eventwise_all_csv, index=False)

    group_results = aggregate_by_condition(per_mouse_summaries, output_dir=output_dir) if per_mouse_summaries else {}
    comp_results = compare_activity_peaks(eventwise_all_df, output_dir=output_dir) if not eventwise_all_df.empty else {}

    return {
        "n_pairs": int(len(pairs)),
        "pairs_source": [p.get("source","") for p in pairs],
        "eventwise_all_csv": eventwise_all_csv,
        "group_results": group_results,
        "condition_comparison": comp_results,
        "output_dir": output_dir
    }


# -----------------------------
# Debug helpers 
# -----------------------------

def debug_psth(psth_path):
    """Print a quick summary of what the loader sees in your PSTH file."""
    try:
        rel_time, fiber_mat, event_times = load_psth_csv(psth_path, verbose=True)
        print(f"OK: rel_time [{np.nanmin(rel_time):.3f}, {np.nanmax(rel_time):.3f}] (n={len(rel_time)})")
        print(f"OK: fiber_mat shape = {fiber_mat.shape}")
        print(f"OK: first 10 event times (s) = {event_times[:10]}")
    except Exception as e:
        print("PSTH DEBUG ERROR:", e)

def debug_ethovision(xlsx_path):
    """Print a quick summary of EthoVision activity columns."""
    try:
        etho = load_ethovision_activity(xlsx_path)
        print(f"OK: EthoVision timespan = [{np.nanmin(etho['recording_time_s']):.3f}, {np.nanmax(etho['recording_time_s']):.3f}] (n={len(etho)})")
        print(f"OK: Activity min/max = [{np.nanmin(etho['activity_pct']):.3f}, {np.nanmax(etho['activity_pct']):.3f}]")
    except Exception as e:
        print("ETHOVISION DEBUG ERROR:", e)


if __name__ == "__main__":
    import argparse
    ap = argparse.ArgumentParser(description="Run TST fiber vs EthoVision multi-pair analysis (v2).")
    ap.add_argument("--input_dir", required=True, help="Root folder with data (or use --manifest_csv).")
    ap.add_argument("--output_dir", required=True, help="Where to write outputs.")
    ap.add_argument("--manifest_csv", default=None, help="Optional manifest CSV path.")
    ap.add_argument("--window_s", type=float, default=5.0, help="Half-window (s), default 5.0.")
    ap.add_argument("--skip_edge_events", action="store_true", help="Skip events too close to EthoVision bounds.")
    args = ap.parse_args()
    results = run_pipeline(
        input_dir=args.input_dir,
        output_dir=args.output_dir,
        manifest_csv=args.manifest_csv,
        window_s=args.window_s,
        skip_edge_events=args.skip_edge_events
    )
    print(json.dumps(results, indent=2))
