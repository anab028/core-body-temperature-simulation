"""
Mechanism & robustness of the workload-swing benefit (ninth review round).

Builds on run_ambiguity.py (same protocols, windows, acceptance). Four parts:
  1. Truly paired robustness: per-setting, participants with an ensemble in BOTH arms; per-participant
     error differences; counts improved / worsened; availability transitions; nested bank subsets from
     one 8000 bank (size effect) vs independent 4000 banks (bank randomness) vs new participant sets
     (participant variability). Participant rows saved.
  2. Participant-level mechanism (sweat_only truth, standard vs work_only): same candidate IDs tracked
     across arms -> kept / lost / gained; are the LOST candidates the wrong ones? which window rejects
     them? what do their sweat/blood-flow gains, evaporation and core->skin transfer look like?
  3. Same for the secondary contrast (perfusion_only, standard vs fan_only) in aggregate only.
  4. Noise beyond ideal white noise: session bias, AR(1)-correlated noise, both.
All synthetic. Candidate bank = reference laws.
"""
import os, json
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from thermo import Body, simulate, make_env

OUT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "out"); os.makedirs(OUT, exist_ok=True)
BLUE, ORANGE, AQUA, YELLOW, INK, MUTED = "#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#0b0b0b", "#52514e"
plt.rcParams.update({"font.size": 9, "axes.spines.top": False, "axes.spines.right": False, "figure.dpi": 130})

body = Body(); DT = 2.0; T_CR0 = 36.9; CAM = 0.3; TOL = 1.5; N_P = 20
GAIN_PRIOR = lambda r, m: {"sweat": r.lognormal(0, 0.45, m), "bloodflow": r.lognormal(0, 0.3, m), "hc": r.lognormal(0, 0.2, m)}
SPM = int(60 / DT); WIN_MIN = [0, 22, 27, 32, 37, 40, 55]
WINDOWS = [(WIN_MIN[i] * SPM, WIN_MIN[i + 1] * SPM) for i in range(len(WIN_MIN) - 1)]
WIN_LAB = [f"{WIN_MIN[i]}–{WIN_MIN[i+1]} min" for i in range(len(WIN_MIN) - 1)]
I_END = 40 * SPM; T_LEN = 55 * SPM


def protocol_and_env(arm):
    rest, ex, rec = 600, 1800, 900
    t = np.arange(0, rest + ex + rec, DT); v = np.full_like(t, 0.15, float)
    P = np.zeros_like(t); in_ex = (t >= rest) & (t < rest + ex); P[in_ex] = 100.0
    if arm in ("fan_only", "combined"):
        v[(t >= 22 * 60) & (t < 27 * 60)] = 1.5
    if arm in ("work_only", "combined"):
        P[(t >= 27 * 60) & (t < 32 * 60)] = 50.0; P[(t >= 32 * 60) & (t < 37 * 60)] = 150.0
    M = np.full_like(t, 58.0, float); W = np.zeros_like(t)
    M[in_ex] = 58.0 + P[in_ex] / 0.20 / body.A_D; W[in_ex] = P[in_ex] / body.A_D
    rec_mask = t >= rest + ex; M[rec_mask] = 58.0 + (M[in_ex][-1] - 58.0) * np.exp(-(t[rec_mask] - (rest + ex)) / 180.0)
    env = make_env(t); env["v"] = v
    return {"t": t, "M": M, "W": W, "dt": DT}, env


def window_rms(cand_T_sk, T_obs):
    return np.stack([np.sqrt(np.mean((cand_T_sk[:, a:b] - T_obs[None, a:b]) ** 2, axis=1)) / CAM for a, b in WINDOWS], 1)  # (n, 6)


def ens_row(d_end, acc, truth):
    n = int(acc.sum())
    if n == 0:
        return {"n": 0, "status": "none"}
    if n < 20:
        return {"n": n, "status": "insufficient"}
    lo, hi = np.percentile(d_end[acc], [2.5, 97.5])
    return {"n": n, "status": "available", "median_error": float(np.median(d_end[acc]) - truth),
            "range95": [float(lo), float(hi)], "truth_outside_range95": bool(not (lo <= truth <= hi))}


def paired_compare(rows_a, rows_b):
    """rows_a = contrast arm, rows_b = standard; same participants in the same order."""
    common = [(a, b) for a, b in zip(rows_a, rows_b) if a["status"] == "available" and b["status"] == "available"]
    d = [abs(a["median_error"]) - abs(b["median_error"]) for a, b in common]
    return {"n_common_available": len(common),
            "n_improved": int(sum(x < 0 for x in d)), "n_worsened": int(sum(x > 0 for x in d)),
            "mean_delta_abs_error_common": float(np.mean(d)) if d else None,
            "median_delta_abs_error_common": float(np.median(d)) if d else None,
            "outside_to_inside": int(sum(b.get("truth_outside_range95", False) and not a.get("truth_outside_range95", True) for a, b in common)),
            "inside_to_outside": int(sum(a.get("truth_outside_range95", False) and not b.get("truth_outside_range95", True) for a, b in common)),
            "available_to_unavailable": int(sum(b["status"] == "available" and a["status"] != "available" for a, b in zip(rows_a, rows_b))),
            "unavailable_to_available": int(sum(a["status"] == "available" and b["status"] != "available" for a, b in zip(rows_a, rows_b))),
            "n_truth_outside": [int(sum(x.get("truth_outside_range95", False) for x in rows_a)), int(sum(x.get("truth_outside_range95", False) for x in rows_b))]}


def make_noise(r, kind, n):
    """kind: white | bias | ar1 | bias+ar1 ; all with marginal SD ~0.3."""
    w = r.normal(0, CAM, (n, T_LEN))
    if "ar1" in kind:
        rho = np.exp(-DT / 60.0)                                   # ~1-min correlation time
        e = np.zeros_like(w); e[:, 0] = w[:, 0]
        for i in range(1, T_LEN):
            e[:, i] = rho * e[:, i - 1] + np.sqrt(1 - rho ** 2) * w[:, i]
        w = e
    if "bias" in kind:
        w = w + r.normal(0, 0.5, (n, 1))
    return w


def evaluate(bank, tr, noise, tol=TOL):
    """rows for N participants given a bank sim and truth sim."""
    d_end = (bank["T_cr"][:, I_END] - bank["T_cr"][:, 0]).astype(float); rows = []; accs = []
    for i in range(tr["T_sk"].shape[0]):
        sc = window_rms(bank["T_sk"], tr["T_sk"][i] + noise[i]).max(1); acc = sc < tol; accs.append(acc)
        rows.append({"participant": i, "truth": float(tr["T_cr"][i, I_END] - T_CR0), **ens_row(d_end, acc, float(tr["T_cr"][i, I_END] - T_CR0))})
    return rows, accs


out = {"design": {"tol": TOL, "windows_min": WIN_MIN, "n_participants": N_P, "acceptance": "same candidate, max over 6 windows of RMS/noise < tol"}}
CONTRASTS = [("sweat_only", "work_only"), ("perfusion_only", "fan_only")]

# ====================================================== 1. paired robustness
# participants + noise fixed (set P1); one big bank B8000 with nested subsets; 3 independent 4000 banks; 2 new participant sets
r = np.random.default_rng(2024)
P1 = GAIN_PRIOR(r, N_P); N1 = make_noise(r, "white", N_P)
big_prior = GAIN_PRIOR(r, 8000)
ind_priors = [GAIN_PRIOR(np.random.default_rng(7000 + k), 4000) for k in range(3)]
new_sets = [(GAIN_PRIOR(np.random.default_rng(9000 + k), N_P), make_noise(np.random.default_rng(9500 + k), "white", N_P)) for k in range(2)]

sims_bank = {}
KEEP = ("T_sk", "T_cr", "E_sk", "q_cs")
def slim(sim, keys=KEEP):
    return {k: sim[k].astype(np.float32) for k in keys}
def bank_sim(arm, prior, tag, cache=True):
    key = (arm, tag)
    if key in sims_bank:
        return sims_bank[key]
    prot, env = protocol_and_env(arm)
    sim = slim(simulate(prot, env, body, gains=prior, dt=DT, T_cr0=T_CR0, family="reference"),
               KEEP if tag == "B8000" and arm != "fan_only" else ("T_sk", "T_cr"))
    if cache:
        sims_bank[key] = sim
    return sim

def truth_sim(arm, fam, part):
    prot, env = protocol_and_env(arm); return simulate(prot, env, body, gains=part, dt=DT, T_cr0=T_CR0, family=fam)

def subset(sim, n):
    return {k: (v[:n] if isinstance(v, np.ndarray) and v.ndim == 2 else v) for k, v in sim.items()}

rob = {}; rob_rows = {}
for fam, arm in CONTRASTS:
    truths = {a: truth_sim(a, fam, P1) for a in ("standard", arm)}
    # (a) nested subsets of one bank, 3 tolerances
    for n in (1000, 4000, 8000):
        for tol in (1.25, 1.5, 2.0):
            rows = {a: evaluate(subset(bank_sim(a, big_prior, "B8000"), n), truths[a], N1, tol)[0] for a in ("standard", arm)}
            k = f"{fam}|{arm}-standard|nested_bank{n}|tol{tol}"; rob[k] = paired_compare(rows[arm], rows["standard"]); rob_rows[k] = rows
    # (b) independent 4000 banks, tol 1.5
    for j, pr in enumerate(ind_priors):
        rows = {a: evaluate(bank_sim(a, pr, f"I{j}", cache=False), truths[a], N1)[0] for a in ("standard", arm)}
        k = f"{fam}|{arm}-standard|indep_bank{j}|tol1.5"; rob[k] = paired_compare(rows[arm], rows["standard"]); rob_rows[k] = rows
    # (c) new participant sets, bank B8000[:4000], tol 1.5 (pre-specified settings)
    for j, (pp, nn) in enumerate(new_sets):
        tr = {a: truth_sim(a, fam, pp) for a in ("standard", arm)}
        rows = {a: evaluate(subset(bank_sim(a, big_prior, "B8000"), 4000), tr[a], nn)[0] for a in ("standard", arm)}
        k = f"{fam}|{arm}-standard|new_participants{j}|bank4000|tol1.5"; rob[k] = paired_compare(rows[arm], rows["standard"]); rob_rows[k] = rows
out["paired_robustness"] = rob; out["paired_robustness_rows"] = rob_rows

# ====================================================== 2. participant-level mechanism (sweat_only, standard vs work_only)
fam, arm = "sweat_only", "work_only"
bankS = subset(bank_sim("standard", big_prior, "B8000"), 4000); bankW = subset(bank_sim("work_only", big_prior, "B8000"), 4000)
trS = truth_sim("standard", fam, P1); trW = truth_sim("work_only", fam, P1)
rowsS, accS = evaluate(bankS, trS, N1); rowsW, accW = evaluate(bankW, trW, N1)
dS = (bankS["T_cr"][:, I_END] - bankS["T_cr"][:, 0]).astype(float); dW = (bankW["T_cr"][:, I_END] - bankW["T_cr"][:, 0]).astype(float)
mech_rows = []
for i in range(N_P):
    truth = rowsS[i]["truth"]
    kept = accS[i] & accW[i]; lost = accS[i] & ~accW[i]; gained = ~accS[i] & accW[i]
    wrW = window_rms(bankW["T_sk"], trW["T_sk"][i] + N1[i])           # window errors under work protocol (n,6)
    fail_win = (wrW[lost] >= TOL)                                          # which windows reject the lost candidates
    row = {"participant": i, "truth": truth, "gains": {k: float(P1[k][i]) for k in P1},
           "status": [rowsS[i]["status"], rowsW[i]["status"]],
           "abs_err": [(abs(rowsS[i]["median_error"]) if "median_error" in rowsS[i] else None),
                       (abs(rowsW[i]["median_error"]) if "median_error" in rowsW[i] else None)],
           "n_kept": int(kept.sum()), "n_lost": int(lost.sum()), "n_gained": int(gained.sum()),
           "err_kept_median": float(np.median(dS[kept]) - truth) if kept.sum() else None,
           "err_lost_median": float(np.median(dS[lost]) - truth) if lost.sum() else None,
           "abs_err_kept_mean": float(np.mean(np.abs(dS[kept] - truth))) if kept.sum() else None,
           "abs_err_lost_mean": float(np.mean(np.abs(dS[lost] - truth))) if lost.sum() else None,
           "gain_kept_median": {k: float(np.median(big_prior[k][:4000][kept])) for k in big_prior} if kept.sum() else None,
           "gain_lost_median": {k: float(np.median(big_prior[k][:4000][lost])) for k in big_prior} if lost.sum() else None,
           "lost_fail_window_fraction": [float(x) for x in fail_win.mean(0)] if lost.sum() else None,
           "lost_fail_only_27_37": float(np.mean(fail_win[:, 2:4].any(1) & ~fail_win[:, [0, 1, 4, 5]].any(1))) if lost.sum() else None}
    mech_rows.append(row)
# equal-weight aggregates over participants with both kept and lost sets
ok = [m for m in mech_rows if m["n_kept"] > 0 and m["n_lost"] > 0]
mech_agg = {"n_participants_with_kept_and_lost": len(ok),
            "frac_participants_lost_worse_than_kept": float(np.mean([m["abs_err_lost_mean"] > m["abs_err_kept_mean"] for m in ok])) if ok else None,
            "mean_abs_err_lost_minus_kept": float(np.mean([m["abs_err_lost_mean"] - m["abs_err_kept_mean"] for m in ok])) if ok else None,
            "mean_fail_window_fraction_equal_weight": [float(x) for x in np.mean([m["lost_fail_window_fraction"] for m in ok], 0)] if ok else None,
            "mean_frac_lost_rejected_only_by_27_37_windows": float(np.mean([m["lost_fail_only_27_37"] for m in ok])) if ok else None,
            "median_gain_lost_minus_kept_equal_weight": {k: float(np.mean([m["gain_lost_median"][k] - m["gain_kept_median"][k] for m in ok])) for k in big_prior} if ok else None}
out["mechanism_sweat_work"] = {"rows": mech_rows, "aggregate": mech_agg}

# ---------------- 2b. four-stage decomposition (all common participants), sweat_only / standard vs work_only
def med_err(d, mask, truth):
    return float(np.median(d[mask]) - truth) if mask.sum() else None
decomp = []
for i_p in range(N_P):
    if rowsS[i_p]["status"] != "available" or rowsW[i_p]["status"] != "available":
        continue
    truth = rowsS[i_p]["truth"]; truthW = rowsW[i_p]["truth"]; kept = accS[i_p] & accW[i_p]   # truth differs by protocol
    A = med_err(dS, accS[i_p], truth); B = med_err(dS, kept, truth); C = med_err(dW, kept, truthW); D = med_err(dW, accW[i_p], truthW)
    decomp.append({"participant": i_p, "truth_standard": truth, "truth_work": truthW, "n_A": int(accS[i_p].sum()), "n_kept": int(kept.sum()), "n_D": int(accW[i_p].sum()),
                   "signed": {"A": A, "B": B, "C": C, "D": D},
                   "abs": {"A": abs(A), "B": (abs(B) if B is not None else None), "C": (abs(C) if C is not None else None), "D": abs(D)},
                   "delta_abs": {"A_to_B_candidate_removal": (abs(B) - abs(A)) if B is not None else None,
                                 "B_to_C_same_candidates_protocol": (abs(C) - abs(B)) if (B is not None and C is not None) else None,
                                 "C_to_D_candidate_addition": (abs(D) - abs(C)) if C is not None else None,
                                 "A_to_D_total": abs(D) - abs(A)}})
ok_d = [d for d in decomp if d["delta_abs"]["B_to_C_same_candidates_protocol"] is not None]
decomp_agg = {"n_common": len(decomp), "n_with_kept": len(ok_d),
              "mean_delta_abs": {k: float(np.mean([d["delta_abs"][k] for d in ok_d])) for k in ["A_to_B_candidate_removal", "B_to_C_same_candidates_protocol", "C_to_D_candidate_addition", "A_to_D_total"]},
              "n_participants_where_removal_helps": int(sum(d["delta_abs"]["A_to_B_candidate_removal"] < 0 for d in ok_d)),
              "n_participants_where_same_candidate_change_helps": int(sum(d["delta_abs"]["B_to_C_same_candidates_protocol"] < 0 for d in ok_d)),
              "n_participants_where_addition_helps": int(sum(d["delta_abs"]["C_to_D_candidate_addition"] < 0 for d in ok_d)),
              "note": "one specific ordering (A->B->C->D); not the only possible attribution"}
out["four_stage_decomposition"] = {"rows": decomp, "aggregate": decomp_agg}

# ---------------- 2c. observation-window ablation (fixed common cases; NOT a physical-transient removal)
def eval_ablate(bank, tr, noise, drop):
    d_end = (bank["T_cr"][:, I_END] - bank["T_cr"][:, 0]).astype(float); rows = []; accs = []
    for i_p in range(N_P):
        wr = window_rms(bank["T_sk"], tr["T_sk"][i_p] + noise[i_p])
        keep_w = [w for w in range(6) if w not in drop]
        acc = wr[:, keep_w].max(1) < TOL; accs.append(acc)
        rows.append({"participant": i_p, "truth": float(tr["T_cr"][i_p, I_END] - T_CR0), **ens_row(d_end, acc, float(tr["T_cr"][i_p, I_END] - T_CR0))})
    return rows, accs
common_ids = [d["participant"] for d in decomp]
ablation = {}
for name, drop in [("all_windows", []), ("without_37_40", [4]), ("without_recovery", [5]), ("without_37_40_and_recovery", [4, 5])]:
    rS_a, aS_a = eval_ablate(bankS, trS, N1, drop); rW_a, aW_a = eval_ablate(bankW, trW, N1, drop)
    sub_S = [rS_a[i_p] for i_p in common_ids]; sub_W = [rW_a[i_p] for i_p in common_ids]
    pc = paired_compare(sub_W, sub_S)
    restored = [int((accS[i_p] & ~accW[i_p] & aW_a[i_p]).sum()) for i_p in common_ids]     # lost under all windows, accepted again when the window is dropped
    ablation[name] = {"dropped_windows": [WIN_LAB[w] for w in drop], "paired_on_fixed_common_cases": pc,
                      "availability_work_arm": int(sum(r["status"] == "available" for r in sub_W)),
                      "availability_standard_arm": int(sum(r["status"] == "available" for r in sub_S)),
                      "restored_candidates_work_arm_per_participant": restored, "restored_candidates_total": int(sum(restored))}
out["window_ablation_fixed_common_cases"] = ablation

# pick three cases: largest improvement, typical (median), no improvement — among common-available
common = [m for m in mech_rows if m["status"] == ["available", "available"]]
imp = sorted(common, key=lambda m: m["abs_err"][1] - m["abs_err"][0])
cases = {"largest_improvement": imp[0], "typical": imp[len(imp) // 2], "no_improvement": imp[-1]} if len(imp) >= 3 else {}
out["mechanism_cases"] = {k: v["participant"] for k, v in cases.items()}

# ---- figure 7: three participants x (dT_core kept/lost | gains kept/lost | window rejection | E and q_cs trajectories)
if cases:
    fig, axes = plt.subplots(3, 4, figsize=(14, 9))
    t_min = np.arange(T_LEN) * DT / 60
    for rI, (name, m) in enumerate(cases.items()):
        i = m["participant"]; truth = m["truth"]
        kept = accS[i] & accW[i]; lost = accS[i] & ~accW[i]
        ax = axes[rI, 0]
        ax.hist(dS[kept] - truth, bins=25, color=BLUE, alpha=0.7, label=f"kept ({kept.sum()})")
        ax.hist(dS[lost] - truth, bins=25, color=ORANGE, alpha=0.7, label=f"lost ({lost.sum()})")
        ax.axvline(0, color=INK, lw=1); ax.set_xlabel("candidate ΔT_core − truth (°C)"); ax.legend(frameon=False, fontsize=7)
        ax.set_title(f"{name}: participant {i}  |err| {m['abs_err'][0]:.3f} → {m['abs_err'][1]:.3f}", loc="left", fontsize=8)
        ax = axes[rI, 1]
        ax.scatter(big_prior["sweat"][:4000][kept], big_prior["bloodflow"][:4000][kept], s=8, color=BLUE, alpha=0.6, label="kept")
        ax.scatter(big_prior["sweat"][:4000][lost], big_prior["bloodflow"][:4000][lost], s=8, color=ORANGE, alpha=0.8, label="lost")
        ax.plot(m["gains"]["sweat"], m["gains"]["bloodflow"], marker="*", ms=12, color=INK, label="true gains (other law)")
        ax.set_xlabel("sweat gain"); ax.set_ylabel("blood-flow gain"); ax.legend(frameon=False, fontsize=7)
        ax = axes[rI, 2]
        ax.bar(range(6), m["lost_fail_window_fraction"] or [0] * 6, color=ORANGE)
        ax.set_xticks(range(6)); ax.set_xticklabels(WIN_LAB, rotation=45, fontsize=6); ax.set_ylim(0, 1)
        ax.set_title("fraction of LOST candidates failing each window (work protocol; non-exclusive)", loc="left", fontsize=7)
        ax = axes[rI, 3]
        for arr, c, lab in [(bankW["E_sk"][kept].mean(0), BLUE, "E, kept"), (bankW["E_sk"][lost].mean(0), ORANGE, "E, lost"), (trW["E_sk"][i], INK, "E, truth")]:
            ax.plot(t_min, arr, color=c, lw=1.5, label=lab)
        for arr, c, lab in [(bankW["q_cs"][kept].mean(0), BLUE, "core→skin, kept"), (bankW["q_cs"][lost].mean(0), ORANGE, "core→skin, lost"), (trW["q_cs"][i], INK, "core→skin, truth")]:
            ax.plot(t_min, arr, color=c, lw=1.2, ls="--", label=lab)
        ax.axvspan(27, 37, color="#f0efeb", zorder=0); ax.set_xlabel("time (min)"); ax.set_ylabel("W/m²"); ax.legend(frameon=False, fontsize=6, ncol=2)
    fig.suptitle("Mechanism, three cases: sweat-law mismatch, standard vs workload-swing protocol — same candidate IDs; 'lost' = accepted in standard, rejected in work_only\n"
                 "window rejection fractions are non-exclusive (a candidate can fail several windows)", x=0.01, ha="left", fontsize=9)
    fig.tight_layout(); fig.savefig(os.path.join(OUT, "fig7_mechanism.png")); plt.close(fig)

    # ---- fig 8: four-stage decomposition + ablation for ALL common participants
    fig, axes = plt.subplots(1, 4, figsize=(14, 3.8))
    stages = ["A", "B", "C", "D"]; xs = np.arange(4); labels_y = []
    ax = axes[0]
    for d in decomp:
        y = [d["abs"][k] for k in stages]
        if None in y:
            continue
        hi = d["participant"] in [m["participant"] for m in cases.values()]
        ax.plot(xs, y, color=ORANGE if hi else MUTED, lw=2 if hi else 0.8, alpha=1 if hi else 0.5, marker="o", ms=3)
        if hi:
            labels_y.append((y[-1], f"p{d['participant']}"))
    labels_y.sort(); last = -1.0
    for yv, lab in labels_y:                      # de-overlap labels
        yv = max(yv, last + 0.006); ax.text(3.08, yv, lab, fontsize=7, va="center"); last = yv
    ax.set_xticks(xs); ax.set_xticklabels(["A std,\nall acc.", "B std,\nkept", "C work,\nkept", "D work,\nall acc."], fontsize=7)
    ax.set_ylabel("|ensemble median error| (°C)"); ax.set_title(f"four-stage decomposition, {len(decomp)} common participants\n(orange = the 3 cases)", loc="left", fontsize=7.5)
    ax = axes[1]
    for d in decomp:
        y = [d["signed"][k] for k in stages]
        if None in y:
            continue
        ax.plot(xs, y, color=MUTED, lw=0.8, alpha=0.5, marker="o", ms=3)
    ax.axhline(0, color=INK, lw=1); ax.set_xticks(xs); ax.set_xticklabels(stages); ax.set_ylabel("signed median error (°C)")
    ax.set_title("signed errors, same participants\n(C, D vs work-protocol truth)", loc="left", fontsize=7.5)
    ax = axes[2]
    labs = ["A→B\ncandidate\nremoval", "B→C\nsame candidates,\nprotocol", "C→D\ncandidate\naddition", "A→D\ntotal"]
    keys = ["A_to_B_candidate_removal", "B_to_C_same_candidates_protocol", "C_to_D_candidate_addition", "A_to_D_total"]
    vals = [decomp_agg["mean_delta_abs"][k] for k in keys]
    ax.bar(range(4), vals, color=[BLUE, AQUA, YELLOW, INK]); ax.axhline(0, color=INK, lw=0.8)
    ax.set_xticks(range(4)); ax.set_xticklabels(labs, fontsize=6.5); ax.set_ylabel("mean Δ|error| (°C)")
    ax.set_title(f"stage contributions (mean over {len(ok_d)};\none ordering, not unique attribution)", loc="left", fontsize=7.5)
    ax = axes[3]
    names = list(ablation); vals = [ablation[n]["paired_on_fixed_common_cases"]["mean_delta_abs_error_common"] or 0 for n in names]
    ax.bar(range(len(names)), vals, color=[BLUE, ORANGE, AQUA, YELLOW]); ax.axhline(0, color=INK, lw=0.8)
    ax.set_xticks(range(len(names))); ax.set_xticklabels([n.replace("_", " ") for n in names], fontsize=6.5, rotation=15)
    for j, n in enumerate(names):
        ax.text(j, vals[j], f"n={ablation[n]['paired_on_fixed_common_cases']['n_common_available']}\nrestored {ablation[n]['restored_candidates_total']}", ha="center", va="bottom", fontsize=6)
    ax.set_ylabel("paired Δ|error| work − std (°C)"); ax.set_title("observation-window ablation,\nfixed common cases (not a physical removal)", loc="left", fontsize=7.5)
    fig.suptitle("Four-stage decomposition and observation-window ablation — all common participants (sweat-law mismatch, standard vs workload swing)",
                 x=0.01, ha="left", fontsize=9)
    fig.tight_layout(); fig.savefig(os.path.join(OUT, "fig8_decomposition.png")); plt.close(fig)


# ====================================================== 3. secondary contrast aggregate (perfusion_only, fan)
fam2, arm2 = "perfusion_only", "fan_only"
bF = subset(bank_sim("fan_only", big_prior, "B8000"), 4000); trS2 = truth_sim("standard", fam2, P1); trF = truth_sim("fan_only", fam2, P1)
rS2, aS2 = evaluate(bankS, trS2, N1); rF, aF = evaluate(bF, trF, N1); dS2 = (bankS["T_cr"][:, I_END] - bankS["T_cr"][:, 0]).astype(float)
sec = []
for i in range(N_P):
    truth = rS2[i]["truth"]; kept = aS2[i] & aF[i]; lost = aS2[i] & ~aF[i]
    if kept.sum() and lost.sum():
        sec.append({"abs_err_lost_minus_kept": float(np.mean(np.abs(dS2[lost] - truth)) - np.mean(np.abs(dS2[kept] - truth))),
                    "bloodflow_gain_lost_minus_kept": float(np.median(big_prior["bloodflow"][:4000][lost]) - np.median(big_prior["bloodflow"][:4000][kept]))})
out["mechanism_perfusion_fan_aggregate"] = {"n": len(sec),
    "mean_abs_err_lost_minus_kept": float(np.mean([s["abs_err_lost_minus_kept"] for s in sec])) if sec else None,
    "frac_lost_worse": float(np.mean([s["abs_err_lost_minus_kept"] > 0 for s in sec])) if sec else None,
    "mean_bloodflow_gain_lost_minus_kept": float(np.mean([s["bloodflow_gain_lost_minus_kept"] for s in sec])) if sec else None}

# ====================================================== 4. noise beyond ideal
noise_res = {}
for kind in ["white", "bias", "ar1", "bias+ar1"]:
    nz = make_noise(np.random.default_rng(3131), kind, N_P)
    for fam, arm in CONTRASTS:
        tr = {a: truth_sim(a, fam, P1) for a in ("standard", arm)}
        rows = {a: evaluate(subset(bank_sim(a, big_prior, "B8000"), 4000), tr[a], nz)[0] for a in ("standard", arm)}
        noise_res[f"{fam}|{arm}-standard|{kind}"] = paired_compare(rows[arm], rows["standard"])
out["noise_robustness"] = noise_res

with open(os.path.join(OUT, "mechanism_results.json"), "w") as f:
    json.dump(out, f, indent=1, allow_nan=False)

# ---- console
print("PAIRED ROBUSTNESS  (common n | improved/worsened | mean Δ|err| common | out→in / in→out | avail→unavail / unavail→avail | truth-outside [arm, std])")
for k, v in rob.items():
    dd = v['mean_delta_abs_error_common']; dd = '  -   ' if dd is None else f"{dd:+.3f}"
    print(f"  {k:62s} {v['n_common_available']:2d} | {v['n_improved']:2d}/{v['n_worsened']:2d} | {dd} | {v['outside_to_inside']}/{v['inside_to_outside']} | {v['available_to_unavailable']}/{v['unavailable_to_available']} | {v['n_truth_outside']}")
print("\nMECHANISM sweat_only/work_only aggregate:", json.dumps(mech_agg, indent=1))
print("cases:", out["mechanism_cases"])
print("\nMECHANISM perfusion_only/fan aggregate:", json.dumps(out["mechanism_perfusion_fan_aggregate"], indent=1))
print("\nNOISE ROBUSTNESS:")
for k, v in noise_res.items():
    dd = v['mean_delta_abs_error_common']; dd = '  -   ' if dd is None else f"{dd:+.3f}"
    print(f"  {k:48s} common {v['n_common_available']:2d} | {v['n_improved']:2d}/{v['n_worsened']:2d} | {dd} | out→in {v['outside_to_inside']} in→out {v['inside_to_outside']} | truth-outside {v['n_truth_outside']}")

print("\nFOUR-STAGE DECOMPOSITION:", json.dumps(decomp_agg, indent=1))
print("\nWINDOW ABLATION (fixed common cases):")
for n, v in ablation.items():
    pc = v["paired_on_fixed_common_cases"]; dd = pc["mean_delta_abs_error_common"]
    print(f"  {n:28s} common {pc['n_common_available']:2d} | impr/wors {pc['n_improved']}/{pc['n_worsened']} | Δ|err| {'-' if dd is None else f'{dd:+.3f}'} | avail work/std {v['availability_work_arm']}/{v['availability_standard_arm']} | restored {v['restored_candidates_total']}")
