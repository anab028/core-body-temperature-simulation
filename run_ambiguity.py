"""
Ambiguity-reduction experiment, v2 (after the seventh review round).

Confound removed: every arm now uses IDENTICAL fitting windows and the identical acceptance
procedure; only the physical protocol differs. Total external work is equal in all arms.

Arms (all: 10 min rest | 30 min exercise | 15 min recovery; 6 identical windows with boundaries at
minutes 0, 22, 27, 32, 37, 40, 55; same-candidate acceptance: max over windows of RMS/noise < tol):
  standard   100 W throughout, fan off
  fan_only   fan ON (1.5 m/s) during exercise minutes 22-27
  work_only  workload 50 W during minutes 27-32 and 150 W during 32-37 (equal total work: 180 kJ)
  combined   both
Optional factor: idealised total-E constraint (+-10 %) on top of each arm.

Paired design: the same participants (gains), the same noise realisations and the same candidate
bank prior are used across arms within a seed; 3 seeds (participants + noise + bank) reported as
mean and range. Truth families: reference, perfusion_only, sweat_only. Candidate bank: reference.

Metrics per participant: ensemble available (n >= 20) / insufficient (1-19) / none (0);
ensemble median error at end of exercise; truth outside the accepted-ensemble 95 % range (NOT a
calibrated interval); width. Everything synthetic.
"""
import os, json
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from thermo import Body, simulate, make_env

OUT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "out"); os.makedirs(OUT, exist_ok=True)
BLUE, ORANGE, AQUA, YELLOW, INK, MUTED = "#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#0b0b0b", "#52514e"
plt.rcParams.update({"font.size": 10, "axes.spines.top": False, "axes.spines.right": False, "figure.dpi": 130})

body = Body(); DT = 2.0; T_CR0 = 36.9; CAM = 0.3; TOL = 1.5; M_S = 4000; N_P = 20; SEEDS = [1, 2, 3]
GAIN_PRIOR = lambda r, m: {"sweat": r.lognormal(0, 0.45, m), "bloodflow": r.lognormal(0, 0.3, m), "hc": r.lognormal(0, 0.2, m)}
ARMS = ["standard", "fan_only", "work_only", "combined"]
FAMS = ["reference", "perfusion_only", "sweat_only"]
SPM = int(60 / DT)                                          # samples per minute
WIN_MIN = [0, 22, 27, 32, 37, 40, 55]
WINDOWS = [(WIN_MIN[i] * SPM, WIN_MIN[i + 1] * SPM) for i in range(len(WIN_MIN) - 1)]
I_END = 40 * SPM


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
    return {"t": t, "M": M, "W": W, "dt": DT}, env, float(np.sum(P) * DT / 1000.0)   # external work kJ


def scores(cand_T_sk, T_obs):
    sc = np.zeros(cand_T_sk.shape[0])
    for a, b in WINDOWS:
        sc = np.maximum(sc, np.sqrt(np.mean((cand_T_sk[:, a:b] - T_obs[None, a:b]) ** 2, axis=1)) / CAM)
    return sc


def per_participant(d_end, acc, truth):
    n = int(acc.sum())
    if n == 0:
        return {"n": 0, "status": "none"}
    if n < 20:
        return {"n": n, "status": "insufficient"}
    lo, hi = np.percentile(d_end[acc], [2.5, 97.5])
    return {"n": n, "status": "available", "median_error": float(np.median(d_end[acc]) - truth),
            "range95": [float(lo), float(hi)], "width": float(hi - lo),
            "truth_outside_range95": bool(not (lo <= truth <= hi))}


def summarise(rows):
    av = [x for x in rows if x["status"] == "available"]
    return {"n_available": len(av), "n_insufficient": sum(x["status"] == "insufficient" for x in rows),
            "n_none": sum(x["status"] == "none" for x in rows),
            "n_truth_outside_of_all": sum(x.get("truth_outside_range95", False) for x in rows),
            "coverage_among_available": (float(np.mean([not x["truth_outside_range95"] for x in av])) if av else None),
            "mean_abs_median_error": (float(np.mean([abs(x["median_error"]) for x in av])) if av else None),
            "mean_median_error": (float(np.mean([x["median_error"] for x in av])) if av else None),
            "mean_width": (float(np.mean([x["width"] for x in av])) if av else None),
            "mean_n_accepted": float(np.mean([x["n"] for x in rows]))}


work = {}; res = {}; rows_all = {}
for seed in SEEDS:
    r = np.random.default_rng(100 + seed)
    bank_prior = GAIN_PRIOR(r, M_S); part = GAIN_PRIOR(r, N_P); noise = r.normal(0, CAM, (N_P, 55 * SPM))
    for arm in ARMS:
        prot, env, wk = protocol_and_env(arm); work[arm] = wk
        bank = simulate(prot, env, body, gains=bank_prior, dt=DT, T_cr0=T_CR0, family="reference")
        d_end = bank["T_cr"][:, I_END] - bank["T_cr"][:, 0]; E_bank = bank["E_sk"][:, :-1].sum(1) * DT
        for fam in FAMS:
            tr = simulate(prot, env, body, gains=part, dt=DT, T_cr0=T_CR0, family=fam)
            for totalE in (False, True):
                rows = []
                for i in range(N_P):
                    T_obs = tr["T_sk"][i] + noise[i]; truth = float(tr["T_cr"][i, I_END] - T_CR0)
                    acc = scores(bank["T_sk"], T_obs) < TOL
                    if totalE:
                        E_true = tr["E_sk"][i, :-1].sum() * DT; acc = acc & (np.abs(E_bank - E_true) / E_true < 0.10)
                    rows.append({"participant": i, "gains": {k: float(part[k][i]) for k in part}, "truth": truth,
                                 **per_participant(d_end, acc, truth)})
                key = f"{fam}|{arm}|{'+totalE' if totalE else 'noE'}"
                res.setdefault(key, {})[str(seed)] = summarise(rows)
                rows_all.setdefault(key, {})[str(seed)] = rows


def agg(key, field):
    vals = [res[key][str(s)][field] for s in SEEDS]; vals = [v for v in vals if v is not None]
    return {"mean": float(np.mean(vals)), "min": float(min(vals)), "max": float(max(vals))} if vals else None


table = {}
for key in res:
    table[key] = {f: agg(key, f) for f in ["n_available", "n_insufficient", "n_none", "n_truth_outside_of_all",
                                             "coverage_among_available", "mean_abs_median_error", "mean_median_error",
                                             "mean_width", "mean_n_accepted"]}
# paired differences vs standard (same seed) for the headline metric
paired = {}
for fam in FAMS:
    for e in ("noE", "+totalE"):
        for arm in ARMS[1:]:
            d = [res[f"{fam}|{arm}|{e}"][str(s)]["n_truth_outside_of_all"] - res[f"{fam}|standard|{e}"][str(s)]["n_truth_outside_of_all"] for s in SEEDS]
            paired[f"{fam}|{arm}-standard|{e}"] = {"delta_truth_outside_per_20": {"mean": float(np.mean(d)), "per_seed": d}}

# ---- robustness: participants & noise FIXED (seed 1); vary bank size and tolerance; noE only
rob = {}; r0 = np.random.default_rng(101); _ = GAIN_PRIOR(r0, M_S)      # replay seed-1 stream: bank prior first,
part1 = GAIN_PRIOR(r0, N_P); noise1 = r0.normal(0, CAM, (N_P, 55 * SPM))  # then the same participants and noise
rb = np.random.default_rng(555)
for bank_n in [1000, 4000, 8000]:
    bp = GAIN_PRIOR(rb, bank_n)
    for arm in ["standard", "fan_only", "work_only"]:
        prot, env, _ = protocol_and_env(arm)
        bank = simulate(prot, env, body, gains=bp, dt=DT, T_cr0=T_CR0, family="reference")
        d_end = bank["T_cr"][:, I_END] - bank["T_cr"][:, 0]
        for fam in ["perfusion_only", "sweat_only"]:
            tr = simulate(prot, env, body, gains=part1, dt=DT, T_cr0=T_CR0, family=fam)
            sc_all = [scores(bank["T_sk"], tr["T_sk"][i] + noise1[i]) for i in range(N_P)]
            for tol in [1.25, 1.5, 2.0]:
                rows = [per_participant(d_end, sc_all[i] < tol, float(tr["T_cr"][i, I_END] - T_CR0)) for i in range(N_P)]
                rob[f"{fam}|{arm}|bank{bank_n}|tol{tol}"] = summarise(rows)
rob_paired = {}
for fam, arm in [("perfusion_only", "fan_only"), ("sweat_only", "work_only")]:
    for bank_n in [1000, 4000, 8000]:
        for tol in [1.25, 1.5, 2.0]:
            a = rob[f"{fam}|{arm}|bank{bank_n}|tol{tol}"]; b = rob[f"{fam}|standard|bank{bank_n}|tol{tol}"]
            rob_paired[f"{fam}|{arm}-standard|bank{bank_n}|tol{tol}"] = {
                "delta_truth_outside": a["n_truth_outside_of_all"] - b["n_truth_outside_of_all"],
                "delta_abs_median_error": (a["mean_abs_median_error"] - b["mean_abs_median_error"]) if (a["mean_abs_median_error"] is not None and b["mean_abs_median_error"] is not None) else None,
                "n_available": [a["n_available"], b["n_available"]]}

# ---- mechanism: which candidate gains survive in each arm (seed 1, tol 1.5, bank 4000)
mech = {}
r1 = np.random.default_rng(101); bp1 = GAIN_PRIOR(r1, M_S); part1b = GAIN_PRIOR(r1, N_P); noise1b = r1.normal(0, CAM, (N_P, 55 * SPM))
for fam, arms_m in [("perfusion_only", ["standard", "fan_only"]), ("sweat_only", ["standard", "work_only"])]:
    for arm in arms_m:
        prot, env, _ = protocol_and_env(arm)
        bank = simulate(prot, env, body, gains=bp1, dt=DT, T_cr0=T_CR0, family="reference")
        tr = simulate(prot, env, body, gains=part1b, dt=DT, T_cr0=T_CR0, family=fam)
        acc_any = np.zeros(M_S, bool); g_acc = {"sweat": [], "bloodflow": [], "hc": []}; n_acc = []
        core_skin_gap_acc, core_skin_gap_truth = [], []
        for i in range(N_P):
            acc = scores(bank["T_sk"], tr["T_sk"][i] + noise1b[i]) < TOL; n_acc.append(int(acc.sum()))
            for k in g_acc:
                g_acc[k].extend(bp1[k][acc].tolist())
            if acc.sum():
                core_skin_gap_acc.append(float(np.mean(bank["T_cr"][acc, I_END] - bank["T_sk"][acc, I_END])))
                core_skin_gap_truth.append(float(tr["T_cr"][i, I_END] - tr["T_sk"][i, I_END]))
        q = lambda v: [float(x) for x in np.percentile(v, [10, 50, 90])] if len(v) else None
        mech[f"{fam}|{arm}"] = {"accepted_gain_p10_p50_p90": {k: q(v) for k, v in g_acc.items()},
                                "participant_true_gain_p10_p50_p90": {k: q(part1b[k]) for k in part1b},
                                "mean_n_accepted": float(np.mean(n_acc)),
                                "core_minus_skin_at_end_accepted_mean": float(np.mean(core_skin_gap_acc)) if core_skin_gap_acc else None,
                                "core_minus_skin_at_end_truth_mean": float(np.mean(core_skin_gap_truth)) if core_skin_gap_truth else None}

out = {"design": {"windows_min": WIN_MIN, "identical_windows_all_arms": True, "tol": TOL, "external_work_kJ": work,
                  "n_participants": N_P, "seeds": SEEDS, "bank_size": M_S, "candidate_family": "reference"},
       "per_seed": res, "aggregate": table, "paired_vs_standard": paired,
       "participant_rows": rows_all, "robustness_fixed_participants": rob, "robustness_paired": rob_paired,
       "mechanism_seed1": mech}
with open(os.path.join(OUT, "ambiguity_results.json"), "w") as f:
    json.dump(out, f, indent=2, allow_nan=False)

# ---- figure: truth-outside count and availability, per family, arms on x, noE only (totalE in JSON)
fig, axes = plt.subplots(1, 3, figsize=(12.5, 3.8), sharey=False)
x = np.arange(len(ARMS)); w = 0.26
for j, (fam, c) in enumerate([("reference", BLUE), ("perfusion_only", ORANGE), ("sweat_only", AQUA)]):
    for ax, field, ttl in [(axes[0], "n_truth_outside_of_all", "truth outside accepted-ensemble 95 % range (of 20)"),
                           (axes[1], "n_available", "participants with an ensemble (n ≥ 20) of 20"),
                           (axes[2], "mean_abs_median_error", "mean |ensemble median error| (°C)")]:
        vals = [table[f"{fam}|{a}|noE"][field] for a in ARMS]
        m = [v["mean"] if v else 0 for v in vals]; lo = [v["mean"] - v["min"] if v else 0 for v in vals]; hi = [v["max"] - v["mean"] if v else 0 for v in vals]
        ax.bar(x + (j - 1) * w, m, w, yerr=[lo, hi], color=c, label=fam, capsize=2, error_kw={"lw": 0.8})
        ax.set_xticks(x); ax.set_xticklabels(ARMS, fontsize=8); ax.set_title(ttl, loc="left", fontsize=8.5)
axes[0].legend(frameon=False, fontsize=8)
fig.suptitle("Ambiguity reduction v2 — identical windows & acceptance in every arm; equal external work; paired participants/noise/bank; "
             "bars = mean over 3 seeds, whiskers = range", x=0.01, ha="left", fontsize=8.5)
fig.tight_layout(); fig.savefig(os.path.join(OUT, "fig6_ambiguity.png")); plt.close(fig)

print("external work kJ:", work)
print(f"{'key':40s} {'avail':>7s} {'insuf':>6s} {'none':>5s} {'outside':>8s} {'cover':>6s} {'|med|':>6s} {'med':>6s} {'width':>6s}")
for k, v in table.items():
    f = lambda a, p=2: "  -  " if a is None else f"{a['mean']:.{p}f}"
    print(f"{k:40s} {f(v['n_available'],1):>7s} {f(v['n_insufficient'],1):>6s} {f(v['n_none'],1):>5s} {f(v['n_truth_outside_of_all'],1):>8s} {f(v['coverage_among_available']):>6s} {f(v['mean_abs_median_error'],3):>6s} {f(v['mean_median_error'],3):>6s} {f(v['mean_width'],3):>6s}")
print("\nPAIRED delta (truth-outside per 20) vs standard, per seed:")
for k, v in paired.items():
    print(f"  {k:40s} {v['delta_truth_outside_per_20']['per_seed']}  mean {v['delta_truth_outside_per_20']['mean']:+.2f}")

print("\nROBUSTNESS (participants/noise fixed): delta truth-outside vs standard | delta |med err| | n_available [arm, standard]")
for k, v in rob_paired.items():
    d = v["delta_abs_median_error"]; print(f"  {k:52s} {v['delta_truth_outside']:+d}  {'' if d is None else f'{d:+.3f}'}  {v['n_available']}")
print("\nMECHANISM (seed 1): accepted gain p10/p50/p90 vs participant true gains; core-skin gap")
for k, v in mech.items():
    print(f"  {k:28s} n_acc={v['mean_n_accepted']:.0f} sweat {v['accepted_gain_p10_p50_p90']['sweat']} bloodflow {v['accepted_gain_p10_p50_p90']['bloodflow']} | truth sweat {v['participant_true_gain_p10_p50_p90']['sweat']} bloodflow {v['participant_true_gain_p10_p50_p90']['bloodflow']} | gap acc {v['core_minus_skin_at_end_accepted_mean']} truth {v['core_minus_skin_at_end_truth_mean']}")
