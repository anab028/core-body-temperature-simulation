"""
Likelihood-based inference with an explicit sensor-error model — final version.

Observation model (stated assumptions):
    y_t = T_sk,model(theta; t) + b + e_t
    b   ~ N(0, SIG_B^2)                    session bias, one per recording            (SIG_B = 0.5 degC)
    e_t ~ AR(1), marginal SD SIG_E, corr exp(-dt/TAU)  (SIG_E = 0.3 degC, TAU = 60 s)
Candidate likelihood: b marginalised analytically, AR(1) whitened (Woodbury on the whitened bias
vector). Prior sample = bank of reference-law simulations with gains from the prior; posterior over
dT_core(40 min) = importance-weighted bank; interval = weighted 2.5-97.5 %; availability = ESS >= 20.
ESS is an importance-sampling reliability diagnostic (it is NOT a mismatch detector).

Main inference = 'full' (bias + AR(1)). 'white' (independent noise, no bias) is kept only as the
naive comparison. The earlier 'full+disc' variant was removed: its covariance did not match the stated
observation model.

Parts:
  1. Preliminary held-out coverage check: 40 reference-family participants, noise from the stated model.
  2. Physiology mismatch (sweat_only, perfusion_only) on the frozen standard / work_only protocols,
     same frozen inference; paired work_only - standard with coverage transitions on common cases.
  3. Numerical stability of the posterior: pre-specified high- and low-ESS cases from each family,
     nested bank 4000 -> 8000 -> 16000 from one bank seed, repeated with a second bank seed; posterior
     median, interval endpoints and ESS compared. Pass = variation small relative to the ~0.02 degC effect.
Coverage is reported with counts and Wilson 95 % intervals; unavailable metrics are N/A, not zero.
All synthetic; no camera model.
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

body = Body(); DT = 2.0; T_CR0 = 36.9
SIG_E, SIG_B, TAU = 0.30, 0.50, 60.0
RHO = float(np.exp(-DT / TAU))
GAIN_PRIOR = lambda r, m: {"sweat": r.lognormal(0, 0.45, m), "bloodflow": r.lognormal(0, 0.3, m), "hc": r.lognormal(0, 0.2, m)}
SPM = int(60 / DT); I_END = 40 * SPM; T_LEN = 55 * SPM; M_S = 4000
ESS_MIN = 20; EFFECT = 0.02          # the effect size the stability check is judged against
VARIANTS = ["full", "white"]; ARMS = ["standard", "work_only"]


def protocol_and_env(arm):
    rest, ex, rec = 600, 1800, 900
    t = np.arange(0, rest + ex + rec, DT); v = np.full_like(t, 0.15, float)
    P = np.zeros_like(t); in_ex = (t >= rest) & (t < rest + ex); P[in_ex] = 100.0
    if arm == "work_only":
        P[(t >= 27 * 60) & (t < 32 * 60)] = 50.0; P[(t >= 32 * 60) & (t < 37 * 60)] = 150.0
    M = np.full_like(t, 58.0, float); W = np.zeros_like(t)
    M[in_ex] = 58.0 + P[in_ex] / 0.20 / body.A_D; W[in_ex] = P[in_ex] / body.A_D
    rec_mask = t >= rest + ex; M[rec_mask] = 58.0 + (M[in_ex][-1] - 58.0) * np.exp(-(t[rec_mask] - (rest + ex)) / 180.0)
    env = make_env(t); env["v"] = v
    return {"t": t, "M": M, "W": W, "dt": DT}, env


def gen_noise(r, n):
    w = r.normal(0, SIG_E, (n, T_LEN)); e = np.zeros_like(w); e[:, 0] = w[:, 0]
    for i in range(1, T_LEN):
        e[:, i] = RHO * e[:, i - 1] + np.sqrt(1 - RHO ** 2) * w[:, i]
    return e + r.normal(0, SIG_B, (n, 1))


def make_bank(arm, prior, chunk=4000):
    """simulate a (possibly large) bank in chunks; keep only float32 T_sk and dT_core(40 min)"""
    prot, env = protocol_and_env(arm); n = len(prior["sweat"]); T_sk = []; d_end = []
    for a in range(0, n, chunk):
        g = {k: v[a:a + chunk] for k, v in prior.items()}
        s = simulate(prot, env, body, gains=g, dt=DT, T_cr0=T_CR0, family="reference")
        T_sk.append(s["T_sk"].astype(np.float32)); d_end.append((s["T_cr"][:, I_END] - s["T_cr"][:, 0]).astype(float))
    return {"T_sk": np.concatenate(T_sk), "d_end": np.concatenate(d_end)}


def loglik(resid, variant):
    if variant == "white":
        return -0.5 * np.sum(resid ** 2, axis=1) / SIG_E ** 2
    w = np.empty_like(resid); w[:, 0] = resid[:, 0]
    w[:, 1:] = (resid[:, 1:] - RHO * resid[:, :-1]) / np.sqrt(1 - RHO ** 2)
    u = np.full(resid.shape[1], (1 - RHO) / np.sqrt(1 - RHO ** 2)); u[0] = 1.0
    uu = float(u @ u); uw = w @ u; s2 = SIG_E ** 2
    quad = np.sum(w ** 2, axis=1) / s2 - (SIG_B ** 2) * uw ** 2 / (s2 * (s2 + SIG_B ** 2 * uu))
    return -0.5 * quad


def posterior(bank, y, variant, truth, n=None):
    T_sk = bank["T_sk"] if n is None else bank["T_sk"][:n]; d_end = bank["d_end"] if n is None else bank["d_end"][:n]
    ll = loglik(y[None, :] - T_sk, variant); ll -= ll.max(); wgt = np.exp(ll); wgt /= wgt.sum()
    ess = float(1.0 / np.sum(wgt ** 2)); order = np.argsort(d_end); cw = np.cumsum(wgt[order])
    lo, med, hi = [float(d_end[order][min(np.searchsorted(cw, q), len(cw) - 1)]) for q in (0.025, 0.5, 0.975)]
    return {"ess": ess, "status": "available" if ess >= ESS_MIN else "insufficient", "median": med,
            "median_error": med - truth, "ci95": [lo, hi], "width": hi - lo, "covered": bool(lo <= truth <= hi)}


def wilson(k, n, z=1.96):
    if n == 0:
        return None
    p = k / n; d = 1 + z * z / n; c = (p + z * z / (2 * n)) / d; h = z * np.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / d
    return [float(max(0, c - h)), float(min(1, c + h))]


def summarise(rows):
    av = [x for x in rows if x["status"] == "available"]; k_av = sum(x["covered"] for x in av); k_all = sum(x["covered"] for x in rows)
    return {"n": len(rows), "n_available": len(av),
            "coverage_available": {"k": int(k_av), "n": len(av), "rate": (k_av / len(av) if av else None), "wilson95": wilson(k_av, len(av))},
            "coverage_all": {"k": int(k_all), "n": len(rows), "rate": k_all / len(rows), "wilson95": wilson(k_all, len(rows))},
            "mean_abs_error": float(np.mean([abs(x["median_error"]) for x in av])) if av else None,
            "rmse": float(np.sqrt(np.mean([x["median_error"] ** 2 for x in av]))) if av else None,
            "mean_width": float(np.mean([x["width"] for x in av])) if av else None,
            "median_ess": float(np.median([x["ess"] for x in rows]))}


def paired(rows_w, rows_s):
    common = [(a, b) for a, b in zip(rows_w, rows_s) if a["status"] == "available" and b["status"] == "available"]
    d = [abs(a["median_error"]) - abs(b["median_error"]) for a, b in common]
    kw = sum(a["covered"] for a, _ in common); ks = sum(b["covered"] for _, b in common)
    return {"n_common": len(common), "n_improved": int(sum(v < 0 for v in d)), "n_worsened": int(sum(v > 0 for v in d)),
            "mean_delta_abs_error": float(np.mean(d)) if d else None,
            "coverage_common": {"work_only": f"{kw}/{len(common)}", "standard": f"{ks}/{len(common)}",
                                "uncovered_to_covered": int(sum(a["covered"] and not b["covered"] for a, b in common)),
                                "covered_to_uncovered": int(sum(b["covered"] and not a["covered"] for a, b in common))},
            "availability": {"work_only": int(sum(x["status"] == "available" for x in rows_w)), "standard": int(sum(x["status"] == "available" for x in rows_s)),
                             "available_to_unavailable": int(sum(b["status"] == "available" and a["status"] != "available" for a, b in zip(rows_w, rows_s))),
                             "unavailable_to_available": int(sum(a["status"] == "available" and b["status"] != "available" for a, b in zip(rows_w, rows_s)))}}


out = {"observation_model": {"SIG_E": SIG_E, "SIG_B": SIG_B, "TAU_s": TAU, "rho_per_step": RHO, "bank": M_S, "ess_min": ESS_MIN,
                             "interval": "weighted 2.5-97.5 % of bank dT_core(40 min)", "main_inference": "full", "comparison": "white"},
       "held_out_reference": {}, "mismatch": {}, "paired_mismatch": {}, "rows": {}}

# banks: seed A (16000, nested 4000/8000/16000 used everywhere; main analyses use the first 4000) and seed B (16000) for stability
prior_A = GAIN_PRIOR(np.random.default_rng(31415), 16000); prior_B = GAIN_PRIOR(np.random.default_rng(27182), 16000)
bankA = {arm: make_bank(arm, prior_A) for arm in ARMS}

# ---- 1. preliminary held-out coverage check (reference family, noise from the stated model)
N_REF = 40; ref_gains = GAIN_PRIOR(np.random.default_rng(777), N_REF); noise_ref = gen_noise(np.random.default_rng(778), N_REF)
tr_ref = {}
for arm in ARMS:
    prot, env = protocol_and_env(arm); tr_ref[arm] = simulate(prot, env, body, gains=ref_gains, dt=DT, T_cr0=T_CR0, family="reference")
    for var in VARIANTS:
        rows = [dict(participant=i, truth=float(tr_ref[arm]["T_cr"][i, I_END] - T_CR0),
                     **posterior(bankA[arm], tr_ref[arm]["T_sk"][i] + noise_ref[i], var, float(tr_ref[arm]["T_cr"][i, I_END] - T_CR0), n=M_S)) for i in range(N_REF)]
        out["held_out_reference"][f"{arm}|{var}"] = summarise(rows); out["rows"][f"reference|{arm}|{var}"] = rows

# ---- 2. physiology mismatch, frozen protocols, frozen inference
N_P = 20; mm_gains = GAIN_PRIOR(np.random.default_rng(2024), N_P); noise_mm = gen_noise(np.random.default_rng(2025), N_P)
tr_mm = {}
for fam in ["sweat_only", "perfusion_only"]:
    for arm in ARMS:
        prot, env = protocol_and_env(arm); tr = simulate(prot, env, body, gains=mm_gains, dt=DT, T_cr0=T_CR0, family=fam); tr_mm[(fam, arm)] = tr
        for var in VARIANTS:
            rows = [dict(participant=i, truth=float(tr["T_cr"][i, I_END] - T_CR0),
                         **posterior(bankA[arm], tr["T_sk"][i] + noise_mm[i], var, float(tr["T_cr"][i, I_END] - T_CR0), n=M_S)) for i in range(N_P)]
            out["mismatch"][f"{fam}|{arm}|{var}"] = summarise(rows); out["rows"][f"{fam}|{arm}|{var}"] = rows
    for var in VARIANTS:
        out["paired_mismatch"][f"{fam}|work_only-standard|{var}"] = paired(out["rows"][f"{fam}|work_only|{var}"], out["rows"][f"{fam}|standard|{var}"])

# ---- 3. numerical stability: pre-specified high/low-ESS cases, nested bank sizes, two bank seeds ('full' only)
bankB = {arm: make_bank(arm, prior_B) for arm in ARMS}
cases = []
for fam in ["reference", "sweat_only", "perfusion_only"]:
    rows = out["rows"][f"{fam}|standard|full"]; ess = np.array([r["ess"] for r in rows])
    cases.append((fam, int(np.argmax(ess)), "high_ESS")); cases.append((fam, int(np.argmin(ess)), "low_ESS"))
stab = {"criterion": f"PASS only if (i) within each bank seed |median(16000) - median(8000)| < {EFFECT/2} and both endpoints move < {EFFECT}, AND (ii) across seeds at 16000 |median_A - median_B| < {EFFECT/2} and both endpoints differ < {EFFECT}; else 'numerically unresolved'", "cases": {}}
for fam, i, tag in cases:
    for arm in ARMS:
        if fam == "reference":
            y = tr_ref[arm]["T_sk"][i] + noise_ref[i]; truth = float(tr_ref[arm]["T_cr"][i, I_END] - T_CR0)
        else:
            y = tr_mm[(fam, arm)]["T_sk"][i] + noise_mm[i]; truth = float(tr_mm[(fam, arm)]["T_cr"][i, I_END] - T_CR0)
        rec = {"participant": i, "truth": truth, "by_bank": {}}
        for seed_name, bank in [("A", bankA[arm]), ("B", bankB[arm])]:
            for n in (4000, 8000, 16000):
                p = posterior(bank, y, "full", truth, n=n)
                rec["by_bank"][f"{seed_name}{n}"] = {"median": p["median"], "ci95": p["ci95"], "ess": p["ess"], "status": p["status"]}
        ok = True
        for sn in ("A", "B"):
            a, b = rec["by_bank"][f"{sn}8000"], rec["by_bank"][f"{sn}16000"]
            ok &= abs(a["median"] - b["median"]) < EFFECT / 2 and abs(a["ci95"][0] - b["ci95"][0]) < EFFECT and abs(a["ci95"][1] - b["ci95"][1]) < EFFECT
        rec["seed_spread_16000"] = {"median": abs(rec["by_bank"]["A16000"]["median"] - rec["by_bank"]["B16000"]["median"]),
                                    "lo": abs(rec["by_bank"]["A16000"]["ci95"][0] - rec["by_bank"]["B16000"]["ci95"][0]),
                                    "hi": abs(rec["by_bank"]["A16000"]["ci95"][1] - rec["by_bank"]["B16000"]["ci95"][1])}
        sp = rec["seed_spread_16000"]
        ok &= sp["median"] < EFFECT / 2 and sp["lo"] < EFFECT and sp["hi"] < EFFECT      # cross-seed agreement
        rec["verdict"] = "PASS" if ok else "numerically unresolved"
        stab["cases"][f"{fam}|{tag}|{arm}|p{i}"] = rec
# whole-population paired benefit at 4000 vs 16000 (seed A) for the sweat mismatch
for n in (4000, 16000):
    rw = [dict(participant=i, truth=float(tr_mm[("sweat_only", "work_only")]["T_cr"][i, I_END] - T_CR0),
               **posterior(bankA["work_only"], tr_mm[("sweat_only", "work_only")]["T_sk"][i] + noise_mm[i], "full", float(tr_mm[("sweat_only", "work_only")]["T_cr"][i, I_END] - T_CR0), n=n)) for i in range(N_P)]
    rs = [dict(participant=i, truth=float(tr_mm[("sweat_only", "standard")]["T_cr"][i, I_END] - T_CR0),
               **posterior(bankA["standard"], tr_mm[("sweat_only", "standard")]["T_sk"][i] + noise_mm[i], "full", float(tr_mm[("sweat_only", "standard")]["T_cr"][i, I_END] - T_CR0), n=n)) for i in range(N_P)]
    stab[f"sweat_only_paired_benefit_bank{n}"] = paired(rw, rs)
out["numerical_stability"] = stab

with open(os.path.join(OUT, "calibrated_results.json"), "w") as f:
    json.dump(out, f, indent=1, allow_nan=False)

# ---- figure 9 (full inference only; N/A where unavailable)
keys = [("reference", "standard"), ("reference", "work_only"), ("sweat_only", "standard"), ("sweat_only", "work_only"), ("perfusion_only", "standard"), ("perfusion_only", "work_only")]
labels = [f"{f.replace('_only', '')}\n{a.replace('_only', '')}" for f, a in keys]; x = np.arange(len(keys))
fig, axes = plt.subplots(1, 4, figsize=(15, 3.8))
def get(f, a):
    return (out["held_out_reference"] if f == "reference" else out["mismatch"])[f"{a}|full" if f == "reference" else f"{f}|{a}|full"]
cov = [get(f, a)["coverage_available"] for f, a in keys]
axes[0].bar(x, [c["rate"] if c["rate"] is not None else 0 for c in cov], 0.55, color=BLUE)
for j, c in enumerate(cov):
    if c["rate"] is None:
        axes[0].text(j, 0.02, "N/A", ha="center", fontsize=7, color=ORANGE)
    else:
        axes[0].plot([j, j], c["wilson95"], color=INK, lw=1); axes[0].text(j, c["wilson95"][1] + 0.02, f"{c['k']}/{c['n']}", ha="center", fontsize=6.5)
axes[0].axhline(0.95, color=INK, lw=1, ls=":"); axes[0].set_ylim(0, 1.15); axes[0].set_title("coverage of nominal 95 % interval, available cases\n(k/n, Wilson 95 %)", loc="left", fontsize=8)
for ax, k, ttl in [(axes[1], "mean_width", "mean 95 % width (°C)"), (axes[2], "mean_abs_error", "mean |posterior-median error| (°C)")]:
    vals = [get(f, a)[k] for f, a in keys]
    ax.bar(x, [v if v is not None else 0 for v in vals], 0.55, color=BLUE)
    for j, v in enumerate(vals):
        if v is None:
            ax.text(j, 0.005, "N/A", ha="center", fontsize=7, color=ORANGE)
    ax.set_title(ttl, loc="left", fontsize=8)
axes[3].bar(x, [get(f, a)["n_available"] for f, a in keys], 0.55, color=BLUE)
for j, (f, a) in enumerate(keys):
    axes[3].text(j, get(f, a)["n_available"] + 0.5, f"of {get(f, a)['n']}", ha="center", fontsize=6.5)
axes[3].set_title("participants with ESS ≥ 20 (availability)", loc="left", fontsize=8)
for ax in axes:
    ax.set_xticks(x); ax.set_xticklabels(labels, fontsize=6.5)
fig.suptitle("Full likelihood (bias + AR(1)): preliminary held-out coverage check (reference) and frozen-inference mismatch cases; 'white' likelihood omitted (ESS ≈ 1 everywhere)",
             x=0.01, ha="left", fontsize=8.5)
fig.tight_layout(); fig.savefig(os.path.join(OUT, "fig9_calibrated.png")); plt.close(fig)

# ---- console
print(f"{'case':36s} {'n_av':>4s} {'cov_av k/n (Wilson)':>24s} {'cov_all k/n':>12s} {'|err|':>6s} {'width':>6s} {'medESS':>7s}")
for sec in ("held_out_reference", "mismatch"):
    for k, v in out[sec].items():
        ca, cl = v["coverage_available"], v["coverage_all"]
        w = f"{ca['k']}/{ca['n']} [{ca['wilson95'][0]:.2f},{ca['wilson95'][1]:.2f}]" if ca["n"] else "N/A"
        f = lambda z: " N/A " if z is None else f"{z:.3f}"
        print(f"{k:36s} {v['n_available']:4d} {w:>24s} {cl['k']:>6d}/{cl['n']:<5d} {f(v['mean_abs_error']):>6s} {f(v['mean_width']):>6s} {v['median_ess']:7.1f}")
print("\nPAIRED (mismatch):", json.dumps(out["paired_mismatch"], indent=1))
print("\nSTABILITY:")
for k, v in stab["cases"].items():
    b = v["by_bank"]; print(f"  {k:40s} {v['verdict']:24s} med A4k/8k/16k {b['A4000']['median']:+.3f}/{b['A8000']['median']:+.3f}/{b['A16000']['median']:+.3f}  B16k {b['B16000']['median']:+.3f} | ESS A {b['A4000']['ess']:.0f}/{b['A8000']['ess']:.0f}/{b['A16000']['ess']:.0f} | seed spread med {v['seed_spread_16000']['median']:.3f} lo {v['seed_spread_16000']['lo']:.3f} hi {v['seed_spread_16000']['hi']:.3f}")
for n in (4000, 16000):
    print(f"  sweat_only paired benefit, bank {n}:", json.dumps(stab[f'sweat_only_paired_benefit_bank{n}']))
