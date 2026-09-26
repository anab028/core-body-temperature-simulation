"""
v3 - second review round. Runs:
  0. oracle consistency check (true alpha(t)) + four-cell error decomposition
  1. forward sanity
  2. error propagation (bias/SD/RMSE), fan off & on, under the Gagge forward model
     AND under an independently specified 'linear' physiology (inverse still assumes Gagge)
  3. observation-consistent ensembles, CONTROLLED: same physiological samples and same
     observation noise across baseline treatments; nested acceptance windows; 5 seeds
  4. provisional accuracy sweep with a finer grid between 0.5 and 1.0 degC
Writes figures + summary.json (strict JSON, no NaN) into ./out
"""
import os, json
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from thermo import Body, make_protocol, make_env, simulate, reconstruct, FAMILIES

OUT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "out"); os.makedirs(OUT, exist_ok=True)
BLUE, ORANGE, AQUA, YELLOW = "#2a78d6", "#eb6834", "#1baf7a", "#eda100"
INK, MUTED = "#0b0b0b", "#52514e"
plt.rcParams.update({"font.size": 10, "axes.edgecolor": "#c9c8c3", "axes.labelcolor": INK,
                     "xtick.color": MUTED, "ytick.color": MUTED, "axes.spines.top": False,
                     "axes.spines.right": False, "figure.dpi": 130})

body = Body(); DT = 2.0; T_CR0 = 36.9
prot = make_protocol(dt=DT, P_ext=100.0, eff_net=0.20)
t_min = prot["t"] / 60.0; T_LEN = len(prot["t"])
I40 = int((600 + 1800) / DT)              # index of t = 40.0 min (inclusive slice is :I40+1)
TARGET = 0.20                             # RMSE target for point estimates ONLY (not ensemble width)
rng = np.random.default_rng(7)
summary = {"notes": [
    "Synthetic two-node study; no camera model, no human data.",
    "Forward 'reference' family shares the sweat/perfusion laws with the inverse -> not independent. 'sweat_only', 'perfusion_only', 'combined' are SYNTHETIC STRESS-TEST variants (bounded, same structural form; coefficients ours, not from data). The inverse always assumes the reference laws.",
    "'matched alpha' rows are ORACLE consistency checks using the true alpha(t); they are not deployable performance.",
    "All perturbations are Gaussian SD of a SESSION-CONSTANT error (not frame-to-frame noise). Sweat gain is lognormal with log-space SD 0.3 (mean ~1.05, SD ~0.32).",
    "'extrapolation error' = ASSUMED regional-extrapolation error on mean skin temperature, not a visibility model.",
    "'total-E constraint' = IDEALISED total evaporative-energy constraint, not a body-mass measurement.",
    "Ensemble spread is a rejection-sampling ensemble width; it is NOT a calibrated interval and is not compared to the RMSE target.",
]}

env_off = make_env(prot["t"], v=0.15); env_on = make_env(prot["t"], v=1.5)
envs = {"fan_off": env_off, "fan_on": env_on}
sims = {(fam, cond): simulate(prot, env, body, dt=DT, T_cr0=T_CR0, family=fam)
        for fam in FAMILIES for cond, env in envs.items()}


def truth_obs(sim, env):
    shp = sim["T_sk"].shape
    o = {"T_sk": sim["T_sk"], "M": np.broadcast_to(prot["M"], shp), "W": np.broadcast_to(prot["W"], shp)}
    for k in ["T_a", "T_r", "v", "RH"]:
        o[k] = np.broadcast_to(env[k], shp)
    return o


def stats(e):
    return {"bias": float(np.mean(e)), "sd": float(np.std(e)), "rmse": float(np.sqrt(np.mean(e ** 2)))}


# ---------------------------- 0. oracle check + decomposition (both families)
dec = {}
for (fam, cond), sim in sims.items():
    truth = sim["T_cr"] - sim["T_cr"][:, :1]
    for Em, al, lab in [("exact", sim["alpha"], "ORACLE: exact E, true alpha(t)"),
                        ("exact", 0.10, "exact E, fixed alpha=0.10"),
                        ("model", sim["alpha"], "ORACLE: reference-modelled E, true alpha(t)"),
                        ("model", 0.10, "reference-modelled E, fixed alpha=0.10  [deployable form]"),
                        ("none", 0.10, "E ignored, fixed alpha=0.10")]:
        d, _ = reconstruct(truth_obs(sim, envs[cond]), body, T_cr0=T_CR0, alpha=al, E_mode=Em,
                           E_exact=sim["E_sk"], dt=DT)
        e = d - truth
        dec[f"{fam} | {cond} | {lab}"] = {"err_40min": float(e[0, I40]),
                                          "max_abs_err": float(np.abs(e).max())}
summary["error_decomposition"] = dec
summary["forward"] = {f"{fam}_{cond}_dTcr_40min": float(s["T_cr"][0, I40] - T_CR0) for (fam, cond), s in sims.items()}

# ------------------------------------------------------------- 1. forward fig
fig, ax = plt.subplots(1, 2, figsize=(10, 3.6))
for fam, ls in [("reference", "-"), ("sweat_only", "--"), ("perfusion_only", ":"), ("combined", "-.")]:
    for cond, c in [("fan_off", BLUE), ("fan_on", ORANGE)]:
        ax[0].plot(t_min, sims[(fam, cond)]["T_cr"][0], color=c, lw=2 if ls == "-" else 1.2, ls=ls,
                   label=f"{fam}, {cond.replace('_', ' ')}")
ax[0].axvspan(10, 40, color="#f0efeb", zorder=0); ax[0].set_xlabel("time (min)"); ax[0].set_ylabel("core °C")
ax[0].set_title("Forward variants: 100 W cycling, 24 °C", loc="left"); ax[0].legend(frameon=False, fontsize=6, ncol=2)
s = sims[("reference", "fan_off")]
for k, c, lab in [("R", BLUE, "radiation R"), ("C", ORANGE, "convection C"), ("E_sk", AQUA, "evaporation E"), ("Resp", YELLOW, "respiratory")]:
    ax[1].plot(t_min, s[k][0], color=c, lw=2, label=lab)
ax[1].plot(t_min, prot["M"] - prot["W"], color=INK, lw=1, ls=":", label="net heat production")
ax[1].axvspan(10, 40, color="#f0efeb", zorder=0); ax[1].set_xlabel("time (min)"); ax[1].set_ylabel("W/m²")
ax[1].set_title("Heat-balance terms (reference, fan off)", loc="left"); ax[1].legend(frameon=False, fontsize=8)
fig.tight_layout(); fig.savefig(os.path.join(OUT, "fig1_forward.png")); plt.close(fig)


# ---------------------------------------------- observation model (assumed)
def observe(sim, env, n, T_sk_bias=0.0, extrap_err=0.0, T_r_err=0.0, v_err_frac=0.0, M_err_frac=0.0):
    """Session-constant Gaussian errors (SD given) applied to a truth run."""
    T = T_LEN; truth_T_sk = np.broadcast_to(sim["T_sk"], (n, T))
    b = rng.normal(0, T_sk_bias, (n, 1)) if T_sk_bias else 0.0
    cov = rng.normal(0, extrap_err, (n, 1)) if extrap_err else 0.0
    obs = {"T_sk": truth_T_sk + b + cov * (truth_T_sk - env["T_a"][None, :]),
           "M": np.broadcast_to(prot["M"], (n, T)) * (1 + (rng.normal(0, M_err_frac, (n, 1)) if M_err_frac else 0)),
           "W": np.broadcast_to(prot["W"], (n, T)), "T_a": np.broadcast_to(env["T_a"], (n, T)),
           "T_r": np.broadcast_to(env["T_r"], (n, T)) + (rng.normal(0, T_r_err, (n, 1)) if T_r_err else 0),
           "v": np.maximum(np.broadcast_to(env["v"], (n, T)) * (1 + (rng.normal(0, v_err_frac, (n, 1)) if v_err_frac else 0)), 0.05),
           "RH": np.broadcast_to(env["RH"], (n, T))}
    return obs


# ------------------------------------------------ 2. tornado, both families
N = 400
perturb = {"mean skin temp, SD 0.5 °C": dict(T_sk_bias=0.5), "mean skin temp, SD 2 °C": dict(T_sk_bias=2.0),
           "assumed regional extrapolation, SD 15 %": dict(extrap_err=0.15), "mean radiant temp, SD 1 °C": dict(T_r_err=1.0),
           "air velocity, SD 30 %": dict(v_err_frac=0.30), "metabolic cart, SD 5 %": dict(M_err_frac=0.05)}
torn = {}
for (fam, cond), sim in sims.items():
    if fam not in ("reference", "combined"):
        continue
    truth40 = sim["T_cr"][0, I40] - T_CR0; rows = {}
    for name, kw in perturb.items():
        dT, _ = reconstruct(observe(sim, envs[cond], N, **kw), body, T_cr0=T_CR0, E_mode="model", dt=DT)
        rows[name] = stats(dT[:, I40] - truth40)
    gs = rng.lognormal(0, 0.3, N)
    sim_g = simulate(prot, envs[cond], body, gains={"sweat": gs}, dt=DT, T_cr0=T_CR0, family=fam)
    dT, _ = reconstruct(truth_obs(sim_g, envs[cond]), body, T_cr0=T_CR0, E_mode="model", dt=DT)
    rows["sweat gain, log-SD 0.3 (E unobserved)"] = stats(dT[:, I40] - (sim_g["T_cr"][:, I40] - T_CR0))
    dT0, _ = reconstruct(truth_obs(sim, envs[cond]), body, T_cr0=T_CR0, E_mode="model", dt=DT)
    rows["(unperturbed model-mismatch error)"] = stats(dT0[:, I40] - truth40)
    torn[f"{fam}_{cond}"] = rows
summary["tornado_40min"] = torn

fig, axes = plt.subplots(2, 2, figsize=(12, 8.2), sharey=True)
order = sorted([k for k in torn["reference_fan_off"] if not k.startswith("(")], key=lambda k: torn["reference_fan_off"][k]["rmse"])
for ax, key, ttl in [(axes[0, 0], "reference_fan_off", "reference forward, fan off"), (axes[0, 1], "reference_fan_on", "reference forward, fan on"),
                     (axes[1, 0], "combined_fan_off", "combined stress-test forward, fan off"), (axes[1, 1], "combined_fan_on", "combined stress-test forward, fan on")]:
    r = [torn[key][k]["rmse"] for k in order]; bb = [abs(torn[key][k]["bias"]) for k in order]
    ax.barh(order, r, color=[ORANGE if v > TARGET else BLUE for v in r], height=0.55, label="RMSE")
    ax.scatter(bb, range(len(order)), color=INK, s=14, zorder=3, label="|bias|")
    ax.axvline(TARGET, color=INK, lw=1, ls=":")
    mm = torn[key]["(unperturbed model-mismatch error)"]["rmse"]
    ax.axvline(mm, color=AQUA, lw=1.2, ls="--")
    for i, v in enumerate(r):
        ax.text(v + 0.005, i, f"{v:.2f}", va="center", fontsize=8, color=MUTED)
    ax.set_title(f"{ttl}\nerror with perfect observed inputs {mm:.2f} °C (dashed) — not a floor", loc="left", fontsize=9)
    ax.set_xlabel("error in ΔT_core at 40 min (°C)")
axes[0, 0].legend(frameon=False, fontsize=8, loc="lower right")
fig.suptitle("Error propagation per assumed session-constant error (Gaussian SD, one-at-a-time, 100 W); inverse assumes reference laws",
             x=0.01, ha="left", fontsize=10)
fig.tight_layout(); fig.savefig(os.path.join(OUT, "fig2_tornado.png")); plt.close(fig)


# --------------------------- 3. controlled ensembles, nested windows, 5 seeds
CAM = 0.3; M_S = 4000; TOLS = [1.05, 1.25, 1.5, 2.0]; SEEDS = [11, 22, 33, 44, 55]
truth_g = {"sweat": 1.3, "bloodflow": 0.9, "hc": 1.1}


def spread(vals, mask):
    if mask.sum() < 20:
        return None
    return float(np.percentile(vals[mask], 97.5) - np.percentile(vals[mask], 2.5))


def run_ensemble(env, seed):
    """One seed: SAME prior samples and SAME observation noise for all baseline treatments;
    only T_cr0 differs. Nested acceptance: exercise window, then exercise AND recovery."""
    r = np.random.default_rng(seed)
    truth = simulate(prot, env, body, gains=truth_g, dt=DT, T_cr0=T_CR0)
    T_obs = truth["T_sk"][0] + r.normal(0, CAM, T_LEN)
    prior = {"sweat": r.lognormal(0, 0.45, M_S), "bloodflow": r.lognormal(0, 0.3, M_S), "hc": r.lognormal(0, 0.2, M_S)}
    z = r.normal(0, 1, M_S); u = r.uniform(0, 1, M_S)          # shared draws, transformed per treatment
    E_true = truth["E_sk"][0, :-1].sum() * DT
    res = {"truth_dTcr_40min": float(truth["T_cr"][0, I40] - T_CR0)}
    for bmode, T0 in [("known", np.full(M_S, T_CR0)), ("uncertain_sd0.3", T_CR0 + 0.3 * z),
                      ("unknown_uniform", 36.2 + 1.4 * u)]:
        cand = simulate(prot, env, body, gains=prior, dt=DT, T_cr0=T0)
        d40 = cand["T_cr"][:, I40] - cand["T_cr"][:, 0]
        rms_ex = np.sqrt(np.mean((cand["T_sk"][:, :I40 + 1] - T_obs[None, :I40 + 1]) ** 2, axis=1))
        rms_rec = np.sqrt(np.mean((cand["T_sk"][:, I40 + 1:] - T_obs[None, I40 + 1:]) ** 2, axis=1))
        E_c = cand["E_sk"][:, :-1].sum(1) * DT
        res[bmode] = {}
        for tol in TOLS:
            a_ex = rms_ex < tol * CAM
            a_both = a_ex & (rms_rec < tol * CAM)              # nested by construction
            a_E = a_both & (np.abs(E_c - E_true) / E_true < 0.10)
            res[bmode][str(tol)] = {
                "exercise_window": {"n": int(a_ex.sum()), "spread95": spread(d40, a_ex)},
                "exercise_and_recovery": {"n": int(a_both.sum()), "removed_by_recovery": int((a_ex & ~a_both).sum()),
                                          "spread95": spread(d40, a_both)},
                "plus_idealised_totalE": {"n": int(a_E.sum()), "spread95": spread(d40, a_E)}}
    return res


ens = {}
for cond, env in envs.items():
    ens[cond] = {"seeds": {str(sd): run_ensemble(env, sd) for sd in SEEDS}}
    # aggregate across seeds
    agg = {}
    for bmode in ["known", "uncertain_sd0.3", "unknown_uniform"]:
        agg[bmode] = {}
        for tol in TOLS:
            agg[bmode][str(tol)] = {}
            for win in ["exercise_window", "exercise_and_recovery", "plus_idealised_totalE"]:
                vals = [ens[cond]["seeds"][str(sd)][bmode][str(tol)][win]["spread95"] for sd in SEEDS]
                ok = [v for v in vals if v is not None]
                agg[bmode][str(tol)][win] = ({"mean": float(np.mean(ok)), "min": float(min(ok)), "max": float(max(ok)),
                                              "n_seeds_ok": len(ok)} if len(ok) == len(SEEDS)
                                             else {"status": "insufficient accepted samples in >=1 seed", "n_seeds_ok": len(ok)})
    ens[cond]["aggregate"] = agg
summary["ensembles"] = ens

fig, axes = plt.subplots(1, 2, figsize=(11, 3.9), sharey=True)
for ax, cond in zip(axes, envs):
    agg = ens[cond]["aggregate"]
    for bmode, c in [("known", BLUE), ("uncertain_sd0.3", ORANGE), ("unknown_uniform", AQUA)]:
        for win, ls, lab in [("exercise_window", "-", "exercise window"), ("exercise_and_recovery", "--", "+ recovery (nested)")]:
            xs, ys, lo, hi = [], [], [], []
            for tol in TOLS:
                a = agg[bmode][str(tol)][win]
                if "mean" in a:
                    xs.append(tol); ys.append(a["mean"]); lo.append(a["min"]); hi.append(a["max"])
            if xs:
                ax.plot(xs, ys, color=c, ls=ls, lw=2, marker="o", ms=4, label=f"baseline {bmode.replace('_', ' ')}, {lab}")
                ax.fill_between(xs, lo, hi, color=c, alpha=0.12, lw=0)
    tr = ens[cond]["seeds"][str(SEEDS[0])]["truth_dTcr_40min"]
    ax.set_title(f"{cond.replace('_', ' ')} — truth ΔT_core(40 min) = {tr:.2f} °C", loc="left", fontsize=10)
    ax.set_xlabel("acceptance tolerance (× skin-temp noise SD)")
axes[0].set_ylabel("95 % ensemble width of ΔT_core at 40 min (°C)\nmean over 5 seeds, band = min–max")
axes[1].legend(frameon=False, fontsize=7, loc="upper left")
fig.suptitle("Observation-consistent ensembles — controlled (same samples & noise per seed; only baseline differs). Width ≠ calibrated interval.",
             x=0.01, ha="left", fontsize=9)
fig.tight_layout(); fig.savefig(os.path.join(OUT, "fig3_ensembles.png")); plt.close(fig)


# ------------------------------------------- 4. accuracy sweep, finer grid
biases = np.array([0.0, 0.25, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0, 1.5, 2.0, 3.0, 5.0]); extraps = [0.0, 0.10, 0.25]
sim0 = sims[("reference", "fan_off")]; truth40 = sim0["T_cr"][0, I40] - T_CR0
sweep = {}
for ex in extraps:
    sweep[ex] = []
    for b in biases:
        dT, _ = reconstruct(observe(sim0, env_off, 300, T_sk_bias=b, extrap_err=ex), body, T_cr0=T_CR0, E_mode="model", dt=DT)
        sweep[ex].append(stats(dT[:, I40] - truth40))
largest_pass = max([b for b, s in zip(biases, sweep[0.0]) if s["rmse"] < TARGET], default=None)
summary["accuracy_sweep_reference_fan_off"] = {"largest_sampled_bias_SD_with_rmse_below_target_no_extrap": float(largest_pass),
                                           "grid": {str(ex): {str(b): s for b, s in zip(biases, v)} for ex, v in sweep.items()}}
fig, ax = plt.subplots(figsize=(7.5, 3.8))
for ex, c, lab in [(0.0, BLUE, "no extrapolation error"), (0.10, ORANGE, "assumed extrapolation error SD 10 %"), (0.25, AQUA, "assumed extrapolation error SD 25 %")]:
    ax.plot(biases, [s["rmse"] for s in sweep[ex]], color=c, lw=2, marker="o", ms=4, label=lab)
ax.axhline(TARGET, color=INK, lw=1, ls=":"); ax.text(5, TARGET, f" RMSE target {TARGET} °C", fontsize=8, color=MUTED, va="bottom", ha="right")
ax.set_xlabel("session-constant mean skin-temperature error, Gaussian SD (°C)"); ax.set_ylabel("RMSE of ΔT_core at 40 min (°C)")
ax.set_title("Provisional accuracy requirement (assumed error models, reference, fan off, 100 W)", loc="left")
ax.legend(frameon=False, fontsize=8, loc="center right")
fig.tight_layout(); fig.savefig(os.path.join(OUT, "fig4_accuracy_sweep.png")); plt.close(fig)




# ============ 5. misspecification experiments (v6)
MS_SEEDS = [11, 22, 33]; TOL = 1.5; TOL_DETECT = 1.25     # TOL_DETECT fixed a priori
GAIN_PRIOR = lambda r, m: {"sweat": r.lognormal(0, 0.45, m), "bloodflow": r.lognormal(0, 0.3, m), "hc": r.lognormal(0, 0.2, m)}


def phys_range(sim, idx=None):
    sl = slice(None) if idx is None else idx
    return {"alpha_max": float(sim["alpha"][sl].max()), "m_bl_min": float(sim["m_bl"][sl].min()),
            "peak_evaporated_water_equivalent_L_h": float(sim["E_sk"][sl].max() * body.A_D / 2.43e6 * 3600),
            "peak_secreted_sweat_L_h": float(sim["m_rsw"][sl].max() * body.A_D * 3600)}


def window_rms(cand_T_sk, T_obs):
    ex = np.sqrt(np.mean((cand_T_sk[:, :I40 + 1] - T_obs[None, :I40 + 1]) ** 2, axis=1))
    rec = np.sqrt(np.mean((cand_T_sk[:, I40 + 1:] - T_obs[None, I40 + 1:]) ** 2, axis=1))
    return ex, rec


def same_candidate_stat(ex, rec):
    """Detection statistic: the single candidate that best explains BOTH windows (min over
    candidates of max(RMS_ex, RMS_rec)), in units of the noise SD."""
    return float(np.min(np.maximum(ex, rec)) / CAM)


def adaptive_search(env, T_obs, gains0, rounds=4, m=1500, shrink=0.5, seed=0, restarts=3):
    """Cross-entropy-style refinement in log-gain space, `restarts` independent runs from
    the best prior draws. Objective = max(RMS_ex, RMS_rec) of the SAME candidate.
    Returns best score (x noise), best gains, window errors and physiological range."""
    best = None
    for rs in range(restarts):
        r = np.random.default_rng(seed * 100 + rs)
        lg = np.log(np.stack([gains0["sweat"], gains0["bloodflow"], gains0["hc"]], 1))
        lg = lg[r.integers(0, len(lg), m)] + r.normal(0, 1, (m, 3)) * np.array([0.3, 0.2, 0.15])
        sd = np.array([0.45, 0.3, 0.2])
        for k in range(rounds):
            g = {"sweat": np.exp(lg[:, 0]), "bloodflow": np.exp(lg[:, 1]), "hc": np.exp(lg[:, 2])}
            c = simulate(prot, env, body, gains=g, dt=DT, T_cr0=T_CR0, family="reference")
            ex, rec = window_rms(c["T_sk"], T_obs); score = np.maximum(ex, rec) / CAM
            top = np.argsort(score)[:50]; b = top[0]
            if best is None or score[b] < best["score"]:
                best = {"score": float(score[b]), "gains": {kk: float(v[b]) for kk, v in g.items()},
                        "rms_ex_over_noise": float(ex[b] / CAM), "rms_rec_over_noise": float(rec[b] / CAM),
                        "phys_range": phys_range(c, b), "restart": rs, "round": k}
            sd = sd * shrink
            lg = lg[top][r.integers(0, 50, m)] + r.normal(0, 1, (m, 3)) * sd
    return best


def status_of(n_acc, searched):
    if n_acc >= 20:
        return "ensemble ok"
    if n_acc > 0:
        return "insufficient ensemble size (fits exist)"
    return ("no acceptable fit found within the tested search procedure" if searched["score"] > TOL
            else "no sampled fit (search found a fit)")


cand_cache = {}
for sd in MS_SEEDS:
    r = np.random.default_rng(sd); prior = GAIN_PRIOR(r, M_S)
    for cond, env in envs.items():
        c = simulate(prot, env, body, gains=prior, dt=DT, T_cr0=T_CR0, family="reference")
        cand_cache[(sd, cond)] = {"prior": prior, "sim": c, "range": phys_range(c), "d40": c["T_cr"][:, I40] - c["T_cr"][:, 0]}


def ens_summary(d40, acc, truth40):
    if acc.sum() < 20:
        return {"n": int(acc.sum())}
    lo, hi = np.percentile(d40[acc], [2.5, 97.5])
    return {"n": int(acc.sum()), "range95": [float(lo), float(hi)], "width": float(hi - lo),
            "median_error": float(np.median(d40[acc]) - truth40), "truth_inside_range95": bool(lo <= truth40 <= hi)}


# ---- 5a. matched comparison: causal point estimate | causal ensemble (data to 40 min) | retrospective ensemble
mis = {}
for fam in FAMILIES:
    for cond, env in envs.items():
        key = f"{fam}_{cond}"; mis[key] = {"seeds": {}}
        truth = simulate(prot, env, body, gains=truth_g, dt=DT, T_cr0=T_CR0, family=fam)
        truth40 = float(truth["T_cr"][0, I40] - T_CR0); mis[key]["truth_phys_range"] = phys_range(truth)
        for sd in MS_SEEDS:
            r = np.random.default_rng(1000 + sd); T_obs = truth["T_sk"][0] + r.normal(0, CAM, T_LEN)
            obs = {"T_sk": T_obs[None, :], "M": prot["M"][None, :], "W": prot["W"][None, :]}
            for k in ["T_a", "T_r", "v", "RH"]:
                obs[k] = env[k][None, :]
            dT_pt, _ = reconstruct(obs, body, T_cr0=T_CR0, E_mode="model", dt=DT)
            cc = cand_cache[(sd, cond)]; ex, rec = window_rms(cc["sim"]["T_sk"], T_obs)
            acc_c = ex < TOL * CAM                              # causal: exercise window only
            acc_r = acc_c & (rec < TOL * CAM)                   # retrospective: nested
            row = {"truth_dTcr_40min": truth40, "point_error_causal": float(dT_pt[0, I40] - truth40),
                   "ensemble_causal_to40": ens_summary(cc["d40"], acc_c, truth40),
                   "ensemble_retrospective_ex_rec": ens_summary(cc["d40"], acc_r, truth40),
                   "same_candidate_stat_over_noise": same_candidate_stat(ex, rec),
                   "min_rms_ex_over_noise": float(ex.min() / CAM), "min_rms_rec_over_noise": float(rec.min() / CAM)}
            searched = None
            if acc_r.sum() == 0:
                best = np.argsort(np.maximum(ex, rec))[:50]
                searched = adaptive_search(env, T_obs, {k: v[best] for k, v in cc["prior"].items()}, seed=sd)
                row["search"] = searched
            row["status_retrospective"] = status_of(int(acc_r.sum()), searched)
            mis[key]["seeds"][str(sd)] = row
summary["misspecified_truth_matched"] = mis
summary["candidate_phys_ranges"] = {f"seed{sd}_{cond}": v["range"] for (sd, cond), v in cand_cache.items()}

# ---- 5b. joint fit (shared gains, both airflow conditions)
joint = {}
for fam in FAMILIES:
    joint[fam] = {"seeds": {}}
    tr = {cond: simulate(prot, env, body, gains=truth_g, dt=DT, T_cr0=T_CR0, family=fam) for cond, env in envs.items()}
    for sd in MS_SEEDS:
        r = np.random.default_rng(2000 + sd); acc_all = np.ones(M_S, bool); per = {}
        for cond in envs:
            T_obs = tr[cond]["T_sk"][0] + r.normal(0, CAM, T_LEN)
            ex, rec = window_rms(cand_cache[(sd, cond)]["sim"]["T_sk"], T_obs)
            a = (ex < TOL * CAM) & (rec < TOL * CAM); per[cond] = int(a.sum()); acc_all &= a
        nj = int(acc_all.sum())
        row = {"n_accepted_single": per, "n_accepted_joint": nj,
               "status": ("joint ensemble ok" if nj >= 20 else
                          "fits exist; insufficient joint ensemble size" if nj > 0 else
                          "no sampled joint fit (no joint search performed)")}
        for cond in envs:
            truth40 = float(tr[cond]["T_cr"][0, I40] - T_CR0)
            row[cond] = {"truth": truth40, **ens_summary(cand_cache[(sd, cond)]["d40"], acc_all, truth40)}
        joint[fam]["seeds"][str(sd)] = row
summary["joint_fit_shared_params"] = joint

# ---- 5c. detection calibration, same-candidate statistic, PAIRED lambda sweep
det = {"threshold_fixed_a_priori": TOL_DETECT, "statistic": "min over candidates of max(RMS_ex, RMS_rec) / noise (same candidate)",
       "what_a_flag_means": "inconsistency between the observations and the assumed model + measurement setup, evaluated on a fixed 4000-candidate bank (adaptive search is NOT part of the calibration); causes can be model mismatch, insufficient search, or sensor bias",
       "false_alarm": {}, "paired_lambda_sweep": {}, "bias_confound": {}}
r = np.random.default_rng(999)
N_FA, N_PAIR = 30, 20
for cond, env in envs.items():
    g_ref = GAIN_PRIOR(r, N_FA); tr = simulate(prot, env, body, gains=g_ref, dt=DT, T_cr0=T_CR0, family="reference")
    noise = r.normal(0, CAM, (N_FA, T_LEN)); st = []
    for i in range(N_FA):
        ex, rec = window_rms(cand_cache[(11, cond)]["sim"]["T_sk"], tr["T_sk"][i] + noise[i]); st.append(same_candidate_stat(ex, rec))
    st = np.array(st)
    det["false_alarm"][cond] = {"n_tested": N_FA, "n_flagged": int((st > TOL_DETECT).sum()), "stat_median": float(np.median(st)), "stat_max": float(st.max())}
    # paired: SAME 20 gain sets and SAME noise realisations at every lambda (lambda=0 is the paired control)
    g_p = GAIN_PRIOR(r, N_PAIR); noise_p = r.normal(0, CAM, (N_PAIR, T_LEN)); det["paired_lambda_sweep"][cond] = {}
    for lam in [0.0, 0.25, 0.5, 0.75, 1.0]:
        tr = simulate(prot, env, body, gains=g_p, dt=DT, T_cr0=T_CR0, family="reference", sweat_mix=lam); st = []
        for i in range(N_PAIR):
            ex, rec = window_rms(cand_cache[(11, cond)]["sim"]["T_sk"], tr["T_sk"][i] + noise_p[i]); st.append(same_candidate_stat(ex, rec))
        st = np.array(st); k = int((st > TOL_DETECT).sum())
        det["paired_lambda_sweep"][cond][str(lam)] = {"n": N_PAIR, "n_flagged": k, "rate": k / N_PAIR, "stat_median": float(np.median(st)),
                                                    "stats": [float(x) for x in st]}
    det["bias_confound"][cond] = {}
    tr = simulate(prot, env, body, gains=truth_g, dt=DT, T_cr0=T_CR0, family="reference"); nb = r.normal(0, CAM, T_LEN)
    for b in [0.0, 0.5, 1.0, 2.0]:
        ex, rec = window_rms(cand_cache[(11, cond)]["sim"]["T_sk"], tr["T_sk"][0] + b + nb); stt = same_candidate_stat(ex, rec)
        det["bias_confound"][cond][f"skin_bias_{b}"] = {"stat": stt, "flagged": bool(stt > TOL_DETECT)}
summary["detection_calibration"] = det

# ---- 5d. population check: point estimator & causal ensemble over 20 held-out reference physiologies
pop = {}
r = np.random.default_rng(4242)
for cond, env in envs.items():
    g = GAIN_PRIOR(r, 20); tr = simulate(prot, env, body, gains=g, dt=DT, T_cr0=T_CR0, family="reference")
    pe, me = [], []
    for i in range(20):
        T_obs = tr["T_sk"][i] + r.normal(0, CAM, T_LEN); truth40 = float(tr["T_cr"][i, I40] - T_CR0)
        obs = {"T_sk": T_obs[None, :], "M": prot["M"][None, :], "W": prot["W"][None, :]}
        for k in ["T_a", "T_r", "v", "RH"]:
            obs[k] = env[k][None, :]
        dT_pt, _ = reconstruct(obs, body, T_cr0=T_CR0, E_mode="model", dt=DT); pe.append(float(dT_pt[0, I40] - truth40))
        ex, _ = window_rms(cand_cache[(11, cond)]["sim"]["T_sk"], T_obs); a = ex < TOL * CAM
        me.append(float(np.median(cand_cache[(11, cond)]["d40"][a]) - truth40) if a.sum() >= 20 else None)
    me_ok = [x for x in me if x is not None]
    pop[cond] = {"n": 20, "point_error": {"mean": float(np.mean(pe)), "sd": float(np.std(pe)), "rmse": float(np.sqrt(np.mean(np.square(pe))))},
                 "ensemble_causal_median_error": {"n_with_ensemble": len(me_ok), "mean": float(np.mean(me_ok)), "sd": float(np.std(me_ok))}}
summary["population_reference_check"] = pop

# ---- figure 5
fig, axes = plt.subplots(1, 2, figsize=(11.5, 4.6), sharey=True)
fams = list(FAMILIES)
for ax, cond in zip(axes, envs):
    for j, fam in enumerate(fams):
        for k, (sd, c) in enumerate(zip(MS_SEEDS, [BLUE, AQUA, YELLOW])):
            row = mis[f"{fam}_{cond}"]["seeds"][str(sd)]; y = j + (k - 1) * 0.24
            ec, er = row["ensemble_causal_to40"], row["ensemble_retrospective_ex_rec"]
            if "range95" in ec:
                ax.plot(ec["range95"], [y + 0.05, y + 0.05], color=c, lw=2.5, alpha=0.45, solid_capstyle="butt")
            if "range95" in er:
                ax.plot(er["range95"], [y - 0.05, y - 0.05], color=c, lw=4, solid_capstyle="butt")
                ax.text(max(er["range95"][1], ec.get("range95", [0, 0])[1]) + 0.01, y, f"n={ec.get('n', 0)}/{er['n']}", va="center", fontsize=6, color=MUTED)
            else:
                ax.text(row["truth_dTcr_40min"] + 0.03, y, row["status_retrospective"].replace(" within", "\nwithin").replace(" (", "\n("), va="center", fontsize=5.5, color=ORANGE)
            ax.plot(row["truth_dTcr_40min"], y, marker="|", ms=12, mew=2, color=INK)
            ax.plot(row["truth_dTcr_40min"] + row["point_error_causal"], y, marker="D", ms=4, color=ORANGE, mec=INK, mew=0.5)
    ax.set_yticks(range(len(fams))); ax.set_yticklabels(fams); ax.margins(x=0.3); ax.set_xlabel("ΔT_core at 40 min (°C)")
    ax.set_title(cond.replace('_', ' '), loc="left", fontsize=10)
fig.suptitle("Matched comparison under misspecified truth — thin bar: causal ensemble (data to 40 min) | thick bar: retrospective (exercise+recovery)\n"
             "tick: truth | diamond: causal point estimate on the same observations | n = causal/retrospective accepted (tol 1.5×, baseline known)",
             x=0.01, ha="left", fontsize=8)
fig.tight_layout(); fig.savefig(os.path.join(OUT, "fig5_misspecified.png")); plt.close(fig)

with open(os.path.join(OUT, "summary.json"), "w") as f:
    json.dump(summary, f, indent=2, allow_nan=False)

# console digest
print("DECOMPOSITION (err at 40 min):")
for k, v in dec.items():
    print(f"  {k:75s} {v['err_40min']:+.4f}  max {v['max_abs_err']:.1e}")
print("\nTORNADO RMSE:")
for key in torn:
    print(" ", key, {k[:28]: round(v["rmse"], 3) for k, v in torn[key].items()})
print("\nENSEMBLE WIDTH mean[min,max] over seeds:")
for cond in ens:
    for b in ["known", "uncertain_sd0.3", "unknown_uniform"]:
        for tol in ["1.25", "1.5"]:
            row = ens[cond]["aggregate"][b][tol]
            def fmt(a): return f"{a['mean']:.2f}[{a['min']:.2f},{a['max']:.2f}]" if "mean" in a else "n/a"
            print(f"  {cond:8s} {b:16s} tol{tol}: ex {fmt(row['exercise_window'])}  +rec {fmt(row['exercise_and_recovery'])}  +E {fmt(row['plus_idealised_totalE'])}")
print("\nlargest sampled skin-bias SD with RMSE<target (no extrap):", largest_pass)

print("\nMATCHED: truth | point(causal) | ens causal med.err/n | ens retro med.err/n | status | same-cand stat")
for key, v in mis.items():
    for sd, row in v["seeds"].items():
        ec, er = row["ensemble_causal_to40"], row["ensemble_retrospective_ex_rec"]
        f = lambda e: f"{e['median_error']:+.2f}/{e['n']}" if "median_error" in e else f"  -  /{e['n']}"
        extra = f" search {row['search']['score']:.2f} gains {row['search']['gains']}" if "search" in row else ""
        print(f"  {key:24s} s{sd}: {row['truth_dTcr_40min']:+.2f} | {row['point_error_causal']:+.2f} | {f(ec):12s} | {f(er):12s} | {row['status_retrospective'][:40]:40s} | {row['same_candidate_stat_over_noise']:.2f}{extra}")
print("\nJOINT:", json.dumps({f: {sd: {"n_joint": r["n_accepted_joint"], "status": r["status"], **{c: {k: r[c][k] for k in r[c] if k in ("median_error", "width")} for c in envs}} for sd, r in v["seeds"].items()} for f, v in joint.items()}, indent=1))
print("\nDETECTION:", json.dumps({k: v for k, v in det.items() if k != "paired_lambda_sweep"}, indent=1))
print("PAIRED LAMBDA:", {c: {l: (v["n_flagged"], v["n"]) for l, v in d.items()} for c, d in det["paired_lambda_sweep"].items()})
print("\nPOPULATION:", json.dumps(pop, indent=1))
