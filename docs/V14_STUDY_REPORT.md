# Core-temperature reconstruction — synthetic feasibility study (v14, final: frozen v12 simulation baseline + sensor-error likelihood with held-out coverage and numerical-stability checks)

Two-node thermoregulation forward model + partitional-calorimetry inverse. **Synthetic only**:
no camera model, no learning, no human data. Point estimator and candidate ensemble both assume
the *reference* laws; the point estimator additionally uses fixed α = 0.10 and unit sweat gain.

## Run
```
pip install numpy matplotlib
python run_analysis.py     # frozen v6 baseline, ~3 min -> out/fig1..fig5 + out/summary.json
python run_ambiguity.py    # ambiguity experiment v2, ~2 min -> out/fig6_ambiguity.png + out/ambiguity_results.json
python run_mechanism.py    # paired robustness + mechanism + noise, ~2 min -> out/fig7_mechanism.png, out/fig8_decomposition.png, out/mechanism_results.json
python run_calibrated.py   # sensor-error likelihood, held-out coverage, numerical stability, ~3 min -> out/fig9_calibrated.png, out/calibrated_results.json
```

## Baseline corrections applied at freeze (sixth review round)
- A detector flag means *inconsistency between the observations and the assumed model + measurement
  setup*, evaluated on a fixed 4000-candidate bank; adaptive search is not part of the calibration.
  A flag can be caused by model mismatch, insufficient search, or sensor bias (the +1 °C bias case).
- "RMSE dominated by unknown sweat gain" → the population experiment varies sweat, blood-flow and
  convection gains together; attributing the spread to sweat alone needs a paired one-factor ablation
  (not done).
- "Monotonic once paired" → the aggregate flag count was monotonic in this run; individual statistics
  were not always (one fan-on participant reverses). Not a guarantee.
- Search-found gains of 9–13 are "far from the assumed prior", not "implausible" — a physiological
  judgement would need empirical sweat-output evidence.
- Joint-fit status bug fixed: 1–19 accepted → "fits exist; insufficient joint ensemble size".

## Scenario
70 kg, 1.8 m², 24 °C / 40 % RH / MRT 24 °C. 10 min rest → 30 min at 100 W (net efficiency 0.20)
→ 15 min recovery. Fan off 0.15 m/s, fan on 1.5 m/s. Baseline core 36.9 °C explicit. Observation:
mean skin temperature with 0.3 °C RMS noise. RMSE target 0.20 °C applies to point estimates only.

## Forward-model variants (synthetic stress tests — exact equations in `thermo.py`)
reference (Gagge/ASHRAE) · sweat_only (linear sweat drive) · perfusion_only (bounded alt perfusion)
· combined. A blend parameter λ interpolates the sweat law (0 = reference, 1 = sweat_only).
Physiological-range diagnostics (α_max, m_bl_min, **peak_evaporated_water_equivalent_L_h** and
**peak_secreted_sweat_L_h** from the pre-cap m_rsw) are saved for every truth, candidate batch and
search result.

## v6 changes (fourth review round)
- **Detector statistic corrected** to the same candidate: `min over candidates of max(RMS_ex, RMS_rec)`
  (previously mixed the best exercise fit of one candidate with the best recovery fit of another).
- **Paired λ sweep**: the same 20 gain sets and the same noise realisations at every λ, with λ = 0 as
  the paired control.
- **Causal vs retrospective** separated: point estimate at 40 min (causal), ensemble accepted on
  data to 40 min (causal), ensemble accepted on exercise + recovery (retrospective).
- Adaptive search: 3 independent restarts; best gains, window errors and physiological range saved.
  Wording: "no acceptable fit found within the tested search procedure". Joint zero-fit cases are
  labelled "no sampled joint fit (no joint search performed)".
- Population check on 20 held-out reference physiologies.
- False-alarm wording: "0 of 30 tested cases flagged", not "rate zero".

## Results

**Matched comparison (fig 5; one reference-family participant, gains 1.3/0.9/1.1, 3 noise/candidate seeds):**
| truth | cond | point (causal) | ensemble causal (to 40 min) | ensemble retrospective | status |
|---|---|---|---|---|---|
| reference | off | +0.29–0.31 | +0.05–0.06 (n 1795–1848) | +0.01 (n 924–938) | ok |
| reference | on | +0.24–0.27 | +0.00–0.01 (n 609–613) | −0.01 (n 392–414) | ok |
| sweat_only | off | +0.81–0.83 | n = 5–9 (insufficient) | n = 0 | no acceptable fit found (search 1.65–1.66×; best gains sweat ≈ 1.9–2.1, blood flow ≈ 0.7) |
| sweat_only | on | +1.16–1.19 | n = 0–1 | n = 0 | **search found a fit** (1.24–1.31×) — but only with blood-flow gain 0.29–0.35 |
| perfusion_only | off | +0.29–0.31 | **−0.19 to −0.21** (n 246–275) | −0.20 to −0.21 (n 173–198) | ok |
| perfusion_only | on | +0.24–0.28 | **−0.15** (n 90–103) | −0.17 to −0.20 (n 60–64) | ok |
| combined | off | +0.74–0.77 | n = 0–4 | n = 0 | no acceptable fit found (search 1.78–1.91×; best sweat gain 9–13, far from the assumed prior) |
| combined | on | +2.17–2.20 | n = 0 | n = 0 | no acceptable fit found (search 6.7–7.9×) |

For this reference physiology the ensemble median error was within ±0.06 (causal) and ±0.01
(retrospective) across three seeds — this is one participant, not population-level unbiasedness.
The perfusion-mismatch underestimate (≈ −0.2 °C) is present in the **causal** ensemble too, so it is
not an artefact of using recovery data. The sweat_only fan-on "fit" found by the search requires a
skin-blood-flow gain of ≈ 0.3 — a fit exists in the reference family, but at the edge of the
physiological prior; that is a diagnostic in itself.

**Joint fit (shared gains, fan off + fan on):** reference — joint accepts 189–227, widths 0.24–0.29 /
0.20–0.23, median error ≤ 0.02. perfusion_only — joint accepts 41–61 and the underestimate persists
(−0.22 to −0.27 off, −0.15 to −0.19 on). sweat_only / combined — no sampled joint fit (no joint search
performed).

**Detection calibration (same-candidate statistic, threshold 1.25× fixed a priori, prior batch of 4000):**
false alarms: 0 of 30 tested reference cases flagged in each condition (max statistic 1.06 / 1.09).
Paired λ sweep (20 participants, same gains and noise at every λ):
| λ | 0 (control) | 0.25 | 0.5 | 0.75 | 1.0 |
|---|---|---|---|---|---|
| fan off, flagged / 20 | 0 | 0 | 4 | 7 | 9 |
| fan on, flagged / 20 | 0 | 1 | 2 | 5 | 7 |

Aggregate counts are monotonic in this run once paired and same-candidate (the v5 non-monotonicity
was the statistic bug plus unpaired sampling); individual statistics are not always monotonic. Sensitivity at full mismatch is 45 % / 35 %. **Skin-bias confound persists**:
a session-constant 1 °C bias on a reference physiology is flagged (1.42× / 1.51×); 0.5 °C is not.

**Population check (20 held-out reference physiologies, causal):** point estimator bias +0.09 / +0.05,
SD 0.33 / 0.31, RMSE 0.35 / 0.31 (fan off / on) — the spread reflects between-person gain
variability (sweat, blood flow and convection varied together) that the fixed-gain inverse cannot see. Causal ensemble median error: mean −0.005 / −0.04,
SD 0.10 / 0.07.

**Earlier results (unchanged):** oracle bookkeeping 1e-13; decomposition (nominal gains) sweat_only
+0.63 / +0.17, perfusion_only +0.05 / +0.00, combined +0.58 / +1.54; tornado ranking; controlled
baseline ensembles; accuracy sweep 0.8 °C pass / 0.9 °C fail.

## Claims table
| claim | status |
|---|---|
| Inverse bookkeeping exact (oracle) | supported |
| Point estimator RMSE on held-out reference physiologies | ≈ 0.3 °C (n = 20); sweat, blood-flow and convection gains all varied — attribution to sweat gain would need a paired one-factor ablation |
| Ensemble median error ≈ 0 under reference truth | supported for this participant (3 seeds) and, causally, across 20 physiologies (mean −0.005 / −0.04) |
| Ensemble systematically underestimates under the tested perfusion mismatch | supported (−0.2 °C; causal and retrospective; single and joint) |
| Sweat-law mismatch detectable from skin fit | partial — fan off: no fit found within the tested search; fan on: fit found at blood-flow gain ≈ 0.3; paired detection 45 % / 35 % at λ = 1 |
| Detector separates physiology mismatch from measurement bias | not supported — 1 °C skin bias is flagged |
| Joint fan-off/on fit exposes perfusion mismatch | not supported |
| Combined mismatch unfittable | reworded: no acceptable fit found within the tested search procedure |
| Recovery data improves the ensemble | for reference truth: median error +0.06 → +0.01 (fan off); does not remove the perfusion bias |

## Ambiguity-reduction experiment, v2 (fig 6, `run_ambiguity.py`)
**v7's version of this experiment was confounded and its headline is withdrawn.** In v7 the standard
arms used 2 fitting windows and the perturbation arms 6, so most of the apparent improvement
(truth-outside 0.30 → 0.10) came from the acceptance rule, not the physics (a matched 6-window control
on the standard protocol already gave 3/20; the reviewer's check). v7 also mis-described six
participants with 2–10 accepted candidates as "no fit", and the perturbation protocol had 8 % more
external work.

v2 design: **identical 6 windows (minute boundaries 0, 22, 27, 32, 37, 40, 55) and identical
acceptance in every arm; equal external work (180 kJ) in every arm.** Arms: standard · fan_only (fan
1.5 m/s during exercise min 22–27) · work_only (50 W min 27–32, 150 W min 32–37) · combined. Optional
idealised total-E ±10 % on each. Paired: same 20 participants, same noise, same candidate-bank prior
within a seed; 3 seeds (participants + noise + bank). Truth families: reference, perfusion_only,
sweat_only; candidates always reference. "Truth outside" = outside the accepted-ensemble 95 % range
(not a calibrated interval). Metrics are means over 3 seeds (range in the JSON and the figure whiskers).

| truth | arm | available / insufficient / none (of 20) | truth outside (of 20) | coverage among available | mean \|median error\| |
|---|---|---|---|---|---|
| reference | standard | 20 / 0 / 0 | 0.0 | 1.00 | 0.054 |
| reference | fan / work / combined | 20 / 0 / 0 | 0.3 / 0.0 / 0.3 | 0.98–1.00 | 0.052–0.053 |
| perfusion_only | standard | 18 / 2 / 0 | 3.3 | 0.81 | 0.149 |
| perfusion_only | fan_only | 17.7 / 2 / 0.3 | **6.0** | 0.66 | 0.154 |
| perfusion_only | work_only | 18 / 2 / 0 | 3.7 | 0.80 | 0.143 |
| perfusion_only | combined | 17.7 / 2 / 0.3 | 5.7 | 0.68 | 0.147 |
| sweat_only | standard | 11.3 / 4.3 / 4.3 | 3.7 | 0.67 | 0.071 |
| sweat_only | fan_only | 9.0 / 4.7 / 6.3 | 1.7 | 0.79 | 0.066 |
| sweat_only | work_only | 11.3 / 5.0 / 3.7 | **0.3** | 0.97 | 0.046 |
| sweat_only | combined | 10.3 / 4.0 / 5.7 | 0.3 | 0.95 | 0.037 |

Paired deltas in truth-outside count vs standard, per seed: perfusion_only / fan_only **+2, +4, +2**;
perfusion_only / work_only +2, 0, −1; sweat_only / work_only **−3, −2, −5**; sweat_only / fan_only −1,
−2, −3. Idealised total-E on the standard protocol: perfusion_only truth-outside 3.3 → 6.0, coverage
0.81 → 0.67; no benefit anywhere else.

Readings (n = 20 × 3 seeds, one perturbation design, tol 1.5): (i) under the reference truth every arm
behaves the same; (ii) under the **sweat-law mismatch**, the workload swing reduces truth-outside cases
among *available* ensembles from 3.7 to 0.3 of 20 and the error from 0.071 to 0.046 °C, **while
availability stays limited (≈ 11 of 20 have an ensemble; 4–6 have none)** — the reviewer's paired
check on the 30 cases available in both arms shows the error falls 0.062 → 0.044 there too, so the
benefit is not only abstention; (iii) under the **perfusion mismatch** the fan pulse raised the
truth-outside count in all three seeds (+2, +4, +2) and slightly raised error among cases available in
both arms (0.145 → 0.153), but see the robustness sweep below; (iv) no arm changes the perfusion
underestimate (−0.14 to −0.15); (v) the idealised evaporation constraint again tightens the
perfusion-mismatch ensemble around a biased answer. Participant-level rows (id, gains, status,
median error, interval endpoints, n accepted) are saved in `ambiguity_results.json → participant_rows`.

**Robustness with participants and noise fixed (seed-1 participants; independent candidate bank;
bank size 1000 / 4000 / 8000 × tolerance 1.25 / 1.5 / 2.0):**
| contrast | Δ truth-outside (of 20) across the 9 settings | Δ mean \|median error\| |
|---|---|---|
| sweat_only: work_only − standard | −2, −1, −1, −4, −2, −2, −4, −2 (0 where no ensembles exist) | −0.008 to −0.032, all negative |
| perfusion_only: fan_only − standard | 0, 0, 0, −1, +1, 0, −1, 0, 0 | +0.004 to +0.013 (one −0.016) |

The **workload-swing benefit under the sweat mismatch was present in every evaluable bank-size and tolerance setting tested** (see the paired analysis below for the proper per-participant version). The
**fan-pulse harm under the perfusion mismatch does not**: with the same participants but a different
candidate bank the truth-outside delta is 0 ± 1; only a small, mostly positive error shift (≈ +0.01 °C)
remains. The v8 statement that the fan pulse "makes the perfusion-mismatch ensemble more often
wrong" is therefore downgraded to: *no benefit, with a small and bank-dependent adverse tendency*.

**Mechanism diagnostics (seed 1, pooled accepted candidates):** under the perfusion mismatch the
reference bank compensates the alternative perfusion law with **low blood-flow gains** (accepted p50
0.75 vs participants' true 0.95) and slightly under-reproduces the end-of-exercise core–skin gap
(2.28 vs 2.35 °C) — consistent with the core underestimate; the fan pulse leaves this accepted-gain
distribution essentially unchanged (p50 0.76) — consistent with, but not a proof of, its lack of benefit. Under the sweat mismatch the bank
compensates the linear sweat law with **high sweat and blood-flow gains** (p50 1.57 / 1.26 vs true
1.07 / 0.95); the workload swing shifts these pooled quantiles only slightly (1.50 / 1.24), so its
per-participant benefit is not explained by pooled gain statistics — the mechanism remains open and
needs per-participant accepted-set analysis.

## Paired robustness, mechanism and non-ideal noise (fig 7, `run_mechanism.py`)
All comparisons below are **truly paired**: per setting, the participants with an ensemble in *both*
arms; per-participant change in |ensemble median error|; counts improved / worsened; availability
transitions. Participant rows for every setting are in `mechanism_results.json → paired_robustness_rows`.
Bank randomness, bank size and participant variability are separated: nested subsets (1000 / 4000 /
8000) of one 8000 bank; three independent 4000 banks with the same participants; two new participant
sets with pre-specified settings (bank 4000, tol 1.5).

**Workload swing vs standard, sweat-law mismatch** — evaluable settings (≥ 3 common participants):
13 of 14. In every one of them more participants improved than worsened (e.g. 8/1, 11/3, 8/2, 13/6,
7/0, 11/1, 12/8, 9/2, 8/2, 9/1, 9/6, 9/3), mean paired Δ|error| −0.014 to −0.026 °C, outside→inside 1–3
and inside→outside 0 throughout; 0–4 participants lost their ensemble. The same direction was observed
across the nested bank sizes, the three independent banks and the two new participant sets tested.
(The 1000/1.25 setting has 2 common participants and is not counted.)

**Fan pulse vs standard, perfusion mismatch** — paired Δ|error| is small and mostly positive
(+0.001 to +0.017 °C; worsened/improved ranges from 17/2 to 8/9 across settings, i.e. not uniform).
Coverage-count deltas are inconsistent (as in v9). Statement: no benefit; a small adverse tendency on
paired error that is not uniform across settings.

**Mechanism — sweat-law mismatch, standard vs workload swing, same candidate IDs (12 participants
with both kept and lost sets):** in 83 % of them the candidates that the workload swing *removed*
("lost") had larger core error than those it kept (mean |error| lost − kept = +0.06 °C). Equal-weight
**rejection fractions per window** (non-exclusive: one candidate can fail several windows, so these are
not shares of a total): 0 / 0 / 0.30 / 0.09 / 0.46 / 0.31 for the windows 0–22 / 22–27 / 27–32 (50 W) /
32–37 (150 W) / 37–40 (back to 100 W) / recovery. The **return transient after the swing (37–40 min)**
and, secondarily, the low-work segment reject the most lost candidates; 28 % of lost candidates are
rejected solely by the 27–37 segments. Lost candidates have slightly lower sweat and blood-flow gains
than kept ones (−0.07 / −0.06 median). Note that the four-stage decomposition below attributes the
larger part of the paired benefit to the same kept candidates' protocol-dependent error change rather
than to this removal. Fig 7 (three cases) and fig 8 (all common participants: decomposition and ablation) show this. The three
selected cases: largest improvement (participant 18: 107 wrong candidates with error 0.05–0.20 °C all
rejected by the 37–40 window; their evaporation and core→skin transfer sit below the truth after the
swing), typical (participant 2: one candidate lost), and no improvement (participant 16: candidate removal helped slightly, A 0.027 → B 0.024,
but the *same kept candidates* have a larger error under the workload protocol, C 0.040; no truth
crossing is involved).
Secondary contrast (perfusion mismatch, fan): lost candidates are only slightly worse than kept
(+0.006 °C, worse in 58 % of participants) and their blood-flow gains differ little (−0.03) — the
candidates the fan pulse removes are not systematically the wrong ones, which is consistent with its
lack of benefit.

**Four-stage decomposition (fig 8; all 10 common participants; signed and absolute errors saved per participant
in `mechanism_results.json → four_stage_decomposition`):** A = standard, all accepted; B = standard, only
candidates kept in both arms; C = workload, the same kept candidates (vs the workload-protocol truth);
D = workload, all accepted. Mean Δ|error|: A→B (candidate removal) **−0.009** (helps in 5/10);
B→C (same candidates, protocol-dependent error change) **−0.018** (helps in 8/10); C→D (candidate
addition) +0.003 (helps in 3/10); A→D total −0.024, identical to the paired comparison. In this ordering
the larger share of the benefit is the protocol-dependent error change of the *same* candidates, not the
removal of wrong ones. This is one specific ordering, not a unique causal attribution.

**Observation-window ablation (fig 8; fixed 10 common cases; an ablation of which observations are used for
acceptance, not a removal of the physical transient):** all windows Δ|error| −0.024 (8/2 improved/
worsened, 0 candidates restored); without 37–40 min −0.018 (9/1, 234 lost candidates restored);
without recovery −0.019 (7/3, 9 restored); without both **+0.003** (5/5, 273 restored). Either
post-swing observation set alone preserves most of the benefit; removing both removes it. Availability
was 10/10 in both arms in every ablation.

**Noise stress test (same participants; session bias SD 0.5 °C, AR(1) with ~1-min correlation, both) — a stress test, not a general bias-sensitivity conclusion:**
the workload-swing benefit keeps its sign — paired Δ|error| −0.006 (bias), −0.012 (AR1), −0.014
(bias + AR1) vs −0.023 (white); improved/worsened 5/3, 5/4, 5/2 — but shrinks, the common-available
count falls to 7–9, and the coverage advantage disappears under session bias (truth-outside 4 vs 4;
6 vs 8 with bias + AR1, where the bias itself dominates). Under session bias the fan pulse's adverse
paired effect grows (+0.020 to +0.032).

## Likelihood-based inference with an explicit sensor-error model (fig 9, `run_calibrated.py`)
**Observation model (stated assumptions):** y_t = T_sk,model(θ; t) + b + e_t, session bias
b ~ N(0, 0.5²) per recording, AR(1) noise e_t with marginal SD 0.3 °C and 60 s correlation time.
The candidate likelihood marginalises b analytically and whitens the AR(1) noise. The reference-law
bank is the prior sample; the posterior over ΔT_core(40 min) is the importance-weighted bank
(4000 candidates in the main analyses); interval = weighted 2.5–97.5 %; availability = ESS ≥ 20.
**ESS is an importance-sampling reliability diagnostic, not a mismatch detector.** Main inference =
`full` (bias + AR(1)); `white` (independent noise, no bias) is the naive comparison only. An earlier
`full+disc` variant was removed because its covariance did not match the stated model.

**Preliminary held-out coverage check (40 reference-family participants, noise from the stated
model — the only case where 95 % is the correct target):** `full`: 38/40 available in both protocols;
coverage among available **36/38 (0.947, Wilson 95 % [0.83, 0.99])** in both; coverage of all 37/40;
width 0.318 / 0.319 °C; |error| 0.067 / 0.064; median ESS ≈ 246. `white`: ESS ≈ 1 for every
participant (the interval collapses to one candidate) — unusable with a discrete bank, so it is
omitted from fig 9. The observed coverage is consistent with 95 % in this limited reference-model
test; it is not a general calibration claim, and it depends on the bank (the AR(1) model reduces the
effective number of independent samples to ≈ 30 per recording, which is what spreads the weights).

**Physiology mismatch on the frozen protocols (20 participants, same noise model, frozen inference, `full`):**
| truth | protocol | available | coverage among available (Wilson) | coverage of all | \|error\| | width | median ESS |
|---|---|---|---|---|---|---|---|
| sweat_only | standard | 8/20 | 3/8 = 0.375 [0.14, 0.69] | 7/20 | 0.119 | 0.222 | 13 |
| sweat_only | work_only | 9/20 | 6/9 = 0.667 [0.35, 0.88] | 9/20 | 0.092 | 0.228 | 17 |
| perfusion_only | standard | 13/20 | 12/13 = 0.923 [0.67, 0.99] | 14/20 | 0.086 | 0.276 | 28 |
| perfusion_only | work_only | 11/20 | 11/11 = 1.00 [0.74, 1.00] | 17/20 | 0.082 | 0.296 | 27 |

Under mismatch the interval is not expected to be calibrated (the likelihood assumes the reference
laws), and with these counts the Wilson intervals are wide. **Paired (common available cases only):**
sweat_only, work_only − standard: n = 8, improved 7 / worsened 1, Δ|error| −0.024 °C, coverage on the
same 8 cases **3/8 → 5/8** (2 uncovered→covered, 0 covered→uncovered), availability 8 → 9 (one
participant gained an ensemble, none lost). perfusion_only: n = 11, 7/4, −0.005, coverage on the same 11 cases 10/11 → 11/11 (1 uncovered→covered,
0 covered→uncovered), availability 13 → 11 (2 lost). The unpaired table above (3/8 vs 6/9) mixes in the availability change; the paired numbers are
the ones to quote.

**Numerical stability of the posterior (`full`; pre-specified high- and low-ESS cases per family;
nested bank 4000 → 8000 → 16000 from bank seed A, repeated with bank seed B). PASS requires both
within-seed convergence (8k→16k: median < 0.010, endpoints < 0.020) AND cross-seed agreement at
16k (median < 0.010, endpoints < 0.020); criterion stored in the JSON. Result: 5 of 12 cases PASS,
7 unresolved.**
| case | ESS (A, 4k/8k/16k) | median A 4k/8k/16k, B 16k | seed spread at 16k (median / lo / hi) | verdict |
|---|---|---|---|---|
| reference high-ESS p19, std / work | 545/1083/2162 · 697/1390/2751 | 0.618/0.616/0.616, 0.621 · 0.676/0.672/0.672, 0.676 | 0.005/0.005/0.005 · 0.005/0.003/0.005 | PASS / PASS |
| reference low-ESS p6, std / work | 4/6/9 | 0.830/0.830/0.894, 0.941 · 0.843/0.903/0.904, 0.954 | 0.047–0.069 | unresolved / unresolved (cross-seed 0.050 / 0.069) |
| sweat high-ESS p4, std / work | 126/241/462 · 116/235/445 | 0.526/0.530/0.528, 0.531 · 0.548/0.552/0.550, 0.552 | ≤ 0.011 | unresolved (seed-B median moved 0.012 between 8k and 16k, limit 0.010) / PASS |
| sweat low-ESS p13, std / work | 2/2/4 · 3/4/6 | 0.250/0.250/0.250, 0.222 · 0.277/0.277/0.253, 0.299 | 0.028–0.046 | unresolved / unresolved |
| perfusion high-ESS p0, std / work | 281/557/1088 · 193/378/751 | 0.448/0.443/0.444, 0.438 · 0.518/0.520/0.524, 0.516 | ≤ 0.008 | PASS / PASS |
| perfusion low-ESS p1, std / work | 2/2/5 | 0.846/0.846/0.773, 0.669 · 0.865/0.865/0.791, 0.692 | 0.06–0.12 | unresolved / unresolved |

Five cases pass (reference high-ESS both protocols, sweat high-ESS work_only, perfusion high-ESS both
protocols) with medians stable to a few thousandths of a degree — well below the ≈ 0.02 °C effect.
**Seven cases are numerically unresolved**: all six low-ESS cases (medians move by 0.03–0.12 °C with
bank size and seed; those posteriors say nothing about physiology and are withheld by the ESS ≥ 20
rule) and one high-ESS case (sweat, standard: within-seed median moved 0.012 in seed B). So ESS ≥ 20
alone is not a convergence guarantee, and "all high-ESS cases stable" is not claimed. Whole-population check: the sweat-mismatch paired benefit is −0.024 °C at bank 4000
(n = 8 common, 7/1) and −0.024 °C at bank 16000 (n = 14 common, 12/2); availability rises with bank size
(8 → 14 of 20), so availability figures are bank-dependent and must be quoted with the bank size.

## What the study says now (final, v14)
Supported in simulation, with the stated limits:
- A partitional-calorimetry inverse with exact bookkeeping (oracle check to 1e-13) whose deployable
  form has ≈ 0.3 °C RMSE across held-out reference physiologies, dominated by between-person gain
  variability it cannot see.
- A good skin-temperature fit does not guarantee a correct core-temperature reconstruction: under the
  tested perfusion mismatch the ensemble underestimates ΔT_core by ≈ 0.2 °C while fitting the skin
  trace as well as the reference case, in causal, retrospective and joint fits.
- Within-session workload redistribution (50 ↔ 150 W swing, equal total work) reduces the paired
  core error under the sweat-law mismatch by ≈ 0.02 °C among participants for whom an estimate exists,
  in every evaluable bank-size/tolerance/seed/participant setting tested and under the explicit
  sensor-error likelihood (−0.024 °C at bank 4000 and 16000). The benefit is linked to both candidate
  selection and the same candidates' protocol-dependent error change; the post-swing observations
  (return transient, recovery) carry the discriminating information. It is a mechanism demonstration
  with a small effect, not a performance claim.
- The fan pulse gives no benefit under the perfusion mismatch, with a small, non-uniform adverse
  tendency on paired error.
- With the bias + AR(1) likelihood, observed held-out coverage (36/38) is consistent with 95 % in this
  limited reference-model test; the naive white-noise likelihood is unusable with a discrete bank.
  Under physiology mismatch the same intervals under-cover. Numerical stability of the posterior was
  demonstrated for 5 of 12 pre-specified cases; the other 7 (all low-ESS cases and one high-ESS case)
  are explicitly unresolved.

**Release statement:** preliminary synthetic study completed; numerical stability demonstrated for
some tested cases, with unresolved cases explicitly identified. The simulation study is closed; not
every posterior in it is numerically validated, and that limitation is stated rather than hidden.

Unresolved limitations:
- Everything is synthetic; the stress-test physiologies and the sensor-error parameters (0.5 °C bias,
  0.3 °C AR(1) noise, 60 s) are assumed, not measured. Real camera calibration data must replace them.
- No regional / visibility observation model — nothing here supports a claim about camera coverage.
- The sweat-mismatch mechanism is characterised for one perturbation design; the four-stage
  decomposition is one ordering, not a unique attribution.
- Availability and coverage depend on bank size and tolerance; quote them with the settings.
- Coverage checks are preliminary (n = 40 reference, 20 mismatch); Wilson intervals are wide.

## Next (outside this simulation package)
Measure the sensor-error parameters on the actual thermal camera (blackbody / phantom), then decide
the human protocol with the professor; the simulation is frozen at v14.
