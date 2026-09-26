"""
Two-node (Gagge-type) thermoregulation model + partitional-calorimetry
reconstruction of core-temperature change from surface observations.

All heat-flux terms are in W/m^2 of DuBois area. Vectorised over a batch of
parameter sets so Monte-Carlo and identifiability sweeps run in numpy.

Conventions
-----------
* State (T_cr, T_sk) and fluxes stored at index i are the values *at* t[i],
  before the step t[i] -> t[i+1]. The reconstruction integrates the same way, so
  reconstructed change at t[i] uses fluxes from t[0]..t[i-1] only (dT[0] == 0).
* Metabolic rate during exercise: M = M_rest + P_ext / eff_net, i.e. eff_net is a
  *net* (delta) efficiency, not gross efficiency. Resting metabolism is therefore
  added once, not double-counted.
* Convective coefficient h_c = max(8.3 v^0.6, 3.0) has a floor for v < ~0.18 m/s.
  A "fan off" scenario at 0.15 m/s sits on that floor, so airflow perturbations
  around it have almost no effect by construction - report airflow sensitivity at
  fan-on speeds as well.
"""
import numpy as np

SIGMA = 5.670e-8          # Stefan-Boltzmann
H_FG = 2.43e6             # J/kg latent heat of sweat
LR = 16.5                 # Lewis ratio K/kPa
C_B = 3490.0              # J/(kg K) body specific heat
T_CR0_DEFAULT = 36.9
T_SK0_DEFAULT = 33.5


def p_sat_kpa(T):
    """Saturation vapour pressure (kPa), Antoine, T in degC."""
    return 0.1 * np.exp(18.6686 - 4030.183 / (T + 235.0))


class Body:
    def __init__(self, mass=70.0, A_D=1.8, emissivity=0.95, A_r_ratio=0.70):
        self.mass, self.A_D, self.eps, self.A_r = mass, A_D, emissivity, A_r_ratio


def alpha_skin(m_bl):
    """Fraction of body mass in the skin node as a function of skin blood flow (L/m^2/h)."""
    return 0.0418 + 0.745 / (m_bl + 0.585)


def h_conv(v_air):
    return np.maximum(8.3 * np.power(np.maximum(v_air, 1e-3), 0.6), 3.0)


def h_rad(T_sk, T_r, body):
    Tm = (T_sk + T_r) / 2.0 + 273.15
    return 4.0 * body.eps * SIGMA * body.A_r * Tm ** 3


def sweat_evap(T_sk, T_b, hc, p_a, g_sw=1.0, sweat_law="gagge", T_cr=None, lam=None):
    """Regulatory + diffusive skin evaporation (W/m^2).
    sweat_law='gagge' : Gagge/ASHRAE: m_rsw = 4.7e-5 * (T_b-36.49)+ * exp((T_sk-33.7)+/10.7)
    sweat_law='linear': SYNTHETIC STRESS-TEST variant (coefficients ours, not from data):
                        m_rsw = 3.0e-4*(T_cr-37.0)+ + 2.5e-5*(T_sk-34.0)+   [kg/m^2/s]
                        Linear-controller *form* as in Stolwijk-type models; set-points and
                        gains chosen only to give plausible sweat rates (0.4-0.7 L/h here)."""
    E_max = np.maximum(LR * hc * (p_sat_kpa(T_sk) - p_a), 1e-3)
    wsig_b = np.maximum(T_b - 36.49, 0); wsig_sk = np.maximum(T_sk - 33.7, 0)
    m_g = 4.7e-5 * wsig_b * np.exp(np.minimum(wsig_sk / 10.7, 5.0))
    Tc = T_b if T_cr is None else T_cr
    m_l = 3.0e-4 * np.maximum(Tc - 37.0, 0) + 2.5e-5 * np.maximum(T_sk - 34.0, 0)
    if lam is None:
        lam = 1.0 if sweat_law == "linear" else 0.0
    m_rsw = g_sw * ((1 - lam) * m_g + lam * m_l)       # lam: mismatch strength 0 (reference) .. 1 (linear)
    E_rsw = np.minimum(m_rsw * H_FG, E_max)
    w_rsw = E_rsw / E_max
    return E_rsw + (1 - w_rsw) * 0.06 * E_max, m_rsw     # (evaporative loss, PRE-CAP secreted sweat)


def skin_blood_flow(T_cr, T_sk, g_bl=1.0, perf_law="gagge"):
    """Skin blood flow (L/m^2/h), clipped to [0.5, 90].
    perf_law='gagge': (6.3 + 200*(T_cr-36.8)+) / (1 + 0.5*(33.7-T_sk)+)
    perf_law='alt'  : SYNTHETIC STRESS-TEST variant, SAME bounded structural form so that the
                      Gagge alpha(m_bl) mapping stays applicable (m_bl >= 6.3/(1+0.3*csig) > 2,
                      hence alpha <= ~0.33): (6.3 + 150*(T_cr-37.0)+) / (1 + 0.3*(33.7-T_sk)+)
                      i.e. higher core set-point, lower gain, weaker cold inhibition."""
    if perf_law == "alt":
        m = (6.3 + 150.0 * np.maximum(T_cr - 37.0, 0)) / (1.0 + 0.3 * np.maximum(33.7 - T_sk, 0))
    else:
        m = (6.3 + 200.0 * np.maximum(T_cr - 36.8, 0)) / (1.0 + 0.5 * np.maximum(33.7 - T_sk, 0))
    return np.clip(g_bl * m, 0.5, 90.0)


FAMILIES = {  # forward-model variants for robustness tests (inverse always assumes 'reference')
    "reference":      dict(sweat_law="gagge", perf_law="gagge"),
    "sweat_only":     dict(sweat_law="linear", perf_law="gagge"),
    "perfusion_only": dict(sweat_law="gagge", perf_law="alt"),
    "combined":       dict(sweat_law="linear", perf_law="alt"),
}


def simulate(protocol, env, body, gains=None, dt=2.0, T_cr0=T_CR0_DEFAULT,
             T_sk0=T_SK0_DEFAULT, n=None, family="reference", sweat_mix=None):
    """
    protocol: dict of arrays over time (seconds): t, M (W/m^2), W (external work W/m^2)
    env:      dict of arrays: T_a, T_r (degC), v (m/s), RH (0-1)
    gains:    dict of scalars/arrays shape (n,): sweat, bloodflow, hc (multipliers)
    T_cr0:    scalar or array (n,) - explicit baseline core temperature
    Returns dict of trajectories with shape (n, T); values at index i are AT t[i].
    """
    t = protocol["t"]; T = len(t)
    fam = FAMILIES[family]
    gains = gains or {}
    n = n or max([np.size(v) for v in gains.values()] + [np.size(T_cr0), 1])
    g_sw = np.broadcast_to(np.asarray(gains.get("sweat", 1.0), float), (n,))
    g_bl = np.broadcast_to(np.asarray(gains.get("bloodflow", 1.0), float), (n,))
    g_hc = np.broadcast_to(np.asarray(gains.get("hc", 1.0), float), (n,))

    T_cr = np.array(np.broadcast_to(np.asarray(T_cr0, float), (n,)))
    T_sk = np.array(np.broadcast_to(np.asarray(T_sk0, float), (n,)))
    keys = ["T_cr", "T_sk", "R", "C", "E_sk", "Resp", "q_cs", "alpha", "m_bl", "S_cr", "S_sk", "m_rsw"]
    out = {k: np.zeros((n, T)) for k in keys}
    for i in range(T):
        M, W = protocol["M"][i], protocol["W"][i]
        T_a, T_r, v, RH = env["T_a"][i], env["T_r"][i], env["v"][i], env["RH"][i]
        p_a = RH * p_sat_kpa(T_a)

        m_bl = skin_blood_flow(T_cr, T_sk, g_bl, perf_law=fam["perf_law"])
        alpha = alpha_skin(m_bl)
        T_b = alpha * T_sk + (1 - alpha) * T_cr

        hc = g_hc * h_conv(v); hr = h_rad(T_sk, T_r, body)
        C = hc * (T_sk - T_a); R = hr * (T_sk - T_r)
        E_sk, m_rsw = sweat_evap(T_sk, T_b, hc, p_a, g_sw, sweat_law=fam["sweat_law"], T_cr=T_cr, lam=sweat_mix)
        Resp = 0.0014 * M * (34.0 - T_a) + 0.0173 * M * (5.87 - p_a)
        q_cs = (5.28 + 1.163 * m_bl) * (T_cr - T_sk)
        S_cr = M - W - Resp - q_cs
        S_sk = q_cs - C - R - E_sk
        for k, val in zip(keys, [T_cr, T_sk, R, C, E_sk, Resp, q_cs, alpha, m_bl, S_cr, S_sk, m_rsw]):
            out[k][:, i] = val
        T_cr = T_cr + dt * S_cr * body.A_D / ((1 - alpha) * body.mass * C_B)
        T_sk = T_sk + dt * S_sk * body.A_D / (alpha * body.mass * C_B)
    out["t"] = t
    return out


def make_protocol(dt=2.0, rest=600, ex=1800, rec=900, P_ext=100.0, eff_net=0.20,
                  body=None, M_rest=58.0):
    """Rest -> constant-power cycling -> recovery. eff_net is NET (delta) efficiency."""
    body = body or Body()
    t = np.arange(0, rest + ex + rec, dt)
    M = np.full_like(t, M_rest, dtype=float); W = np.zeros_like(t, dtype=float)
    in_ex = (t >= rest) & (t < rest + ex)
    M[in_ex] = M_rest + (P_ext / eff_net) / body.A_D
    W[in_ex] = P_ext / body.A_D
    rec_mask = t >= rest + ex
    M[rec_mask] = M_rest + (M[in_ex][-1] - M_rest) * np.exp(-(t[rec_mask] - (rest + ex)) / 180.0)
    return {"t": t, "M": M, "W": W, "dt": dt}


def make_env(t, T_a=24.0, T_r=24.0, v=0.15, RH=0.40):
    ones = np.ones_like(t, dtype=float)
    return {"T_a": T_a * ones, "T_r": T_r * ones, "v": v * ones, "RH": RH * ones}


def reconstruct(obs, body, T_cr0, alpha=0.10, E_mode="model", E_exact=None,
                E_total=None, dt=2.0):
    """
    Partitional-calorimetry reconstruction of core-temperature change.

    obs:    dict of *observed* arrays (n,T): T_sk, M, W, T_a, T_r, v, RH
    T_cr0:  baseline core temperature used by the reconstruction (scalar or (n,)).
            This is an explicit input: 'known', 'uncertain' or 'unknown' baselines
            are modelled by what the caller passes here.
    alpha:  scalar (fixed) or array (n,T) (matched to the forward model, for checks)
    E_mode: 'none'  - evaporation ignored
            'model' - Gagge sweat law driven by observed T_sk and the running core estimate
            'exact' - use E_exact (truth) -> bookkeeping / error-decomposition checks
            'total' - like 'model' but rescaled so the session integral equals E_total
                      (an IDEALISED total-evaporative-energy constraint, not a real
                      body-mass measurement)
    Returns dT_cr (n,T), terms dict.
    """
    T_sk, M, W = obs["T_sk"], obs["M"], obs["W"]
    n, T = T_sk.shape
    scale = body.A_D / (body.mass * C_B)
    p_a = obs["RH"] * p_sat_kpa(obs["T_a"])
    hc = h_conv(obs["v"]); hr = h_rad(T_sk, obs["T_r"], body)
    C = hc * (T_sk - obs["T_a"]); R = hr * (T_sk - obs["T_r"])
    Resp = 0.0014 * M * (34.0 - obs["T_a"]) + 0.0173 * M * (5.87 - p_a)
    al = np.broadcast_to(np.asarray(alpha, float), (n, T))
    Tcr0 = np.broadcast_to(np.asarray(T_cr0, float), (n,))
    dT_sk = T_sk - T_sk[:, :1]

    if E_mode == "none":
        E = np.zeros_like(T_sk)
    elif E_mode == "exact":
        E = np.asarray(E_exact, float)
    else:
        E = np.zeros_like(T_sk)
        dTcr_run = np.zeros(n)                    # reconstructed core change at t[i]
        for i in range(T):
            T_cr_hat = Tcr0 + dTcr_run
            T_b = al[:, i] * T_sk[:, i] + (1 - al[:, i]) * T_cr_hat
            E[:, i], _ = sweat_evap(T_sk[:, i], T_b, hc[:, i], p_a[:, i], T_cr=T_cr_hat)
            S = M[:, i] - W[:, i] - Resp[:, i] - R[:, i] - C[:, i] - E[:, i]
            if i < T - 1:
                dTcr_run = dTcr_run + (S * dt * scale - al[:, i] * (T_sk[:, i + 1] - T_sk[:, i])) / (1 - al[:, i])
        if E_mode == "total" and E_total is not None:
            integ = E[:, :-1].sum(axis=1, keepdims=True) * dt     # intervals t[0..T-2]
            E = E * (np.asarray(E_total)[:, None] / np.maximum(integ, 1e-6))

    S = M - W - Resp - R - C - E
    # per-step bookkeeping: (1-a_i) dT_cr_i + a_i dT_sk_i = S_i dt scale  (exact for the
    # two-node model when a_i is the forward model's alpha at t[i])
    step_sk = np.diff(T_sk, axis=1)
    inc = (S[:, :-1] * dt * scale - al[:, :-1] * step_sk) / (1 - al[:, :-1])
    dT_cr = np.concatenate([np.zeros((n, 1)), np.cumsum(inc, axis=1)], axis=1)
    return dT_cr, {"R": R, "C": C, "E": E, "Resp": Resp, "S": S}
