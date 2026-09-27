"""
Statistics-agnostic OFDM channel estimator.

Pipeline (all statistics are estimated from the received batch itself):
  1. LS estimates on the 72 pilots.
  2. Frequency correlation: non-negative least-squares fit of a fine-grid
     power-delay profile (PDP) to the pilot autocorrelation, with a
     noise-aware soft threshold on each delay bin (removes spurious taps),
     plus light smoothing / floor at high SNR (keeps the prior from being
     over-confident about the fine delay structure).
  3. Time correlation: Jakes/J0 model fitted to the pilot-symbol
     correlation rho(9) (with sign disambiguation: J0 < 0 vs LOS rotation,
     using an envelope fading-depth statistic), phase kept only if reliable.
  4. Joint 2-D MAP/LMMSE in a reduced-rank Kronecker eigenbasis
     (Ut (x) Uf) with heteroscedastic per-RE observation precisions.
  5. Semi-blind refinement (SNR >= ~2 dB): expectation-propagation style
     virtual observations from the unknown 16-QAM data REs (exact 16-point
     Gaussian-mixture moment matching, cavity = posterior minus own message),
     tempered by an SNR-dependent factor, iterated to convergence.
  6. Re-estimation of the Doppler spectrum (NNLS on a Doppler grid) from the
     posterior channel + posterior covariance (EM step), then a second
     semi-blind pass with the refined time prior.
  At very low SNR only one weakly-weighted EP step is applied.
"""
import numpy as np
import torch
from scipy.optimize import nnls
from scipy.special import j0
from link_config import (PILOT_MASK, PILOT_SYMBOLS, FFT_SIZE, NUM_OFDM_SYMBOLS,
                         PILOT_OFDM_SYMBOLS)

N = FFT_SIZE
T = NUM_OFDM_SYMBOLS
PS = np.array(PILOT_OFDM_SYMBOLS)
KP = np.where(PILOT_MASK[PS[0]])[0]
NP = len(KP)
XP = np.stack([PILOT_SYMBOLS[s, KP] for s in PS])
PMASK = PILOT_MASK
DMASK = ~PILOT_MASK
PSYM = PILOT_SYMBOLS.astype(np.complex128)
KK = np.arange(N)

# ---- delay grid for the PDP fit (samples) ---------------------------------
DT = 0.125
TAU = np.arange(-2.0, 24.0, DT)
A = np.exp(-2j * np.pi * np.outer(np.arange(NP) * 2, TAU) / N)      # pilot lags
EF = np.exp(-2j * np.pi * (KK[:, None] - KK[None, :])[..., None] * TAU / N)
TBINS = np.floor(TAU + 0.5).astype(int)
UBINS = np.unique(TBINS)
BIDX = np.searchsorted(UBINS, TBINS)

# ---- J0 inversion table ---------------------------------------------------
_x = np.linspace(0, 3.8317, 4000)
_J = j0(_x)

# ---- Doppler grid for the time-correlation re-estimation -------------------
NU = np.arange(-0.12, 0.1201, 0.0025)                                # cycles / symbol
AT = np.exp(2j * np.pi * np.outer(np.arange(T), NU))

# ---- 16-QAM ----------------------------------------------------------------
PAM = np.array([-3., -1., 1., 3.]) / np.sqrt(10)
QAM = (PAM[:, None] + 1j * PAM[None, :]).ravel()
QA2 = np.abs(QAM) ** 2
_q = np.round(QA2 * 10).astype(int)                                  # |x|^2 classes 0.2, 1.0, 1.8
QCL = np.array(sorted(set(_q))) / 10.0
_C = (_q[:, None] == np.array(sorted(set(_q)))[None, :]).astype(np.float32)   # [16,3]
RED = torch.from_numpy(np.concatenate([_C, _C * QAM.real[:, None], _C * QAM.imag[:, None]], 1).astype(np.float32))
QCLt = torch.from_numpy(QCL.astype(np.float32))
XRt = torch.from_numpy(QAM.real.astype(np.float32))
XIt = torch.from_numpy(QAM.imag.astype(np.float32))
CIDX = torch.from_numpy(np.searchsorted(QCL, np.round(QA2 * 10) / 10.0))

# ---- hyper-parameters --------------------------------------------------------
KAP_PR = 2.0        # PDP bin soft-threshold, in units of the bin-power noise std
SMW, SMW_SNR = 0.25, 7.0          # PDP smoothing half-width (samples), active above SNR
FLOOR, FLW, FL_SNR = 1e-3, 4.0, 22.0   # PDP floor over the support, active above SNR
PH_LO, PH_W = 0.5, 0.3            # keep time-correlation phase only when |rho| is large
FLIP_A, F_LOS = 0.45, 0.45        # J0<0 vs LOS rotation disambiguation
EPS, KMAX = 1e-5, 200             # reduced-rank basis
A0, A1, AMAX = 2.0, 10.0, 0.75    # EP tempering: alpha = clip((snr_db-A0)/A1, 0, AMAX)
LOW_A = 0.05                      # tempering of the single EP step below A0 (scaled by SNR)
N_ITER, N_ITER2 = 8, 6            # EP iterations (first / second pass)
CONV = 0.01                       # stop when mean |dH|^2 < CONV * no
RS_SNR = 3.0                      # Doppler-spectrum re-estimation above this SNR
CT = torch.complex128


def _fit_pdp(hp, no, smw=0.0, floor=0.0):
    h = hp.reshape(-1, NP)
    r = np.array([np.mean(h[:, m:] * np.conj(h[:, :NP - m])) for m in range(NP)])
    r[0] -= no
    cnt = h.shape[0] * (NP - np.arange(NP))
    w = np.sqrt(cnt / cnt[0])
    M = np.concatenate([(A * w[:, None]).real, (A * w[:, None]).imag])
    v = np.concatenate([(r * w).real, (r * w).imag])
    p = nnls(M, v)[0]
    # noise-aware soft threshold per 1-sample delay bin
    sig = (no / NP) / np.sqrt(h.shape[0])
    pb = np.bincount(BIDX, weights=p, minlength=len(UBINS))
    keep = np.maximum(pb - KAP_PR * sig, 0) / np.maximum(pb, 1e-30)
    p = p * keep[BIDX]
    if smw > 0:
        nk = int(round(smw / DT))
        p = np.convolve(p, np.ones(2 * nk + 1) / (2 * nk + 1), mode='same')
    if p.sum() <= 1e-9:
        p = np.zeros_like(p)
        p[np.argmin(np.abs(TAU))] = max(np.real(r[0]), 1e-3)
    if floor > 0:
        tmax = TAU[np.nonzero(p > 1e-3 * p.max())[0].max()] + FLW
        sup = (TAU <= tmax) & (TAU >= 0)
        p = p + floor * p.sum() * sup / max(1, np.sum(sup))
    return p


def _time_model(rho, fdepth):
    """J0 (Jakes) time correlation matched to the pilot correlation rho at lag 9."""
    ph = np.angle(rho)
    a = min(abs(rho), 1.0)
    if abs(ph) > np.pi / 2 and (a < FLIP_A or fdepth > F_LOS):
        # NLOS fast fading: J0 beyond its first zero (negative correlation)
        ph = np.angle(-rho)
        a = -a
    D = PS[1] - PS[0]
    x = _x[np.argmin(abs(_J - a))]
    ph = ph * np.clip((abs(a) - PH_LO) / PH_W, 0, 1)
    return lambda d: j0(x * np.abs(d) / D) * np.exp(1j * ph * d / D)


class _Post:
    """Reduced-rank 2-D Gaussian posterior in the Kronecker eigenbasis Ut (x) Uf."""

    def __init__(self, Rt, Rf):
        lt, Ut = torch.linalg.eigh(torch.from_numpy(np.ascontiguousarray(Rt, dtype=np.complex128)))
        lf, Uf = torch.linalg.eigh(torch.from_numpy(np.ascontiguousarray(Rf, dtype=np.complex128)))
        lt, Ut, lf, Uf = lt.numpy(), Ut.numpy(), lf.numpy(), Uf.numpy()
        lt = np.maximum(lt[::-1], 0); Ut = Ut[:, ::-1]
        lf = np.maximum(lf[::-1], 0); Uf = Uf[:, ::-1]
        lam = np.outer(lt, lf)
        thr = EPS * lam.max()
        nt = max(1, int(np.sum(lt * lf[0] > thr)))
        nf = max(1, int(np.sum(lf * lt[0] > thr)))
        lam = lam[:nt, :nf].ravel()
        order = np.argsort(-lam)
        K = max(1, int(min(KMAX, np.sum(lam > thr))))
        self.sel = torch.from_numpy(np.sort(order[:K]))
        self.lam = np.maximum(lam[self.sel.numpy()], 1e-300)
        self.nt, self.nf, self.K = nt, nf, K
        self.Ut = torch.from_numpy(Ut[:, :nt].copy()).to(CT)                  # [14,nt]
        self.Uf = torch.from_numpy(Uf[:, :nf].copy()).to(CT)                  # [72,nf]
        self.P = (self.Ut.conj()[:, :, None] * self.Ut[:, None, :]).reshape(T, nt * nt)
        O = (self.Uf.conj()[:, :, None] * self.Uf[:, None, :]).reshape(N, nf * nf)
        self.Or = O.real.contiguous(); self.Oi = O.imag.contiguous()
        self.Lam = torch.diag(torch.from_numpy(1.0 / self.lam).to(CT))
        Uk = (Ut[:, None, :nt, None] * Uf[None, :, None, :nf]).reshape(T * N, nt * nf)[:, self.sel.numpy()]
        self.UkH = torch.from_numpy(Uk).to(CT).T.conj().contiguous()          # [K,1008]
        self.Z = None

    def solve(self, d, r, want_var=True):
        """d: per-RE precision [B,14,72], r = d * (virtual LS observation)."""
        B = d.shape[0]
        nt, nf = self.nt, self.nf
        dt = torch.from_numpy(np.ascontiguousarray(d.reshape(B * T, N), dtype=np.float64))
        G = torch.complex(dt @ self.Or, dt @ self.Oi)                          # [B*14,nf*nf]
        Q4 = (self.P.T[None] @ G.reshape(B, T, nf * nf)).reshape(B, nt, nt, nf, nf)
        Q = Q4.permute(0, 1, 3, 2, 4).reshape(B, nt * nf, nt * nf)
        Q = Q[:, self.sel][:, :, self.sel] + self.Lam
        Q = 0.5 * (Q + Q.conj().transpose(-1, -2))
        R2 = self.Ut.conj().T[None] @ torch.from_numpy(r).to(CT) @ self.Uf.conj()
        rhs = R2.reshape(B, -1)[:, self.sel]
        L, info = torch.linalg.cholesky_ex(Q)
        if bool((info != 0).any()):
            sc = Q.diagonal(dim1=-2, dim2=-1).real.mean(-1)[:, None, None]
            eye = torch.eye(Q.shape[-1], dtype=Q.dtype)
            L, info = torch.linalg.cholesky_ex(Q + 1e-9 * sc * eye)
            if bool((info != 0).any()):
                L = torch.linalg.cholesky(Q + 1e-6 * sc * eye)
        c = torch.cholesky_solve(rhs[..., None], L)[..., 0]
        C = torch.zeros(B, nt * nf, dtype=CT)
        C[:, self.sel] = c
        H = self.Ut[None] @ C.reshape(B, nt, nf) @ self.Uf.T[None]
        if not want_var:
            return H.numpy(), None
        # posterior variance per RE (batch-averaged precision)
        Qm = Q.mean(0)
        Lm = torch.linalg.cholesky(Qm + 1e-12 * Qm.diagonal().real.mean() * torch.eye(Qm.shape[-1], dtype=Qm.dtype))
        Z = torch.linalg.solve_triangular(Lm, self.UkH, upper=False)
        self.Z = Z
        E = (Z.real ** 2 + Z.imag ** 2).sum(0).reshape(1, T, N).expand(B, T, N)
        return H.numpy(), np.maximum(E.numpy(), 1e-12)


def _ep_step(Y, H, E, dold, hold, no, alpha, P0):
    """EP virtual observations for the data REs (16-QAM, unknown symbols)."""
    B = Y.shape[0]
    Hd = H[:, DMASK]; Ed = E[:, DMASK]
    dd = dold[:, DMASK]; hd = hold[:, DMASK]
    pc = np.maximum(1.0 / Ed - dd, 1e-3 / P0)            # cavity precision
    mc = (Hd / Ed - dd * hd) / pc                        # cavity mean
    vc = 1.0 / pc
    yy = Y[:, DMASK]
    shp = mc.shape
    mr = torch.from_numpy(np.ascontiguousarray(mc.real, dtype=np.float32).reshape(-1, 1))
    mi = torch.from_numpy(np.ascontiguousarray(mc.imag, dtype=np.float32).reshape(-1, 1))
    yr = torch.from_numpy(np.ascontiguousarray(yy.real, dtype=np.float32).reshape(-1, 1))
    yi = torch.from_numpy(np.ascontiguousarray(yy.imag, dtype=np.float32).reshape(-1, 1))
    v = torch.from_numpy(vc.astype(np.float32).reshape(-1, 1))
    m2 = mr * mr + mi * mi
    y2 = yr * yr + yi * yi
    ur = yr * mr + yi * mi
    ui = yr * mi - yi * mr
    # symbol posterior: p(x|y) ~ CN(y; m x, no + |x|^2 v)
    is_c = 1.0 / (no + QCLt * v)                                           # [n,3]
    bc = torch.log(is_c) - (y2 + m2 * QCLt) * is_c
    ll = (ur * (2 * XRt) - ui * (2 * XIt)).mul_(is_c[:, CIDX]).add_(bc[:, CIDX])
    w = torch.softmax(ll, -1)
    R = w @ RED
    Wc, WX, WY = R[:, 0:3], R[:, 3:6], R[:, 6:9]
    # moments of the Gaussian-mixture posterior of h
    vx = 1.0 / (1.0 / v + QCLt / no)
    a = vx / v
    b = vx / no
    aW = (a * Wc).sum(1, keepdim=True)
    bX = b * WX; bY = b * WY
    sX = bX.sum(1, keepdim=True); sY = bY.sum(1, keepdim=True)
    mur = aW * mr + yr * sX + yi * sY
    mui = aW * mi + yi * sX - yr * sY
    Pp = mr * yr + mi * yi
    Qq = mr * yi - mi * yr
    sec = (Wc * (vx + a * a * m2 + b * b * QCLt * y2) + 2 * a * (Pp * bX + Qq * bY)).sum(1, keepdim=True)
    var = torch.clamp(sec - (mur * mur + mui * mui), min=1e-12).numpy().reshape(shp).astype(np.float64)
    mu = (mur.numpy() + 1j * mui.numpy()).reshape(shp)
    # extrinsic (posterior / cavity) message
    de = 1.0 / var - 1.0 / vc
    ok = de > 1e-6
    he = np.where(ok, (mu / var - mc / vc) / np.where(ok, de, 1.0), 0.0)
    de = np.where(ok, de, 0.0)
    d = np.empty((B, T, N)); ht = np.empty((B, T, N), complex)
    d[:, PMASK] = 1.0 / no; ht[:, PMASK] = Y[:, PMASK] / PSYM[PMASK]
    d[:, DMASK] = de * alpha; ht[:, DMASK] = he
    return d, ht


def _nnls_c(Amat, r, w):
    M = np.concatenate([(Amat * w[:, None]).real, (Amat * w[:, None]).imag])
    v = np.concatenate([(r * w).real, (r * w).imag])
    return nnls(M, v)[0]


def _ep_loop(post, Y, H, E, dold, hold, no, alpha, P0, n_iter):
    for _ in range(n_iter):
        d, ht = _ep_step(Y, H, E, dold, hold, no, alpha, P0)
        dold, hold = d, ht
        Hn, E = post.solve(d, d * ht)
        dH = np.mean(np.abs(Hn - H) ** 2)
        H = Hn
        if dH < CONV * no:
            break
    return H, E


def estimate_channel(y, no):
    nth = torch.get_num_threads()
    torch.set_num_threads(1)          # small problems: threading only adds overhead
    try:
        return _estimate(y, no)
    finally:
        torch.set_num_threads(nth)


def _estimate(y, no):
    Y = (y.numpy() if isinstance(y, torch.Tensor) else np.asarray(y)).astype(np.complex128)
    no = float(np.mean(no.numpy() if isinstance(no, torch.Tensor) else no))
    no = max(no, 1e-8)
    snr_db = -10 * np.log10(no)
    B = Y.shape[0]
    t = np.arange(T)

    # ---- second-order statistics from the pilots --------------------------
    hp = Y[:, PS][:, :, KP] / XP[None]
    p = _fit_pdp(hp, no, SMW if snr_db >= SMW_SNR else 0.0, FLOOR if snr_db >= FL_SNR else 0.0)
    P0 = max(p.sum(), 1e-3)
    Rf = (EF * p).sum(-1)
    rho = np.mean(hp[:, 1] * np.conj(hp[:, 0])) / P0
    Pe = np.abs(Y) ** 2
    Pe = Pe - Pe.mean()
    fdepth = np.mean(Pe[:, 1:] * Pe[:, :-1]) / P0 ** 2        # ~1 Rayleigh, << 1 strong LOS
    Rt = _time_model(rho, fdepth)(t[:, None] - t[None, :])
    post = _Post(Rt, Rf)

    # ---- pilot-only 2-D LMMSE ---------------------------------------------
    d0 = np.zeros((B, T, N)); d0[:, PMASK] = 1.0 / no
    h0 = np.zeros((B, T, N), complex); h0[:, PMASK] = Y[:, PMASK] / PSYM[PMASK]
    H, E = post.solve(d0, d0 * h0)
    alpha = float(np.clip((snr_db - A0) / A1, 0, AMAX))
    if alpha <= 0:
        # very low SNR: one weakly-weighted semi-blind step
        d, ht = _ep_step(Y, H, E, d0, h0, no, LOW_A * min(1.0, 1.0 / no), P0)
        H, _ = post.solve(d, d * ht, want_var=False)
        return torch.tensor(H.astype(np.complex64))
    Hpil = H.copy()
    ylim = 4.0 * max(np.mean(np.abs(Y) ** 2), P0) + 1e-9

    # ---- semi-blind pass 1 --------------------------------------------------
    H, E = _ep_loop(post, Y, H, E, d0, h0, no, alpha, P0, N_ITER)

    # ---- EM re-estimation of the Doppler spectrum, semi-blind pass 2 -------
    if snr_db >= RS_SNR and post.Z is not None:
        rtt = np.array([np.mean(H[:, m:] * np.conj(H[:, :T - m])) for m in range(T)])
        Zr = post.Z.reshape(-1, T, N)
        St = torch.einsum('atk,ask->ts', Zr.conj(), Zr).numpy() / N
        rtt = rtt + np.array([np.mean(np.diag(St, -m)) for m in range(T)])
        wt = np.sqrt((T - np.arange(T)) / T); wt[0] = 0.0
        q = _nnls_c(AT, rtt, wt)
        if q.sum() > 1e-6:
            Rt = (np.exp(2j * np.pi * (t[:, None] - t[None, :])[..., None] * NU) * q).sum(-1) / q.sum()
            post = _Post(Rt, Rf)
            H, E = post.solve(d0, d0 * h0)
            H, E = _ep_loop(post, Y, H, E, d0, h0, no, alpha, P0, N_ITER2)

    if not np.all(np.isfinite(H)) or np.mean(np.abs(H) ** 2) > ylim:
        H = Hpil
    return torch.tensor(H.astype(np.complex64))
