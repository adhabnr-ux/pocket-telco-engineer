"""
Statistics-agnostic OFDM channel estimator for one 14 x 72 slot (SISO, comb-2 DMRS on
symbols 2 and 11, 16QAM data).  See NOTES.md for the full story.

Outline
  1. Prior family.  The channel is modelled as zero-mean Gaussian with a Kronecker covariance
     R = R_t (x) R_f.  R_f comes from a family of power-delay profiles (exponential PDPs,
     optionally with a LOS spike or a second, shorter exponential); R_t from Jakes (J0) and
     flat (sinc) Doppler spectra.
  2. Pilot stage.  The exact Gaussian evidence of the 72 pilots is computed for every
     (PDP, Doppler) pair.  Instead of picking the best pair, the posterior over the pairs
     (Bayesian model averaging) gives a mixture prior R_f = sum_f w_f R_f, R_t = sum_d w_d R_t,
     and the pilot-only LMMSE estimate under that prior.
  3. Data-aided stage (12 iterations).  Every data RE is turned into a Gaussian "message"
     on its channel coefficient: EP messages during the first iterations at medium/high SNR
     (they tolerate a poor initial estimate), EM soft-symbol messages otherwise.  The full-grid
     posterior mean is then recomputed with a preconditioned CG solver in the eigen-basis of the
     Kronecker prior.  Tricks that make this robust:
       - progressive inclusion of data symbols by distance to the nearest pilot symbol
         (avoids locking onto 90-degree rotated constellations when the channel moves fast);
       - deterministic annealing of the noise level at high SNR (same purpose);
       - over-relaxed EM updates (faster convergence);
       - after all symbols are included, the Doppler and PDP mixture weights are re-estimated
         from the full-grid messages (evidence) and the prior basis is rebuilt;
       - the EM soft demapper uses the extrinsic (leave-one-out) channel estimate.
  4. Energy constraint.  The channel of every slot has unit average energy, so the final Gaussian
     posterior is tilted (h_i -> h_i / (1 + g v_i)) so that its expected energy is exactly one.
"""
import numpy as np
import torch
from scipy.special import j0
from link_config import (PILOT_MASK, PILOT_SYMBOLS, FFT_SIZE, NUM_OFDM_SYMBOLS,
                         SUBCARRIER_SPACING, OFDM_SYMBOL_DURATION, PILOT_OFDM_SYMBOLS)

torch.set_flush_denormal(True)
torch.set_num_threads(1)          # tiny tensors: threads only add overhead / contention
CD = torch.complex64
RD = torch.float32
K = FFT_SIZE
T = NUM_OFDM_SYMBOLS
KP = np.arange(0, K, 2)
PS = list(PILOT_OFDM_SYMBOLS)

# ------------------------------------------------------------------ hyper-parameters
SW_EP_DB = 6.0          # effective SNR (dB) above which the first iterations use EP messages
N_IT = 12               # data-aided iterations
N_EP = 4                # number of initial EP iterations (only above SW_EP_DB)
NCG = 3                 # CG iterations per data-aided iteration (warm started)
STAGES = [1, 1, 2, 2, 3, 3]   # inclusion radius (symbols from a pilot symbol) per iteration
ANN0, ANNR, ANN_N = 0.1, 0.7, 8   # noise-floor annealing: max(no, ANN0*ANNR**it) for it < ANN_N
OVR = 1.8               # EM over-relaxation factor (from iteration 1 on)
REW_IT = 7              # iteration at which the mixture weights are re-estimated
BMA_T = 0.75            # temperature of the pilot-evidence posterior over (PDP, Doppler)
RP = 16                 # eigen-components per PDP used in the pilot evidence
RWR = 16                # eigen-components per PDP used in the re-estimation evidence
TRUNC = 1e-7            # relative eigenvalue truncation of the prior basis

# ------------------------------------------------------------------ frequency covariance family
_FS = [2e-9 * 1.5 ** i for i in range(17)]                        # exp. mean delay 2 ns ... 1.3 us
FAM = [(s, w, 0.0) for s in _FS for w in [0.0, 0.5, 0.8, 0.95]]    # w * LOS spike + (1-w) * exp(s)
FAM += [(s, w, 0.15) for s in _FS for w in [0.3, 0.6]]            # w * exp(0.15 s) + (1-w) * exp(s)
_dk = (np.arange(K)[:, None] - np.arange(K)[None, :]) * SUBCARRIER_SPACING
RF_ALL = np.stack([w / (1.0 + 2j * np.pi * _dk * s * r) + (1 - w) / (1.0 + 2j * np.pi * _dk * s)
                   for s, w, r in FAM])                             # [F,72,72], unit diagonal
RF_T = torch.tensor(RF_ALL).to(CD)
_e, _V = np.linalg.eigh(RF_ALL[:, KP][:, :, KP])                    # pilot sub-grid
VPR = torch.tensor(_V[:, :, -RP:]).to(CD).conj().contiguous()       # [F,36,RP]
EPR = torch.tensor(np.maximum(_e[:, -RP:], 0)).to(RD)
_e, _V = np.linalg.eigh(RF_ALL)
VFR = torch.tensor(_V[:, :, -RWR:]).to(CD).conj().contiguous()      # [F,72,RWR]
EFR = torch.tensor(np.maximum(_e[:, -RWR:], 0)).to(RD)

# ------------------------------------------------------------------ time covariance family
_FDL = [0.0, 30, 80, 150, 250, 400, 600, 850, 1200, 1450, 1700, 2000, 2300]    # max Doppler [Hz]
FD = [('j', f) for f in _FDL] + [('s', f) for f in _FDL[1:]]                  # Jakes and flat
_tt = np.arange(T) * OFDM_SYMBOL_DURATION
_dt = _tt[:, None] - _tt[None, :]
RT = np.stack([j0(2 * np.pi * f * _dt) if k_ == 'j' else np.sinc(2 * f * _dt) for k_, f in FD])
RT = RT + 1e-6 * np.eye(T)
RT_T = torch.tensor(RT).to(CD)                                      # real, kept complex for convenience
_e, _V = np.linalg.eigh(RT)
ET = torch.tensor(np.maximum(_e, 0)).to(RD)
VT = torch.tensor(_V).to(CD)
RHO = RT_T[:, PS[0], PS[1]]                                         # correlation of the two pilot symbols
ARHO = RHO.abs().to(RD)
POS = RHO.real >= 0                                                # sign of the (real) pilot correlation

# ------------------------------------------------------------------ constants
XP = torch.tensor(PILOT_SYMBOLS[PS][:, KP]).to(CD)                  # [2,36]
XFULL = torch.tensor(PILOT_SYMBOLS).to(CD)
PMASK = torch.tensor(PILOT_MASK)
LV = torch.tensor([-3., -1., 1., 3.], dtype=RD) / np.sqrt(10.)
SR = LV[:, None].expand(4, 4).reshape(-1)                          # 16QAM points, real part
SI = LV[None, :].expand(4, 4).reshape(-1)                          # imaginary part
SA2 = SR ** 2 + SI ** 2
DIST = torch.tensor(np.min(np.abs(np.arange(T)[:, None] - np.array(PS)[None, :]), 1))


def _a2(x):
    return x.real ** 2 + x.imag ** 2


# ------------------------------------------------------------------ pilot stage
def _pilot_stage(y, no):
    B = y.shape[0]
    Zp = y[:, PS][:, :, KP] / XP                                    # LS on the pilots [B,2,36]
    # Gaussian evidence for every (PDP, Doppler) pair; the top-RP eigen-components of each PDP
    # are modelled, the rest is treated as noise only.  A 2x2 time covariance [[1,r],[r,1]] has
    # eigenvectors (1, +-1)/sqrt2 and eigenvalues 1 +- r, so the pilots split into sum / difference.
    p0 = torch.einsum('fji,bj->bfi', VPR, Zp[:, 0])                  # [B,F,RP]
    p1 = torch.einsum('fji,bj->bfi', VPR, Zp[:, 1])
    qa = (_a2(p0 + p1) / 2).permute(1, 0, 2)                         # [F,B,RP] (rho >= 0: "sum" part)
    qb = (_a2(p0 - p1) / 2).permute(1, 0, 2)
    ll = torch.empty(B, len(FAM), len(FD), dtype=RD)
    nou, inv = torch.unique(no, return_inverse=True)                 # tables once per distinct noise level
    for u in range(nou.shape[0]):
        bi = torch.nonzero(inv == u).reshape(-1)
        nv = nou[u]
        lam_s = EPR[:, None, :] * (1 + ARHO)[None, :, None] + nv     # [F,D,RP]
        lam_d = EPR[:, None, :] * (1 - ARHO)[None, :, None] + nv
        ltab = torch.log(lam_s * lam_d / nv ** 2).sum(-1)            # [F,D]
        cs, cd = 1.0 / lam_s - 1.0 / nv, 1.0 / lam_d - 1.0 / nv
        ca = torch.where(POS[None, :, None], cs, cd).transpose(1, 2)   # coefficient of qa [F,RP,D]
        cb = torch.where(POS[None, :, None], cd, cs).transpose(1, 2)
        l_u = torch.bmm(qa[:, bi], ca) + torch.bmm(qb[:, bi], cb)    # [F,b,D]
        ll[bi] = -(ltab[None] + l_u.permute(1, 0, 2))
    # Bayesian model averaging -> mixture prior (product of the marginal mixtures)
    post = torch.softmax(ll.reshape(B, -1) / BMA_T, 1).reshape(B, len(FAM), len(FD))
    Rf = torch.einsum('bf,fkl->bkl', post.sum(2).to(CD), RF_T)
    Rt = torch.einsum('bd,dkl->bkl', post.sum(1).to(CD), RT_T)
    # pilot-only LMMSE under the mixture prior
    e, V = torch.linalg.eigh(Rf[:, KP][:, :, KP])
    e = e.clamp(min=0)
    A = Rf[:, :, KP] @ V                                            # [B,72,36]
    rho = Rt[:, PS[0], PS[1]]
    ph = torch.exp(1j * torch.angle(rho)).to(CD)                    # e^{j arg rho}
    rho = rho.abs().to(RD)
    s = (Zp[:, 0] + ph[:, None] * Zp[:, 1]) / np.sqrt(2)
    d = (Zp[:, 0] - ph[:, None] * Zp[:, 1]) / np.sqrt(2)
    s_ = torch.einsum('bji,bj->bi', V.conj(), s)
    d_ = torch.einsum('bji,bj->bi', V.conj(), d)
    cs = (Rt[:, :, PS[0]] + Rt[:, :, PS[1]] * ph.conj()[:, None]) / np.sqrt(2)   # [B,14]
    cd = (Rt[:, :, PS[0]] - Rt[:, :, PS[1]] * ph.conj()[:, None]) / np.sqrt(2)
    gs = 1.0 / (e * (1 + rho)[:, None] + no[:, None])               # [B,36]
    gd = 1.0 / (e * (1 - rho)[:, None] + no[:, None])
    fs = torch.einsum('bki,bi->bk', A, gs * s_)
    fd_ = torch.einsum('bki,bi->bk', A, gd * d_)
    H = cs[:, :, None] * fs[:, None, :] + cd[:, :, None] * fd_[:, None, :]
    A2 = _a2(A)
    vs = torch.einsum('bki,bi->bk', A2, gs)
    vd = torch.einsum('bki,bi->bk', A2, gd)
    var = 1.0 - (_a2(cs)[:, :, None] * vs[:, None, :] + _a2(cd)[:, :, None] * vd[:, None, :])
    ef, vf = torch.linalg.eigh(Rf)
    et, vt = torch.linalg.eigh(Rt)
    return H, var.clamp(min=1e-7), _Basis(ef.clamp(min=0), vf, et.clamp(min=0), vt)


# ------------------------------------------------------------------ data messages
def _em_msg(y, H, var, no):
    """EM message: soft 16QAM symbol -> observation z of H with precision E|x|^2 / no."""
    g2 = _a2(H).clamp(min=1e-9)
    r = y * H.conj() / g2
    s2 = (no[:, None, None] + var) / g2

    q = 0.1 / s2                                                     # 16QAM = two 4-PAM {+-1,+-3}/sqrt10

    def pam(u):
        t = (2.0 / np.sqrt(10.0)) * u / s2
        ta = t.abs()
        l1p, l1m = ta - q, -ta - q
        l3p, l3m = 3 * ta - 9 * q, -3 * ta - 9 * q
        mx = torch.maximum(l1p, l3p)
        e1p, e1m = torch.exp(l1p - mx), torch.exp((l1m - mx).clamp(min=-80.0))
        e3p, e3m = torch.exp(l3p - mx), torch.exp((l3m - mx).clamp(min=-80.0))
        zi = 1.0 / (e1p + e1m + e3p + e3m)
        m = torch.sign(t) * (e1p - e1m + 3 * (e3p - e3m)) * zi / np.sqrt(10.0)
        return m, (e1p + e1m + 9 * (e3p + e3m)) * zi / 10.0
    mr, er = pam(r.real)
    mi, ei = pam(r.imag)
    ex2 = er + ei                                                    # E|x|^2
    z = torch.complex(mr, mi).conj() * y / ex2.clamp(min=1e-12)
    return z, ex2 / no[:, None, None]


def _ep_msg(y, m, var, no):
    """EP message on H from one 16QAM observation, given the cavity CN(m, var)."""
    inv = 1.0 / (no[:, None, None, None] + var[..., None] * SA2)    # [B,T,K,16]
    yr, yi = y.real[..., None], y.imag[..., None]
    mr, mi = m.real[..., None], m.imag[..., None]
    er = yr - (mr * SR - mi * SI)                                    # e = y - m s
    ei = yi - (mr * SI + mi * SR)
    d2 = er * er + ei * ei
    lg = -d2 * inv
    p = inv * torch.exp((lg - lg.amax(-1, keepdim=True)).clamp(min=-80.0))
    p = p / p.sum(-1, keepdim=True)                                  # posterior of the symbol
    pi = p * inv
    q1r = (pi * (SR * er + SI * ei)).sum(-1)                         # E[s^* e / den]
    q1i = (pi * (SR * ei - SI * er)).sum(-1)
    q2 = (pi * inv * SA2 * d2).sum(-1)                               # E|s^* e / den|^2
    v1 = pi.sum(-1) * no[:, None, None]
    mu = m + var * torch.complex(q1r, q1i)                           # tilted mean / variance
    vpost = (var * v1 + var ** 2 * (q2 - q1r ** 2 - q1i ** 2)).clamp(min=1e-12)
    prec = 1.0 / vpost - 1.0 / var
    eta = mu / vpost - m / var
    ok = (prec > 0) & torch.isfinite(prec) & torch.isfinite(eta.real) & torch.isfinite(eta.imag)
    z = torch.where(ok, eta / torch.where(ok, prec, torch.ones_like(prec)), torch.zeros_like(eta))
    prec = torch.where(ok, prec, torch.zeros_like(prec))
    return z, torch.minimum(prec, 1.8 / no[:, None, None])          # one RE cannot say more


# ------------------------------------------------------------------ full-grid LMMSE
class _Basis:
    """Kronecker prior R_t (x) R_f in its (truncated) eigen-basis: H = Qt (S * C) Qf^T."""

    def __init__(self, ef, vf, et, vt):
        rf = int((ef > TRUNC * ef[:, -1:]).sum(1).max())
        rt = int((et > TRUNC * et[:, -1:]).sum(1).max())
        self.Qf = vf[:, :, -rf:]
        self.Qt = vt[:, :, -rt:]
        self.Ef = ef[:, -rf:]
        self.Et = et[:, -rt:]
        self.S = (self.Et[:, :, None] * self.Ef[:, None, :]).sqrt()   # [B,rt,rf]
        self.QfT = self.Qf.transpose(1, 2)
        self.QtH = self.Qt.conj().transpose(1, 2)
        self.Qfc = self.Qf.conj()
        self.G = self.Qt * self.Et.sqrt().to(CD)[:, None, :]
        self.Qf2T = _a2(self.Qf).transpose(1, 2)

    def fwd(self, C):
        return self.Qt @ (self.S * C) @ self.QfT

    def adj(self, G):
        return self.S * (self.QtH @ G @ self.Qfc)


def _pcg(bs, w, z, C, n_it):
    """(I + S Q^H W Q S) C = S Q^H W z by PCG; the preconditioner (exact when W only varies
    per OFDM symbol) also gives the approximate posterior variance per RE."""
    wt = w.mean(2).to(CD)
    G = bs.G
    GWG = G.conj().transpose(1, 2) @ (wt[:, :, None] * G)
    g, E = torch.linalg.eigh(GWG.to(torch.complex128))
    g = g.clamp(min=0).to(RD)
    E = E.to(CD)
    Dinv = 1.0 / (1.0 + g[:, :, None] * bs.Ef[:, None, :])
    Eh = E.conj().transpose(1, 2)

    def Pc(R):
        return E @ (Dinv * (Eh @ R))

    def Aop(X):
        return X + bs.adj(w * bs.fwd(X))
    r = bs.adj(w * z) - Aop(C)
    zr = Pc(r)
    p = zr
    rz = (r.conj() * zr).real.sum((1, 2))
    for _ in range(n_it):
        Ap = Aop(p)
        alpha = rz / (p.conj() * Ap).real.sum((1, 2)).clamp(min=1e-30)
        C = C + alpha[:, None, None] * p
        r = r - alpha[:, None, None] * Ap
        zr = Pc(r)
        rz_new = (r.conj() * zr).real.sum((1, 2))
        p = zr + (rz_new / rz.clamp(min=1e-30))[:, None, None] * p
        rz = rz_new
    cov = _a2(G @ E) @ (Dinv * bs.Ef[:, None, :])                   # [B,14,rf]
    return C, (cov @ bs.Qf2T).clamp(min=1e-9)


def _reweight(bs, w, z):
    """Re-estimate the Doppler and PDP mixture weights from the full-grid messages
    (Gaussian evidence with the mean message precision) and rebuild the prior basis."""
    nv = 1.0 / w.mean((1, 2))
    Zf = z @ bs.Qfc
    Y = torch.einsum('dta,btx->bdax', VT.conj(), Zf)                # [B,D,14,rf]
    lam = ET[None, :, :, None] * bs.Ef[:, None, None, :] + nv[:, None, None, None]
    ll = -(torch.log(lam) + _a2(Y) / lam).sum((-1, -2))
    Rt = torch.einsum('bd,dkl->bkl', torch.softmax(ll, 1).to(CD), RT_T)
    et, vt = torch.linalg.eigh(Rt)
    et = et.clamp(min=0)
    Zt = vt.conj().transpose(1, 2) @ z
    Y = torch.einsum('bak,fkx->bfax', Zt, VFR)                      # [B,F,14,RWR]
    nv4 = nv[:, None, None, None]
    lam = et[:, None, :, None] * EFR[None, :, None, :] + nv4
    y2 = _a2(Y)
    llf = -(torch.log(lam / nv4) + y2 / lam - y2 / nv4).sum((-1, -2))
    Rf = torch.einsum('bf,fkl->bkl', torch.softmax(llf, 1).to(CD), RF_T)
    ef, vf = torch.linalg.eigh(Rf)
    return _Basis(ef.clamp(min=0), vf, et, vt)


def _data_aided(y, no, H, var, bs, use_ep_first):
    B = y.shape[0]
    C = torch.zeros(B, bs.S.shape[1], bs.S.shape[2], dtype=CD)
    zp = y * XFULL.conj()                                            # pilots: |x| = 1
    wprec = torch.zeros(B, T, K, dtype=RD)
    zmsg = torch.zeros(B, T, K, dtype=CD)
    for it in range(N_IT):
        rad = STAGES[it] if it < len(STAGES) else 99
        act = torch.nonzero(DIST <= rad).reshape(-1)                 # active OFDM symbols
        noi = torch.clamp(no, min=ANN0 * ANNR ** it) if it < ANN_N else no
        ya, Ha, va = y[:, act], H[:, act], var[:, act]
        use_ep = use_ep_first & (it < N_EP)                          # [B]
        if bool(use_ep.any()):
            wa, za = wprec[:, act], zmsg[:, act]
            vc = 1.0 / (1.0 / va - wa).clamp(min=1e-6)               # cavity
            mc = vc * (Ha / va - wa * za)
            z_ep, p_ep = _ep_msg(ya, mc, vc, noi)
        if not bool(use_ep.all()):
            # the soft demapper sees the extrinsic (leave-one-out) channel estimate
            wa, za = wprec[:, act], zmsg[:, act]
            vc = 1.0 / (1.0 / va - wa).clamp(min=1e-6)
            mc = vc * (Ha / va - wa * za)
            z_em, p_em = _em_msg(ya, mc, vc, noi)
        if bool(use_ep.all()):
            pa, za = p_ep, z_ep
        elif not bool(use_ep.any()):
            pa, za = p_em, z_em
        else:
            u = use_ep[:, None, None]
            pa, za = torch.where(u, p_ep, p_em), torch.where(u, z_ep, z_em)
        wprec = torch.zeros(B, T, K, dtype=RD)
        zmsg = torch.zeros(B, T, K, dtype=CD)
        wprec[:, act] = pa
        zmsg[:, act] = za
        w = torch.where(PMASK, (1.0 / noi)[:, None, None].expand(B, T, K), wprec)
        z = torch.where(PMASK, zp, zmsg)
        if it == REW_IT:
            bs = _reweight(bs, w, z)
            C = bs.S * (bs.QtH @ H @ bs.Qfc) / (bs.S ** 2 + 1e-4)   # warm start in the new basis
            C, var = _pcg(bs, w, z, C, NCG)
        else:
            C_old = C
            C, var = _pcg(bs, w, z, C, NCG)
            if it >= 1:
                C = C_old + OVR * (C - C_old)                        # over-relaxation
        H = bs.fwd(C)
    return bs.fwd(C)  # ABLATION (added by analysis, not the agent): energy constraint removed


def _energy_constraint(bs, w, C):
    """Posterior mean under the (known) unit average channel energy, via exponential tilting
    of the Gaussian posterior: h_i -> h_i / (1 + g v_i) with g such that E||h||^2 = T*K."""
    wbar = w.mean((1, 2))
    lam = bs.S ** 2                                                   # prior variances [B,rt,rf]
    v = lam / (1.0 + lam * wbar[:, None, None])                       # posterior variances
    h2 = _a2(bs.S * C)                                                # |posterior mean|^2
    N = float(T * K)
    lo = -1.0 / v.amax((1, 2)) * 0.999                               # keep 1 + g v > 0
    hi = torch.full_like(lo, 1e4)
    g = torch.zeros_like(lo)
    for _ in range(40):                                               # bisection on g (E decreasing in g)
        g = 0.5 * (lo + hi)
        den = 1.0 + g[:, None, None] * v
        E = (h2 / den ** 2 + v / den).sum((1, 2))
        big = E > N
        lo = torch.where(big, g, lo)
        hi = torch.where(big, hi, g)
    den = 1.0 + g[:, None, None] * v
    return bs.fwd(C / den)


def estimate_channel(y, no):
    y = torch.as_tensor(y).to(CD)
    no = torch.as_tensor(no).to(RD).reshape(-1).clamp(min=1e-6)
    H, var, bs = _pilot_stage(y, no)
    P = (_a2(y).mean((1, 2)) - no).clamp(min=1e-3)                  # received channel power
    use_ep_first = 10 * torch.log10(P / no) >= SW_EP_DB
    H = _data_aided(y, no, H, var, bs, use_ep_first)
    return H.to(torch.complex64)
