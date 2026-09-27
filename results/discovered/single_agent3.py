"""
Single-slot, statistics-agnostic OFDM channel estimator (14 x 72 grid, comb-2 pilots on symbols 2 and 11).

Pipeline (all per slot, no pooling across slots):
  1. Pilot stage   : Bayesian-model-averaged 2-D LMMSE from the 72 pilots.  Hypotheses = generic
                     power-delay-profile (PDP) candidates x (sum/diff-mode powers of the two pilot
                     symbols <-> power and Jakes Doppler).  Gives an estimate + per-symbol error variance.
  2. Sequential DD : (SNR >= -2.5 dB) GP/Jakes prediction outward from the pilot symbols, one symbol
                     at a time, with soft 16-QAM decisions and a frequency-domain LMMSE update.
  3. EM stage      : 7 iterations of  soft 16-QAM demapping (E-step)  ->  pseudo-observations on all
                     1008 REs  ->  marginal-likelihood Bayesian averaging of the frequency covariance
                     (PDP candidates) and of the time covariance (Jakes candidates)  ->  heteroscedastic
                     Kronecker-structured LMMSE solved by warm-started PCG (M-step).
"""
import numpy as np
import torch
from scipy.special import j0 as _j0
from scipy.special import j1 as _j1
from link_config import (PILOT_MASK, PILOT_SYMBOLS, FFT_SIZE, NUM_OFDM_SYMBOLS,
                         PILOT_OFDM_SYMBOLS, OFDM_SYMBOL_DURATION)

K = FFT_SIZE
S = NUM_OFDM_SYMBOLS
PS = list(PILOT_OFDM_SYMBOLS)
KP = np.arange(0, K, 2)
NP = len(KP)
C = torch.complex64
F = torch.float32


def a2(x):
    return x.real ** 2 + x.imag ** 2


# ----------------------------------------------------------------------------------
# Candidate power-delay profiles (generic families); delays in samples of 1/(K*df)
# ----------------------------------------------------------------------------------
TAU = np.arange(0, 34 * 8) / 8.0


def _exp(ds):
    return np.exp(-TAU / ds)


def _build_pdps():
    c, fam = [], []
    for ds in np.geomspace(0.02, 4.0, 16):                     # exponential
        c.append(_exp(ds)); fam.append(0)
    for tm in np.geomspace(0.5, 20.0, 8):                      # uniform
        c.append((TAU <= tm).astype(float)); fam.append(1)
    for d1 in np.geomspace(0.05, 2.0, 10):                     # two-slope exponential
        for rr in (4.0, 10.0):
            for a in (0.05, 0.2):
                e1 = _exp(d1); e1 = e1 / e1.sum(); e2 = _exp(d1 * rr); e2 = e2 / e2.sum()
                c.append((1 - a) * e1 + a * e2); fam.append(2)
    for kf in (2.0, 8.0):                                      # specular spike + exponential tail
        for ds in np.geomspace(0.1, 2.5, 6):
            e = _exp(ds); e = e / e.sum()
            sp = np.zeros_like(TAU); sp[0] = 1.0
            c.append(kf / (kf + 1) * sp + 1 / (kf + 1) * e); fam.append(3)
    c = np.stack(c)
    return c / c.sum(1, keepdims=True), np.array(fam)


PDP, FAM = _build_pdps()
NC = PDP.shape[0]
_AF = np.exp(-2j * np.pi * np.outer(np.arange(K), TAU) / K)
_AP = _AF[KP]
_RF = np.einsum('kg,ng,lg->nkl', _AF, PDP, _AF.conj())         # full-band covariances (trace K)
_lf, _uf = np.linalg.eigh(_RF)
RF_FULL_C = torch.tensor(_RF, dtype=C)
LF_C = torch.tensor(_lf[:, ::-1].clip(min=0).copy(), dtype=F)  # [NC,K] descending
UF_C = torch.tensor(_uf[:, :, ::-1].copy(), dtype=C)          # [NC,K,K]
_RP = np.einsum('kg,ng,lg->nkl', _AP, PDP, _AP.conj())         # pilot-subcarrier covariances
_lp, _up = np.linalg.eigh(_RP)
_RFP = np.einsum('kg,ng,lg->nkl', _AF, PDP, _AP.conj())        # full-band x pilot cross covariances
PSEL = torch.tensor(np.flatnonzero(np.isin(FAM, (0, 3))))     # candidate subset used by the pilot stage
NCP = len(PSEL)
LP_S = torch.tensor(_lp.clip(min=0), dtype=F)[PSEL]
UP_S = torch.tensor(_up, dtype=C)[PSEL]
RPP_S = torch.tensor(_RP, dtype=C)[PSEL]
RFP_S = torch.tensor(_RFP, dtype=C)[PSEL]

XP = torch.tensor(np.stack([PILOT_SYMBOLS[s, KP] for s in PS], 0), dtype=C)   # [2,NP]
PM = torch.tensor(PILOT_MASK)
XFULL = torch.tensor(PILOT_SYMBOLS, dtype=C)

_xx = np.linspace(0, 3.8317, 4000)           # J0 inverse on its first branch (down to its minimum)
_JJ = _j0(_xx)


def jinv(r):
    return np.interp(-np.clip(r, -0.4027, 1.0), -_JJ, _xx)


# ----------------------------------------------------------------------------------
# 1. Pilot stage: Bayesian averaging over (sum-mode power u, diff-mode power v) and PDPs
# ----------------------------------------------------------------------------------
NU = 32
UGR = np.geomspace(1e-4, 4.0, NU)
UG = torch.tensor(UGR, dtype=F)
_R9 = (UGR[:, None] - UGR[None, :]) / (UGR[:, None] + UGR[None, :])    # implied corr. between pilot symbols
_X9 = jinv(_R9)                                                          # implied 2*pi*fD*9T
_DL = np.abs(np.arange(S)[:, None] - np.array(PS)[None, :])
_RS = _j0(_X9[:, :, None, None] * _DL[None, None] / 9.0)               # [NU,NU,S,2]
ALT = torch.tensor((_RS[..., 0] + _RS[..., 1]) / np.sqrt(2), dtype=F)   # time weights, sum mode
BET = torch.tensor((_RS[..., 0] - _RS[..., 1]) / np.sqrt(2), dtype=F)   # time weights, diff mode
X9T = torch.tensor(_X9, dtype=F)
PHALF = torch.tensor((UGR[:, None] + UGR[None, :]) / 2, dtype=F)
# prior: uniform in the Jakes argument x; hypotheses without a J0 solution excluded
_lpr = np.log(1 - _R9 ** 2 + 1e-9) - np.log(_j1(_X9) + 1e-4)
LPRIOR = torch.tensor(np.where(_R9 >= -0.4027, _lpr, -1e30), dtype=F)


def pilot_stage(y, no):
    B = y.shape[0]
    Yp = torch.stack([y[:, s, KP] for s in PS], 2) / XP.T.unsqueeze(0)              # LS on pilots [B,NP,2]
    pt = a2(Yp).mean((1, 2))
    ysd = torch.stack([Yp[:, :, 0] + Yp[:, :, 1], Yp[:, :, 0] - Yp[:, :, 1]], -1) / np.sqrt(2.0)
    ug = pt.view(B, 1) * UG.view(1, NU)
    # evidence of each PDP candidate (marginalised over u, v)
    Ca = a2(torch.einsum('njk,bjm->bnkm', UP_S.conj(), ysd))                        # [B,NCP,NP,2]
    d = LP_S.view(1, NCP, 1, NP) * ug.view(B, 1, NU, 1) + no.view(B, 1, 1, 1)
    ld = torch.log(d)
    LLs = -(ld + Ca[..., 0].unsqueeze(2) / d).sum(-1)
    LLd = -(ld + Ca[..., 1].unsqueeze(2) / d).sum(-1)
    sc = torch.logsumexp(LLs, -1) + torch.logsumexp(LLd, -1)
    i = sc.argmax(1)
    # posterior-weighted frequency covariance
    wf = torch.softmax(sc, 1).to(C)
    Rpp = torch.einsum('bn,nkl->bkl', wf, RPP_S)
    Rfp = torch.einsum('bn,nkl->bkl', wf, RFP_S)
    li, U = torch.linalg.eigh(Rpp)
    li = li.clamp(min=0)
    ci = U.conj().transpose(1, 2) @ ysd
    RFPU = Rfp @ U
    nrm = a2(RFPU).sum(1)
    d2 = li.view(B, 1, NP) * ug.view(B, NU, 1) + no.view(B, 1, 1)
    ld2 = torch.log(d2)
    Ls = -(ld2 + a2(ci[:, :, 0]).unsqueeze(1) / d2).sum(-1)
    Ld = -(ld2 + a2(ci[:, :, 1]).unsqueeze(1) / d2).sum(-1)
    W = torch.softmax((Ls.unsqueeze(2) + Ld.unsqueeze(1) + LPRIOR).reshape(B, -1), -1).reshape(B, NU, NU)
    # BMA of the per-hypothesis 2-D LMMSE estimates (closed form in the sum/diff eigenbasis)
    P = pt.view(B, 1, 1) * PHALF.view(1, NU, NU)
    iu = 1.0 / (ug.view(B, NU, 1) * li.view(B, 1, NP) + no.view(B, 1, 1))
    WP = W * P
    Qs = torch.einsum('bav,avs->bas', WP, ALT)
    Qd = torch.einsum('bav,avs->bvs', WP, BET)
    As = torch.einsum('bas,bak->bsk', Qs, iu)
    Ad = torch.einsum('bvs,bvk->bsk', Qd, iu)
    cs = ci[:, :, 0].unsqueeze(1); cd = ci[:, :, 1].unsqueeze(1)
    Z1 = As * cs + Ad * cd
    H = Z1 @ RFPU.transpose(1, 2)
    # error variance per symbol = E_h[LMMSE error] + spread of the hypothesis estimates
    WP2 = W * P ** 2
    Q2s = torch.einsum('bav,avs->bas', WP2, ALT ** 2)
    Q2d = torch.einsum('bav,avs->bvs', WP2, BET ** 2)
    a2s = torch.einsum('bas,bak->bsk', Q2s, iu ** 2)
    a2d = torch.einsum('bvs,bvk->bsk', Q2d, iu ** 2)
    asd = torch.einsum('bavs,bak,bvk->bsk', WP2.unsqueeze(-1) * (ALT * BET).unsqueeze(0), iu, iu)
    E2 = a2s * a2(cs) + a2d * a2(cd) + 2 * asd * (cs * cd.conj()).real
    spread = ((E2 - a2(Z1)).clamp(min=0) * nrm.unsqueeze(1)).sum(-1) / K
    en = (iu * nrm.unsqueeze(1)).sum(-1) / K
    errs = (WP.sum((1, 2))).view(B, 1) - (Q2s * en.unsqueeze(-1)).sum(1) - (Q2d * en.unsqueeze(-1)).sum(1)
    err = errs.clamp(min=0) + spread + 1e-6
    xm = (W * X9T.unsqueeze(0)).sum((1, 2))
    return H, err, PSEL[i], xm


# ----------------------------------------------------------------------------------
# Soft 16-QAM demapper (closed form, numerically stable)
# ----------------------------------------------------------------------------------
_A = float(1.0 / np.sqrt(10.0))


def demap_soft(y, H, v):
    """posterior mean and 2nd moment of unit-energy 16-QAM x given y = H x + n, n ~ CN(0, v)."""
    g2 = a2(H)
    z = y * H.conj()
    iv = 1.0 / v
    t = (2 * _A) * torch.stack([z.real, z.imag], 0) * iv
    q = (g2 * iv) * (_A * _A)
    at = t.abs()
    l1 = at - q
    l3 = 3 * at - 9 * q
    M = torch.maximum(l1, l3)
    w1p = torch.exp(l1 - M); w1m = torch.exp(-at - q - M)
    w3p = torch.exp(l3 - M); w3m = torch.exp(-3 * at - 9 * q - M)
    iz = 1.0 / (w1p + w1m + w3p + w3m)
    mm = torch.sign(t) * (_A * (w1p - w1m + 3 * (w3p - w3m)) * iz)
    m2 = ((_A * _A) * (w1p + w1m + 9 * (w3p + w3m)) * iz).sum(0)
    return torch.complex(mm[0], mm[1]), m2


# ----------------------------------------------------------------------------------
# 2. Sequential decision-directed tracking outward from the pilot symbols
# ----------------------------------------------------------------------------------
ROUNDS = [[1, 3, 10, 12], [0, 4, 9, 13], [5, 8], [6, 7]]
SEQ_SNR = -2.5          # dB; stage used at and above this SNR
EFLOOR = 0.03
N_INNER = 3
ERR_INFL = 3.0
INNER_DECAY = 0.5
FD_INFL = 1.2
BETA_MAX = 0.5          # weight of a maximum-Doppler Jakes kernel mixed into the GP kernel (robustness)
VAR_INFL = 1.5
RF_THR = 1e-6
_JMAX = _j0(3.8317 * np.abs(np.arange(-S, S + 1)) / 9)


def trunc_basis(i):
    lf = LF_C[i]
    r = int((lf > RF_THR * K).sum(-1).max())
    return lf[:, :r].contiguous(), UF_C[i][:, :, :r].contiguous()


def seq_stage(y, no, H, err, lf, Uf, xa, P):
    B = y.shape[0]
    Hk = H.clone(); ek = err.clone()
    dl = np.arange(-S, S + 1)
    xx = np.clip(xa * FD_INFL, 0, 3.8317)
    rt = (1 - BETA_MAX) * _j0(np.outer(xx, np.abs(dl)) / 9) + BETA_MAX * _JMAX[None]
    rt = torch.tensor(rt, dtype=C)
    known = list(PS)
    Pc = P.to(C)
    for rnd in ROUNDS:
        O = torch.tensor(known)
        T = torch.tensor(rnd)
        KOO = Pc.view(B, 1, 1) * rt[:, (O.view(-1, 1) - O.view(1, -1)) + S] \
            + torch.diag_embed(torch.maximum(ek[:, O], EFLOOR * P.view(B, 1))).to(C)
        kTO = Pc.view(B, 1, 1) * rt[:, (T.view(-1, 1) - O.view(1, -1)) + S]
        Wt = torch.linalg.solve(KOO, kTO.transpose(1, 2)).transpose(1, 2)          # GP prediction weights
        m = Wt @ Hk[:, O]
        var = (P.view(B, 1) - (Wt * kTO.conj()).sum(-1).real).clamp(min=1e-6) * VAR_INFL
        yT = y[:, T]
        cur = m; cv = var
        for inner in range(N_INNER):
            xs, m2 = demap_soft(yT, cur, (no.view(B, 1) + cv).unsqueeze(-1))
            ht = yT * xs.conj() / m2
            w = (m2 / no.view(B, 1, 1)).to(C)
            wb = w.real.mean(-1, keepdim=True)
            dg = 1.0 / (wb + 1.0 / (var.unsqueeze(-1) * lf.unsqueeze(1)).clamp(min=1e-12))
            c = ((w * (ht - m)) @ Uf.conj()) * dg                                       # prior N(m, var*Rf)
            cur = m + c @ Uf.transpose(1, 2)
            e_new = dg.sum(-1) / K
            cv = torch.maximum(e_new * ERR_INFL, var * INNER_DECAY ** (inner + 1))
        Hk[:, T] = cur
        ek[:, T] = cv
        known = known + rnd
    return Hk, ek


# ----------------------------------------------------------------------------------
# 3. EM stage: covariance model averaging + Kronecker LMMSE (PCG)
# ----------------------------------------------------------------------------------
FD_GRID = np.concatenate([[0.0], np.geomspace(15.0, 2500.0, 23)])
NT = len(FD_GRID)
_tt = np.arange(S) * OFDM_SYMBOL_DURATION
_RTC = np.stack([_j0(2 * np.pi * f * np.abs(_tt[:, None] - _tt[None, :])) for f in FD_GRID]) + 1e-5 * np.eye(S)
_lt, _ut = np.linalg.eigh(_RTC)
RT_FULL_C = torch.tensor(_RTC, dtype=C)
LT_C = torch.tensor(_lt[:, ::-1].clip(min=0).copy(), dtype=F)
UT_C = torch.tensor(_ut[:, :, ::-1].copy(), dtype=C)
NTOP = 2


def rf_weights(hte, nv, jt, P):
    """posterior weights of the PDP candidates (time model jt fixed)."""
    B = hte.shape[0]
    lt = LT_C[jt][:, :NTOP]
    Ut = UT_C[jt][:, :, :NTOP]
    Zt = Ut.conj().transpose(1, 2) @ hte
    Cc = torch.einsum('bak,nkf->bnaf', Zt, UF_C.conj())
    d = lt.view(B, 1, NTOP, 1) * LF_C.view(1, NC, 1, K) * P.view(B, 1, 1, 1) + nv.view(B, 1, 1, 1)
    LL = -(torch.log(d) + a2(Cc) / d).sum((2, 3))
    return torch.softmax(LL, 1)


def rt_weights(hte, nv, lf, Uf, P):
    """posterior weights of the Jakes time-correlation candidates (frequency model fixed)."""
    B = hte.shape[0]
    Zf = hte @ Uf.conj()
    Cj = torch.einsum('jsa,bsr->bjar', UT_C.conj(), Zf)
    d = LT_C.view(1, NT, S, 1) * lf.view(B, 1, 1, -1) * P.view(B, 1, 1, 1) + nv.view(B, 1, 1, 1)
    LL = -(torch.log(d) + a2(Cj) / d).sum((2, 3))
    return torch.softmax(LL, 1)


def kron_pcg(z, w, lf, Uf, lt, Ut, P, iters, h0):
    """argmin sum w|z-h|^2 + prior, h = Ut c Uf^T with prior var lt*lf*P; warm-started PCG."""
    lam = (lt.unsqueeze(2) * lf.unsqueeze(1) * P.view(-1, 1, 1)).clamp(min=1e-12)
    wc = w.to(C)
    UtH = Ut.conj().transpose(1, 2); UfT = Uf.transpose(1, 2); Ufc = Uf.conj()

    def fwd(c):
        return Ut @ c @ UfT

    def adj(h):
        return UtH @ h @ Ufc

    wbar = w.mean((1, 2)).view(-1, 1, 1)
    Minv = 1.0 / (wbar + 1.0 / lam)
    b = adj(wc * z)
    c = adj(h0)
    r = b - (adj(wc * fwd(c)) + c / lam)
    zz = Minv * r
    p = zz
    rz = (r.conj() * zz).real.sum((1, 2), keepdim=True)
    for _ in range(iters):
        Ap = adj(wc * fwd(p)) + p / lam
        alpha = rz / (p.conj() * Ap).real.sum((1, 2), keepdim=True).clamp(min=1e-30)
        c = c + alpha * p
        r = r - alpha * Ap
        zz = Minv * r
        rz_new = (r.conj() * zz).real.sum((1, 2), keepdim=True)
        p = zz + (rz_new / rz.clamp(min=1e-30)) * p
        rz = rz_new
    err = (a2(Ut) @ Minv.sum(-1, keepdim=True)).squeeze(-1) / K
    return fwd(c), err


N_IT = 7
RF_SEL_ITS = (0, 2)
SEL_SNR = -2.5
PCG_IT = 4
PCG_IT2 = 2


def _est(y, no):
    y = y.to(C)
    no = no.to(F).reshape(-1).clamp(min=1e-7)
    B = y.shape[0]
    H, err, fi, xm = pilot_stage(y, no)
    H0 = H
    P = torch.clamp(a2(y).mean((1, 2)) - no, min=1e-3)
    lf, Uf = trunc_basis(fi)
    hs = no <= 10 ** (-SEQ_SNR / 10)
    if bool(hs.any()):
        H2, e2 = seq_stage(y, no, H, err, lf, Uf, xm.numpy().astype(np.float64), P)
        H = torch.where(hs.view(B, 1, 1), H2, H)
        err = torch.where(hs.view(B, 1), e2, err)
    fd0 = xm.numpy() / (2 * np.pi * 9 * OFDM_SYMBOL_DURATION)
    jt = torch.tensor(np.abs(FD_GRID[None, :] - fd0[:, None]).argmin(1))
    corr_on = (no <= 10 ** (-SEL_SNR / 10)).float()
    for it in range(N_IT):
        # E-step: soft symbols -> pseudo-observations of the channel on every RE
        xs, m2 = demap_soft(y, H, no.view(B, 1, 1) + err.view(B, S, 1))
        xs = torch.where(PM, XFULL, xs)
        m2 = torch.where(PM, torch.ones_like(m2), m2)
        hte = y * xs.conj() / m2
        nvr = no.view(B, 1, 1) / m2
        nvm = nvr.mean((1, 2))
        # account for the shrinkage of soft pseudo-observations when scoring covariance models
        u = (1 - a2(xs) / m2).clamp(min=0).mean((1, 2)) * corr_on
        Ps = (1 - u) ** 2 * P
        nvs = nvm + 0.5 * u * (1 - u) * P
        if it in RF_SEL_ITS:
            wf = rf_weights(hte, nvs, jt, Ps)
            lb, ub = torch.linalg.eigh(torch.einsum('bn,nkl->bkl', wf.to(C), RF_FULL_C))
            lb = lb.flip(-1).clamp(min=0); ub = ub.flip(-1)
            r = int((lb > RF_THR * K).sum(-1).max())
            lf, Uf = lb[:, :r].contiguous(), ub[:, :, :r].contiguous()
        wt = rt_weights(hte, nvs, lf, Uf, Ps)
        ltb, utb = torch.linalg.eigh(torch.einsum('bn,nkl->bkl', wt.to(C), RT_FULL_C))
        ltb = ltb.flip(-1).clamp(min=0); utb = utb.flip(-1)
        jt = wt.argmax(1)
        # M-step
        H, err = kron_pcg(hte, 1.0 / nvr, lf, Uf, ltb, utb, P, PCG_IT if it == 0 else PCG_IT2, H)
    # safety net: fall back to the pilot-stage estimate for any slot with non-finite values
    bad = ~torch.isfinite(torch.view_as_real(H)).all(-1).all(-1).all(-1)
    if bool(bad.any()):
        H = torch.where(bad.view(B, 1, 1), torch.nan_to_num(H0), H)
    return H


def estimate_channel(y, no):
    nt = torch.get_num_threads()
    torch.set_num_threads(1)
    try:
        with torch.no_grad():
            return _est(torch.as_tensor(y), torch.as_tensor(no)).to(torch.complex64)
    finally:
        torch.set_num_threads(nt)
