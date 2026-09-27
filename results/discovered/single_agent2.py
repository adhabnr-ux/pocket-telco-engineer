"""
Statistics-agnostic OFDM channel estimator (single-slot).

Pipeline per slot (see NOTES.md):
  1. Pilot frequency estimate: Bayesian model averaging over a dictionary of power-delay-profile
     (PDP) covariances (3GPP TDL-shaped profiles over a grid of delay spreads + generic shapes),
     weighted by their exact Gaussian evidence on the 72 pilots.
  2. Doppler model averaging: Jakes / LOS+Jakes time-correlation candidates scored by the
     16-QAM data likelihood of their pilot-only predictions on the data REs.
  3. Data-aided 2-D LMMSE with a Kronecker prior Rt x Rf (preconditioned CG in the eigenbasis),
     alternating soft-symbol messages: EP messages (high SNR / high Doppler) and over-relaxed
     variational-EM messages (low SNR). High-Doppler slots are initialised by a sequential
     symbol-by-symbol decision-directed pass that propagates outward from the pilot symbols.
"""
import time
import numpy as np, torch
from scipy.special import j0
from link_config import *

N = FFT_SIZE
T = NUM_OFDM_SYMBOLS
PS = list(PILOT_OFDM_SYMBOLS)
KP = np.arange(PILOT_SUBCARRIER_OFFSET, N, PILOT_SUBCARRIER_SPACING)
NP = len(KP)
TS = OFDM_SYMBOL_DURATION
CD = torch.complex64
RD = torch.float32
FAST32 = True
TT = np.arange(T)
PIL = torch.tensor(PILOT_SYMBOLS[PS][:, KP], dtype=CD)
PMASK = torch.tensor(PILOT_MASK)
DMASK = ~PMASK
PGRID = torch.tensor(PILOT_SYMBOLS, dtype=CD)
LV = torch.tensor([-3.0, -1.0, 1.0, 3.0], dtype=RD) / np.sqrt(10.0)
QAM = (LV[:, None] + 1j * LV[None, :]).reshape(-1).to(CD)
QE = QAM.abs() ** 2
QR = QAM.real.contiguous()
QI = QAM.imag.contiguous()
SQ10 = np.sqrt(10.0)


def hpd_inv(M):
    # robust inverse of Hermitian PD matrices (float64 internally)
    Md = M.to(torch.complex128)
    Md = 0.5 * (Md + Md.conj().transpose(-1, -2))
    L, info = torch.linalg.cholesky_ex(Md)
    if bool((info != 0).any()):
        sc = torch.diagonal(Md, dim1=-2, dim2=-1).real.mean(-1).clamp(min=1e-30)
        eye = torch.eye(Md.shape[-1], dtype=Md.dtype)
        for j in (1e-9, 1e-7, 1e-5, 1e-3):
            L2, info2 = torch.linalg.cholesky_ex(Md + (j * sc)[..., None, None] * eye)
            bad = (info != 0)
            L = torch.where(bad[..., None, None], L2, L)
            info = torch.where(bad, info2, info)
            if not bool((info != 0).any()):
                break
    return torch.cholesky_inverse(L).to(M.dtype)


def hpd_inv32(M):
    # fast float32 path; falls back to float64 robust inverse if any factorization fails
    L, info = torch.linalg.cholesky_ex(M)
    if bool((info != 0).any()):
        return hpd_inv(M)
    return torch.cholesky_inverse(L)


def a2(z):
    return z.real ** 2 + z.imag ** 2


def _lc(u):
    u = u.abs()
    return u + torch.log1p(torch.exp(-2 * u))


def _lse2(A1, A2):
    mx = torch.maximum(A1, A2)
    return mx + torch.log1p(torch.exp(-(A1 - A2).abs()))

# ---------------- frequency-domain covariance candidates (PDP shapes) -------------
TAU = np.arange(-3.0, 34.0 + 1e-9, 0.25)
FD = np.exp(-2j * np.pi * np.arange(N)[:, None] * TAU[None, :] / N)


def _pdp_cands():
    c = []
    for t0 in [-0.5, 0.0]:
        for L in [0.25, 0.5, 1, 2, 3, 4, 6, 8, 12, 16, 24, 32]:
            u = ((TAU >= t0 - 1e-9) & (TAU <= t0 + L + 1e-9)).astype(float)
            ex = np.where(TAU >= t0 - 1e-9, np.exp(-(TAU - t0) / max(L / 2, 0.1)), 0.)
            ex[TAU > t0 + 4 * L + 1] = 0
            sp = ((TAU >= t0 - 1e-9) & (TAU <= t0 + 0.25 + 1e-9)).astype(float)
            for shape in (u, ex):
                shape = shape / shape.sum()
                c.append(shape)
                if L >= 1:
                    for b in (0.9,):
                        m = b * sp / sp.sum() + (1 - b) * shape
                        c.append(m / m.sum())
    return np.array(c)


PC = _pdp_cands()
# 3GPP TR 38.901 / TS 38.101-4 TDL power-delay profiles (public standard tables):
# name -> (scale_delays, los, normalized delays or delays [ns], powers [dB])
TDL_TABLES = {'A': (True, 0, [0.0, 0.3819, 0.4025, 0.5868, 0.461, 0.5375, 0.6708, 0.575, 0.7618, 1.5375, 1.8978, 2.2242, 2.1718, 2.4942, 2.5119, 3.0582, 4.081, 4.4579, 4.5695, 4.7966, 5.0066, 5.3043, 9.6586], [-13.4, 0.0, -2.2, -4.0, -6.0, -8.2, -9.9, -10.5, -7.5, -15.9, -6.6, -16.7, -12.4, -15.2, -10.8, -11.3, -12.7, -16.2, -18.3, -18.9, -16.6, -19.9, -29.7]), 'B': (True, 0, [0.0, 0.1072, 0.2155, 0.2095, 0.287, 0.2986, 0.3752, 0.5055, 0.3681, 0.3697, 0.57, 0.5283, 1.1021, 1.2756, 1.5474, 1.7842, 2.0169, 2.8294, 3.0219, 3.6187, 4.1067, 4.279, 4.7834], [0.0, -2.2, -4.0, -3.2, -9.8, -1.2, -3.4, -5.2, -7.6, -3.0, -8.9, -9.0, -4.8, -5.7, -7.5, -1.9, -7.6, -12.2, -9.8, -11.4, -14.9, -9.2, -11.3]), 'C': (True, 0, [0.0, 0.2099, 0.2219, 0.2329, 0.2176, 0.6366, 0.6448, 0.656, 0.6584, 0.7935, 0.8213, 0.9336, 1.2285, 1.3083, 2.1704, 2.7105, 4.2589, 4.6003, 5.4902, 5.6077, 6.3065, 6.6374, 7.0427, 8.6523], [-4.4, -1.2, -3.5, -5.2, -2.5, 0.0, -2.2, -3.9, -7.4, -7.1, -10.7, -11.1, -5.1, -6.8, -8.7, -13.2, -13.9, -13.9, -15.8, -17.1, -16, -15.7, -21.6, -22.8]), 'D': (True, 1, [0.0, 0.0, 0.035, 0.612, 1.363, 1.405, 1.804, 2.596, 1.775, 4.042, 7.937, 9.424, 9.708, 12.525], [-0.2, -13.5, -18.8, -21, -22.8, -17.9, -20.1, -21.9, -22.9, -27.8, -23.6, -24.8, -30.0, -27.7]), 'E': (True, 1, [0.0, 0.0, 0.5133, 0.544, 0.563, 0.544, 0.7112, 1.9092, 1.9293, 1.9589, 2.6426, 3.7136, 5.4524, 12.0034, 20.6519], [-0.03, -22.03, -15.8, -18.1, -19.8, -22.9, -22.4, -18.6, -20.8, -22.6, -22.3, -25.6, -20.2, -29.8, -29.2]), 'A10': (False, 0, [0.0, 4.0, 6.0, 8.0, 16.0, 18.0, 22.0, 24.0, 26.0, 30.0, 40.0, 44.0, 46.0, 48.0, 50.0, 96.0], [-16.1, 0.0, -4.0, -10.2, -18.6, -9.3, -13.7, -17.9, -13.5, -14.0, -15.4, -18.9, -21.0, -21.6, -19.3, -25.9]), 'A30': (False, 0, [0.0, 10.0, 15.0, 20.0, 25.0, 50.0, 65.0, 75.0, 105.0, 135.0, 150.0, 290.0], [-15.5, 0.0, -5.1, -5.1, -9.6, -8.2, -13.1, -11.5, -11.0, -16.2, -16.6, -26.2]), 'B100': (False, 0, [0.0, 10.0, 20.0, 30.0, 35.0, 45.0, 55.0, 120.0, 170.0, 245.0, 330.0, 480.0], [0.0, -2.2, -0.6, -0.6, -0.3, -1.2, -5.9, -2.2, -0.8, -6.3, -7.5, -7.1]), 'C300': (False, 0, [0.0, 65.0, 70.0, 190.0, 195.0, 200.0, 240.0, 325.0, 520.0, 1045.0, 1510.0, 2595.0], [-6.9, 0.0, -7.7, -2.5, -2.4, -9.9, -8.0, -6.6, -7.1, -13.0, -14.2, -16.0]), 'C60': (False, 0, [0.0, 15.0, 40.0, 50.0, 55.0, 75.0, 80.0, 130.0, 210.0, 300.0, 360.0, 520.0], [-7.8, -0.3, 0.0, -8.9, -14.5, -8.5, -10.2, -12.1, -13.9, -15.2, -16.9, -19.4]), 'D10': (False, 1, [0.0, 0.0, 6.0, 14.0, 18.0, 26.0, 40.0, 80.0, 94.0, 98.0, 126.0], [-0.2, -12.4, -21.1, -16.7, -18.3, -22.0, -27.9, -23.7, -24.9, -30.0, -27.7]), 'D30': (False, 1, [0.0, 0.0, 20.0, 40.0, 55.0, 80.0, 120.0, 240.0, 285.0, 290.0, 375.0], [-0.2, -12.4, -21.0, -16.7, -18.3, -21.9, -27.8, -23.6, -24.8, -30.0, -27.6])}
DS_GRID = np.geomspace(5e-9, 1500e-9, 32)
SAMPLE_T = 1.0 / (N * SUBCARRIER_SPACING)


TDL_SMEAR = (0.1, 0.1)  # relative delay smear for scalable profiles, absolute smear [samples]


def _tdl_rf():
    out = []
    k = np.arange(N)
    dk = k[:, None] - k[None, :]
    for name, (scale, los, dl, pw) in TDL_TABLES.items():
        p = 10 ** (np.array(pw) / 10.0)
        p = p / p.sum()
        dss = DS_GRID if scale else [1e-9]
        for ds in dss:
            tau = np.array(dl) * ds / SAMPLE_T  # in samples
            rel = TDL_SMEAR[0] if scale else 0.0
            R = np.zeros((N, N), complex)
            for i in range(len(tau)):
                lo = tau[i] / (1 + rel) - TDL_SMEAR[1]
                hi = tau[i] * (1 + rel) + TDL_SMEAR[1]
                ts = np.linspace(lo, hi, 9)
                R += p[i] * np.mean(np.exp(-2j * np.pi * dk[None] * ts[:, None, None] / N), 0)
            out.append(R)
    return np.array(out)


RF_GEN = np.einsum('kg,cg,lg->ckl', FD, PC, FD.conj())
RF_ALL = torch.tensor(np.concatenate([_tdl_rf(), RF_GEN], 0), dtype=CD)
NC = RF_ALL.shape[0]
RP_ALL = RF_ALL[:, KP][:, :, KP]
LP, UP = torch.linalg.eigh(RP_ALL)
LP = LP.clamp(min=0)
UPH = UP.conj().transpose(1, 2)
RFP_ALL = RF_ALL[:, :, KP]




ALPH_ABS = torch.tensor(np.geomspace(0.05, 5.0, 7), dtype=RD)
FE_OPT = {'topk': 8, 'prior': 2.0, 'temp': 1.0}


_sig = (LP > 1e-7 * LP[:, -1:])  # [C,36]
SIG_C, SIG_I = torch.nonzero(_sig, as_tuple=True)
SIG_L = LP[SIG_C, SIG_I]  # [S]
NSIG = _sig.sum(1).to(RD)  # [C]
_u = UPH[SIG_C, SIG_I, :]  # [S,36] rows
UST = torch.cat([torch.cat([_u.real, -_u.imag], 1), torch.cat([_u.imag, _u.real], 1)], 0).T.contiguous()  # [72,2S]
IND = torch.zeros(len(SIG_C), NC, dtype=RD)
IND[torch.arange(len(SIG_C)), SIG_C] = 1.0
NS = len(SIG_C)
GF = (RFP_ALL @ UP).contiguous()  # [C,72,36]
GF2 = GF.real ** 2 + GF.imag ** 2
RFD = torch.diagonal(RF_ALL, dim1=-2, dim2=-1).real.contiguous()  # [C,72]


def freq_est(Yp, no):
    """Yp [B,M,36] LS pilots on M pilot symbols. Evidence-based PDP selection (BMA) + LMMSE."""
    B, M, _ = Yp.shape
    Yc = torch.cat([Yp.real, Yp.imag], -1).reshape(B * M, 2 * NP)
    Pj = (Yc @ UST).pow(2).reshape(B, M, 2, NS).sum((1, 2))  # [B,S]
    ytot = a2(Yp).sum((1, 2))  # [B]
    k = FE_OPT['prior']
    nb = no[:, None]
    base = -(M * (NP - NSIG)[None] * torch.log(nb))  # [B,C] (plus residual term below)

    def evid(al_s, al_c):
        # al_s [B,A,S] alpha per sig-eig, al_c [B,A,C]
        d = al_s * SIG_L + nb[:, :, None]
        t = M * torch.log(d) + Pj[:, None, :] / d - Pj[:, None, :] / nb[:, :, None]
        return -(t @ IND) + base[:, None] - (ytot[:, None, None] / nb[:, :, None]) + (k - 1) * torch.log(al_c) - k * al_c

    A = ALPH_ABS.shape[0]
    ev = evid(ALPH_ABS[None, :, None].expand(B, A, NS), ALPH_ABS[None, :, None].expand(B, A, NC))  # [B,A,C]
    evb, ib = ev.max(1)  # [B,C]
    al = ALPH_ABS[ib]  # [B,C]
    for _ in range(2):
        als = al[:, SIG_C]  # [B,S]
        d = als * SIG_L + nb
        g = -((M * SIG_L / d - Pj * SIG_L / d ** 2) @ IND) + (k - 1) / al - k
        h = (((M * SIG_L ** 2) / d ** 2 - 2 * Pj * SIG_L ** 2 / d ** 3) @ IND) - (k - 1) / al ** 2
        step = torch.where(h < 0, -g / h, torch.sign(g) * 0.2 * al)
        an = (al + step.clamp(min=-0.5 * al, max=0.5 * al)).clamp(0.02, 8.0)
        en = evid(an[:, SIG_C][:, None], an[:, None])[:, 0]
        better = en > evb
        al = torch.where(better, an, al)
        evb = torch.where(better, en, evb)
    a = al
    ev = evb
    K = FE_OPT['topk']
    top = ev.topk(K, dim=1)
    wts = torch.softmax(top.values / FE_OPT['temp'], 1)  # [B,K]
    c = top.indices
    alpha = torch.gather(a, 1, top.indices)  # [B,K]
    lk = LP[c]  # [B,K,36]
    inv = 1.0 / (alpha[..., None] * lk + no[:, None, None])  # [B,K,36]
    Z = UPH[c] @ Yp.transpose(1, 2)[:, None]  # [B,K,36,M]
    Hk = (alpha[..., None, None] * (GF[c] @ (inv[..., None] * Z))).transpose(2, 3)  # [B,K,M,72]
    pk = (alpha[..., None] * RFD[c] - alpha[..., None] ** 2 * (GF2[c] @ inv[..., None])[..., 0]).clamp(min=0)
    Hf = (wts[:, :, None, None] * Hk).sum(1)
    post = (wts[:, :, None] * pk).sum(1) + (wts[:, :, None, None] * a2(Hk - Hf[:, None])).sum(1).mean(1)
    Rmix = (wts[..., None, None] * alpha[..., None, None] * RF_ALL[c]).sum(1)
    return Hf, post, Rmix, (wts * alpha).sum(1)


# ---------------- time-domain covariance candidates (Doppler) ----------------------
FD_GRID = np.array([0, 80, 160, 260, 380, 520, 700, 900, 1150, 1450, 1800, 2300.0])
LOS_A = [0.8]
DD = torch.arange(-(T - 1), T, dtype=RD)
J0TAB = torch.tensor(j0(2 * np.pi * FD_GRID[:, None] * np.arange(-(T - 1), T)[None, :] * TS), dtype=RD)
TIDX = torch.tensor(TT[:, None] - TT[None, :] + T - 1)
PTIDX = torch.tensor(TT[:, None] - np.array(PS)[None, :] + T - 1)
PPIDX = torch.tensor(np.array(PS)[:, None] - np.array(PS)[None, :] + T - 1)


FD_ALLQ = torch.tensor(np.concatenate([FD_GRID] * (1 + len(LOS_A))), dtype=RD)
FD_PEN = torch.tensor(np.concatenate([-np.log1p(FD_GRID / 200.0)] * (1 + len(LOS_A))), dtype=RD)


def build_cands(nu0):
    B = nu0.shape[0]
    rs = [J0TAB[None].expand(B, -1, -1).to(CD)]
    ph = torch.exp(1j * 2 * np.pi * nu0[:, None] * DD[None, :] * TS)
    for a in LOS_A:
        rs.append(a * ph[:, None, :] + (1 - a) * J0TAB[None].to(CD))
    return torch.cat(rs, 1)


def loglik16(y, H, s2):
    # uniform 16QAM marginal log-likelihood (up to constants), closed form per I/Q
    yh = y.conj() * H
    h2 = a2(H)
    u1 = 2 * yh.real / (s2 * SQ10)
    u2 = -2 * yh.imag / (s2 * SQ10)
    v = h2 / (10 * s2)
    lr = _lse2(-v + _lc(u1), -9 * v + _lc(3 * u1))
    li = _lse2(-v + _lc(u2), -9 * v + _lc(3 * u2))
    return lr + li - a2(y) / s2 - torch.log(s2)


def soft16(s, var):
    def pam(r, v):
        d = -(r[..., None] - LV) ** 2 / v[..., None]
        d = d - d.max(-1, keepdim=True).values
        p = torch.exp(d)
        p = p / p.sum(-1, keepdim=True)
        m = (p * LV).sum(-1)
        e2 = (p * LV ** 2).sum(-1)
        return m, e2 - m ** 2
    mr, vr = pam(s.real, var)
    mi, vi = pam(s.imag, var)
    return torch.complex(mr, mi), vr + vi


def lmmse2d_b(wz, w, Ut, lt, Uf, lf, iters=20, D0=None):
    S = torch.sqrt((lt[:, :, None] * lf[:, None, :]).clamp(min=0))
    UtH = Ut.conj().transpose(1, 2)
    Ufc = Uf.conj()

    def Phi(C):
        return Ut @ C @ Uf.transpose(1, 2)

    def PhiH(X):
        return UtH @ X @ Ufc

    def Aop(D):
        return D + S * PhiH(w * Phi(S * D))

    wt = w.mean(2)
    K = UtH @ (wt[:, :, None] * Ut)
    Sj = S.transpose(1, 2)
    M = torch.eye(T, dtype=CD) + Sj[:, :, :, None] * K[:, None] * Sj[:, :, None, :]
    Mi = hpd_inv32(M) if FAST32 else hpd_inv(M)

    def prec(R):
        return (Mi * R.transpose(1, 2)[:, :, None, :]).sum(-1).transpose(1, 2)

    b = S * PhiH(wz)
    D = prec(b) if D0 is None else D0
    r = b - Aop(D)
    zz = prec(r)
    p = zz
    rz = (r.conj() * zz).real.sum((1, 2))
    for it in range(iters):
        Ap = Aop(p)
        al = rz / (p.conj() * Ap).real.sum((1, 2)).clamp(min=1e-30)
        D = D + al[:, None, None] * p
        r = r - al[:, None, None] * Ap
        zz = prec(r)
        rzn = (r.conj() * zz).real.sum((1, 2))
        p = zz + (rzn / rz.clamp(min=1e-30))[:, None, None] * p
        rz = rzn
    H = Phi(S * D)
    Pj = Sj[:, :, :, None] * Mi * Sj[:, :, None, :]
    dg = ((Ut[:, None] @ Pj) * Ut.conj()[:, None]).sum(-1).real
    err = dg.transpose(1, 2) @ a2(Uf).transpose(1, 2)
    return H, err.clamp(min=1e-12), dg.sum(-1), D


_E3 = torch.tensor([0.2, 1.0, 1.8], dtype=RD)
_GE = torch.round((QE - 0.2) / 0.8).long()  # group index 0..2
GT = torch.zeros(3, 16, dtype=RD)
GT[_GE, torch.arange(16)] = 1.0
C4 = torch.stack([2 * QR, -2 * QI, -QE, -torch.ones(16, dtype=RD)], 0)  # [4,16]
F9 = torch.cat([GT.T, GT.T * QR[:, None], GT.T * QI[:, None]], 1)  # [16,9]


def ep_msgs(y, m, P, no):
    # tilted moments of h for y = h x + n, h ~ CN(m,P), x uniform 16QAM (grouped by |x|^2)
    nb = no[:, None, None, None]
    s3 = _E3 * P[..., None] + nb  # [B,T,N,3]
    is3 = 1.0 / s3
    g = y.conj() * m
    y2 = a2(y)
    m2 = a2(m)
    feat = torch.stack([g.real, g.imag, m2, y2], -1)  # [B,T,N,4]
    lp = (feat @ C4) * (is3 @ GT) - (torch.log(s3) @ GT)
    lp = lp - lp.max(-1, keepdim=True).values
    pi = torch.exp(lp)
    pi = pi / pi.sum(-1, keepdim=True)
    S = pi @ F9  # [B,T,N,9]
    S0, SR, SI = S[..., 0:3], S[..., 3:6], S[..., 6:9]
    A = torch.complex((SR * is3).sum(-1), -(SI * is3).sum(-1))
    Bq = (_E3 * S0 * is3).sum(-1)
    muc = y * A - m * Bq
    mu = m + P * muc
    pq = y * m.conj()
    is3s = is3 * is3
    c2 = (is3s * (_E3 * y2[..., None] * S0 - 2 * _E3 * (pq.real[..., None] * SR + pq.imag[..., None] * SI)
                  + _E3 ** 2 * m2[..., None] * S0)).sum(-1)
    Em2 = m2 + 2 * P * (m.conj() * muc).real + P ** 2 * c2
    Px = P * no[:, None, None] * (S0 * is3).sum(-1)
    V = torch.maximum(Px + Em2 - a2(mu), Px)
    return mu, V.clamp(min=1e-12)


def _eig_f(Rf, ftr):
    lf, Uf = torch.linalg.eigh(Rf.to(torch.complex128) if not FAST32 else Rf)
    lf = lf.to(RD); Uf = Uf.to(CD)
    lf = lf.clamp(min=0)
    kf = int((lf > ftr * lf[:, -1:]).sum(1).max().item())
    kf = max(kf, 1)
    return lf[:, -kf:].contiguous(), Uf[:, :, -kf:].contiguous()


DSYM = [t for t in range(T) if t not in PS]


SEQ_ORDER = [3, 10, 1, 12, 4, 9, 0, 13, 5, 8, 6, 7]


def seq_init(y, no, Hf, post, pw, rb, Rf0, Uf, lf):
    """Sequential (symbol-by-symbol, outward from the pilots) data-aided initialisation.
    Returns EP messages (w, wz) for all data REs."""
    B = y.shape[0]
    ests = {PS[0]: Hf[:, 0], PS[1]: Hf[:, 1]}
    e0 = (post.mean(-1) / pw).clamp(min=1e-6)
    errs = {PS[0]: e0, PS[1]: e0}
    Rdiag = torch.diagonal(Rf0, dim1=-2, dim2=-1).real.clamp(min=1e-12)  # [B,N]
    UfH = Uf.conj().transpose(1, 2)
    wm = torch.zeros(B, T, N, dtype=RD)
    wzm = torch.zeros(B, T, N, dtype=CD)
    for t in SEQ_ORDER:
        S = sorted(ests.keys())
        Si = torch.tensor(S)
        rS = rb[:, t - Si + T - 1]  # [B,|S|]  E[h_t h_s^*]
        RSS = rb[:, (Si[:, None] - Si[None, :]) + T - 1] + torch.diag_embed(torch.stack([errs[s] for s in S], 1)).to(CD)
        g = (rS[:, None, :] @ hpd_inv32(RSS))[:, 0]  # [B,|S|]
        eps = (1.0 - (g * rS.conj()).sum(-1).real).clamp(min=1e-5)  # [B]
        Hs = torch.stack([ests[s] for s in S], 1)  # [B,|S|,N]
        Hpred = (g[:, :, None] * Hs).sum(1)  # [B,N]
        P = eps[:, None] * Rdiag
        mu, V = ep_msgs(y[:, t:t + 1], Hpred[:, None], P[:, None], no)
        mu, V = mu[:, 0], V[:, 0]
        lam = 1.0 / V - 1.0 / P
        ok = (lam > 0) & torch.isfinite(lam) & torch.isfinite(mu)
        lam = torch.where(ok, lam, torch.zeros_like(lam))
        wz = torch.where(ok, mu / V - Hpred / P, torch.zeros_like(mu))
        A = torch.diag_embed((1.0 / (eps[:, None] * lf).clamp(min=1e-30)).to(CD)) + UfH @ (lam[:, :, None] * Uf)
        Ai = hpd_inv32(A)
        c = (Ai @ (UfH @ (wz - lam * Hpred)[:, :, None]))[..., 0]
        ests[t] = Hpred + (Uf @ c[:, :, None])[..., 0]
        errs[t] = (((Uf @ Ai) * Uf.conj()).sum(-1).real.mean(-1) / pw).clamp(min=1e-6)
        wm[:, t] = lam
        wzm[:, t] = wz
    return wm, wzm


def _da_group(y, no, H, err, Hf, post, pw, rb, Rf0, Uf, lf, Ut, lt, use_seq, sched, relax, cg_it, warm,
              deadline=None):
    """Data-aided refinement for a sub-batch that shares one iteration schedule.
    Every iteration yields a valid estimate, so the loop stops early if the time budget runs out."""
    B = y.shape[0]
    pm = PMASK[None].expand(B, -1, -1)
    w_old = torch.zeros(B, T, N, dtype=RD)
    wz_old = torch.zeros(B, T, N, dtype=CD)
    wp = (1.0 / no)[:, None, None].expand(B, T, N)
    wzp = y * PGRID.conj() / no[:, None, None]
    Dw = None
    t0 = time.perf_counter()
    if deadline is not None and t0 > deadline - 0.03:
        return H
    if deadline is not None and t0 > deadline - 0.08:
        use_seq = False
    if use_seq:
        w_old, wz_old = seq_init(y, no, Hf, post, pw, rb, Rf0, Uf, lf)
        H, err, _, Dw = lmmse2d_b(torch.where(pm, wzp, wz_old), torch.where(pm, wp, w_old), Ut, lt, Uf, lf,
                                  iters=cg_it)
    t_it = 0.0
    for it, md in enumerate(sched):
        t1 = time.perf_counter()
        if deadline is not None and t1 + 1.5 * t_it > deadline:
            break
        if md == 'ep':
            pc = (1.0 / err - w_old).clamp(min=1e-3 / pw[:, None, None])
            Pc = 1.0 / pc
            mc = Pc * (H / err - wz_old)
            mu, V = ep_msgs(y, mc, Pc, no)
            lam = (1.0 / V - pc)
            ok = (lam > 0) & ~pm & torch.isfinite(lam) & torch.isfinite(mu)
            w_old = torch.where(ok, lam, torch.zeros_like(lam))
            wz_old = torch.where(ok, mu / V - mc * pc, torch.zeros_like(mu))
        else:
            h2 = a2(H)
            s = y / torch.where(h2 > 1e-12, H, torch.full_like(H, 1e-6))
            var = (no[:, None, None] + err) / h2.clamp(min=1e-12)
            xm, xv = soft16(s, var)
            w_old = (a2(xm) + xv) / no[:, None, None]
            wz_old = y * xm.conj() / no[:, None, None]
        w = torch.where(pm, wp, w_old)
        wz = torch.where(pm, wzp, wz_old)
        Hn, err, _, Dw = lmmse2d_b(wz, w, Ut, lt, Uf, lf, iters=cg_it, D0=Dw if warm else None)
        if md == 'em' and relax != 1.0 and it > 0:
            Hn = H + relax * (Hn - H)  # over-relaxed EM step
        H = Hn
        t_it = time.perf_counter() - t1
    return H


def _estimate(y, no, n_da=1, cg_it=4, thr=8.0, ssub=4, ftr=1e-6,
              hi_sched=('ep', 'ep', 'em', 'em'), lo_sched=('em', 'em', 'em', 'em'), warm=True,
              ttemp=4.0, tprior=7.0, relax=2.0, fdthr=400.0, thr_hi=14.0, seq=True, seq_fd=400.0,
              deadline=None):
    y = y.to(CD)
    no = no.to(RD)
    B = y.shape[0]
    # ---- stage 1: pilot-based frequency estimate with evidence-weighted PDP model averaging
    Yp = y[:, PS][:, :, KP] / PIL
    Hf, post, Rmix, alpha = freq_est(Yp, no)
    pw = alpha
    # ---- stage 2: Doppler model averaging scored by the data likelihood of pilot-only predictions
    num = (Yp[:, 1] * Yp[:, 0].conj()).mean(-1)
    nu0 = torch.angle(num) / (2 * np.pi * 9 * TS)
    r = build_cands(nu0)
    e = (post.mean(-1) / pw).clamp(min=1e-5)
    Rtp = r[:, :, PTIDX]
    Rpp = r[:, :, PPIDX] + e[:, None, None, None] * torch.eye(2, dtype=CD)
    W = Rtp @ hpd_inv(Rpp)
    evt = (1.0 - (W * Rtp.conj()).sum(-1).real).clamp(min=0) * pw[:, None, None]
    fe = post.mean(-1)[:, None, None] * a2(W).sum(-1)
    s2 = no[:, None, None] + evt + fe
    Hs = W[:, :, DSYM] @ Hf[:, None, :, 1::ssub]
    ll = loglik16(y[:, None, DSYM, 1::ssub], Hs, s2[:, :, DSYM, None].expand(-1, -1, -1, Hs.shape[-1]))
    sc_t = ll.sum((-1, -2)) * ssub + tprior * FD_PEN[None]
    wq = torch.softmax(sc_t / ttemp, 1)  # [B,Q]
    Wm = (wq[:, :, None, None] * W).sum(1)
    H = Wm @ Hf
    Hq = W @ Hf[:, None]
    spread = (wq[:, :, None] * a2(Hq - H[:, None]).mean(-1)).sum(1)
    err = ((wq[:, :, None] * (evt + fe)).sum(1) + spread)[:, :, None].expand(B, T, N).clone()
    rb = (wq[:, :, None] * r).sum(1)
    fdm = (wq * FD_ALLQ[None]).sum(1)
    if n_da == 0:
        return H
    # ---- stage 3: data-aided 2-D LMMSE (Kronecker prior Rt x Rf) with EP / EM soft decisions
    eyeN = torch.eye(N, dtype=CD)
    Rf0 = Rmix + (1e-6 * pw)[:, None, None] * eyeN
    lf, Uf = _eig_f(Rf0, ftr)
    Rt0 = rb[:, TIDX] + 1e-6 * torch.eye(T, dtype=CD)
    lt, Ut = torch.linalg.eigh(Rt0.to(torch.complex128))
    lt = lt.to(RD).clamp(min=0)
    Ut = Ut.to(CD)
    snr_db = 10 * torch.log10(1.0 / no)
    hi0 = (snr_db >= thr) & ((fdm >= fdthr) | (snr_db >= thr_hi))
    useq = (fdm >= seq_fd) & torch.tensor(bool(seq))
    Hout = H.clone()
    for gh, gs in ((True, True), (True, False), (False, True), (False, False)):
        m = (hi0 == gh) & (useq == gs)
        if not bool(m.any()):
            continue
        idx = torch.nonzero(m).flatten()
        Hout[idx] = _da_group(y[idx], no[idx], H[idx], err[idx], Hf[idx], post[idx], pw[idx], rb[idx],
                              Rf0[idx], Uf[idx], lf[idx], Ut[idx], lt[idx], gs,
                              list(hi_sched) if gh else list(lo_sched), 1.0 if gh else relax, cg_it, warm,
                              deadline)
    return Hout


def _ls_linear(y):
    Hp = y[:, PS][:, :, KP] / PIL  # [B,2,36]
    k = np.arange(N)
    i0 = np.clip(k // 2, 0, NP - 2)
    wk = torch.tensor((k - KP[i0]) / 2.0, dtype=RD)
    Hf = Hp[:, :, i0] * (1 - wk) + Hp[:, :, i0 + 1] * wk
    wt = torch.tensor((TT - PS[0]) / float(PS[1] - PS[0]), dtype=RD)[None, :, None]
    return Hf[:, :1] * (1 - wt) + Hf[:, 1:] * wt


TIME_BUDGET = 0.36  # [s] per call; data-aided iterations stop early if exceeded (limit is 0.5 s)


def estimate_channel(y, no, **kw):
    t_start = time.perf_counter()
    nt = torch.get_num_threads()
    torch.set_num_threads(1)
    try:
        with torch.no_grad():
            y = torch.as_tensor(y).to(CD)
            no = torch.as_tensor(no).to(RD).reshape(-1)
            if no.numel() == 1 and y.shape[0] > 1:
                no = no.expand(y.shape[0]).clone()
            no = no.clamp(min=1e-8)
            kw.setdefault('deadline', t_start + TIME_BUDGET)
            try:
                H = _estimate(y, no, **kw)
            except Exception:
                H = None
            if H is None or not bool(torch.isfinite(torch.view_as_real(H)).all()):
                try:
                    H0 = _estimate(y, no, n_da=0)
                except Exception:
                    H0 = _ls_linear(y)
                if not bool(torch.isfinite(torch.view_as_real(H0)).all()):
                    H0 = torch.where(torch.isfinite(H0.real) & torch.isfinite(H0.imag), H0, _ls_linear(y))
                if H is None:
                    H = H0
                else:
                    okm = (torch.isfinite(H.real) & torch.isfinite(H.imag)).all(-1).all(-1)
                    H = torch.where(okm[:, None, None], H, H0)
            return H.to(torch.complex64)
    finally:
        torch.set_num_threads(nt)
