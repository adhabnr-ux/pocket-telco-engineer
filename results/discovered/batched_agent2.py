"""
Data-aided, statistics-adaptive OFDM channel estimator (multi-group Kronecker model).

Pipeline (per call, all examples share one scenario and SNR):
  1. LS on pilots; M-SBL on the 64 pilot snapshots -> power-delay profile on a fine delay grid
     -> frequency covariance Rf (Toeplitz, PSD by construction).
  2. Time covariance Rt from the pilot-symbol cross-correlation (Jakes J0 model + LOS phase).
  3. Optional split into a dominant-path group and a residual group (non-separable LOS channels):
     R = sum_g Rt_g (x) Rf_g.
  4. Data-aided EM: channel posterior mean in a truncated Kronecker eigenbasis, soft 16-QAM
     demapping, symbols admitted ring-by-ring outwards from the pilot symbols, over-relaxation,
     and EM / SBL re-estimation of the time and frequency covariances from the data-aided channel.
"""
import numpy as np, torch
from scipy.special import j0
from link_config import PILOT_MASK, PILOT_SYMBOLS, FFT_SIZE, NUM_OFDM_SYMBOLS, PILOT_OFDM_SYMBOLS

C128 = torch.complex128
C64 = torch.complex64
NS, NF = NUM_OFDM_SYMBOLS, FFT_SIZE
PS = list(PILOT_OFDM_SYMBOLS)
KP = np.where(PILOT_MASK[PS[0]])[0]
NP = len(KP)
XP = torch.tensor(np.stack([PILOT_SYMBOLS[s, KP] for s in PS]), dtype=C128)   # [2,36]
TAU = np.arange(-3, 30, 0.25)                                                  # delay grid [samples]
A_all = torch.tensor(np.exp(-2j*np.pi*np.outer(np.arange(NF), TAU)/NF), dtype=C128)
A_p = A_all[KP].contiguous()
PM_T = torch.tensor(PILOT_MASK)
PX_T = torch.tensor(PILOT_SYMBOLS.astype(np.complex128))
_x = np.linspace(0, 3.8317, 3000); _j = j0(_x)
DL = np.arange(NS)[:, None] - np.arange(NS)[None]
NU = torch.arange(-0.6, 0.6001, 0.01, dtype=torch.float64)                    # Doppler grid [rad/symbol]
AT = torch.exp(1j*torch.arange(NS, dtype=torch.float64)[:, None]*NU[None]).to(C128)
DIST = torch.tensor([min(abs(l - p) for p in PS) for l in range(NS)]).double()
EDGE = torch.tensor([l < min(PS) or l > max(PS) for l in range(NS)])

# ---- parameters ----
SBL_IT = 40          # M-SBL iterations (PDP)
SBLT_IT = 50         # SBL iterations (Doppler spectrum)
RMAX = 160           # max basis size
TTHR = 1e-5          # time eigen threshold (relative)
FTHR = 1e-5          # freq eigen threshold (relative)
FN = 0.6             # freq eigen threshold relative to noise variance
PH_THR = 0.4         # use pilot-correlation phase (LOS Doppler shift) only if |c| above this
E_INFL = 3.0         # inflation of channel posterior variance in the demapper
OVR = 1.8            # EM over-relaxation
FAST_THR = 0.8       # pilot correlation below which the channel is treated as fast-fading
STEP_FAST, STEP_SLOW = 0.5, 1.0   # ring growth per EM iteration
EDGE_PEN = 0.5       # extrapolated edge symbols admitted later
RT_N0MAX = 10**(0.25)            # data-SBL Doppler re-estimation only above -2.5 dB
MULTI = True
LOS_W = 0.5          # half-width [samples] of the dominant-path delay window
LOS_FMIN = 0.5       # min power fraction of the dominant window to enable the split
LOS_FMAX = 1.01      # (no upper limit)
LAST = [None]
LASTU = [None]
PH_MAXANG = 2.5      # |phase| above this is read as a negative real correlation
SLIP = True          # restart EM for examples stuck in a wrong local optimum
SLIP_N0MAX = 10**(-0.75)
SLIP_MAX = 6
SLIP_Z = -3.0
SLIP_MARGIN = 0.0
PAM4 = torch.tensor([-3., -1., 1., 3.]) / np.sqrt(10)
N0 = [1.0]


def _sbl(S, n0, A, iters):
    """M-SBL (MacKay updates) on sample covariance S for dictionary A, known noise n0."""
    n = A.shape[0]; I = torch.eye(n, dtype=C128)
    G = A.shape[1]
    g = torch.full((G,), max(float(torch.diagonal(S).real.mean()) - n0, 1e-3) / G * 4, dtype=torch.float64)
    for _ in range(iters):
        Si = torch.linalg.inv((A*g) @ A.conj().T + n0*I)
        SiA = Si @ A
        q = (SiA.conj() * (S @ SiA)).sum(0).real
        d = (A.conj() * SiA).sum(0).real
        g = (g*q/d).clamp(min=1e-12)
    return g


def _pilot_corr(hp, n0):
    P0 = max(float((hp[:, 0].abs()**2).mean()) - n0, 1e-6)
    P1 = max(float((hp[:, 1].abs()**2).mean()) - n0, 1e-6)
    return complex((hp[:, 1]*hp[:, 0].conj()).mean()) / np.sqrt(P0*P1)


def _phase(c):
    """LOS Doppler-shift phase of the lag-9 pilot correlation; a phase near +-pi is read as a
    negative real (NLOS, beyond the first J0 zero) correlation instead."""
    a = float(np.angle(c))
    return a if (abs(c) > PH_THR and abs(a) < PH_MAXANG) else 0.0


def _rt_from_c(c):
    ph = _phase(c)
    val = float(np.clip(np.real(c*np.exp(-1j*ph)), -0.4027, 1.0))
    x9 = np.interp(val, _j[::-1], _x[::-1])
    return torch.tensor(j0(x9*DL/9.0) * np.exp(1j*ph*DL/9.0), dtype=C128)


def _eigh(R):
    R = 0.5*(R + R.conj().T)
    try:
        return torch.linalg.eigh(R)
    except Exception:
        return torch.linalg.eigh(R + 1e-9*float(torch.diagonal(R).real.abs().mean() + 1e-12)*torch.eye(R.shape[0], dtype=R.dtype))


def _basis(Rt, Rf, rmax):
    lt, Ut = _eigh(Rt); lf, Uf = _eigh(Rf)
    lt = lt.flip(0).clamp(min=0); Ut = Ut.flip(1)
    lf = lf.flip(0).clamp(min=0); Uf = Uf.flip(1)
    rt = max(1, int((lt > TTHR*float(lt.sum())).sum()))
    rf = max(1, int((lf > max(FTHR*float(lf.sum()), FN*N0[0])).sum()))
    rf = min(rf, max(1, rmax // rt))
    return [Ut[:, :rt].contiguous(), lt[:rt].contiguous(), Uf[:, :rf].contiguous(), lf[:rf].contiguous()]


def _solve(groups, w, b, need_e=True):
    """Posterior mean (and variance) of h = sum_g (Ut_g (x) Uf_g) c_g, c_g ~ CN(0, diag(lt (x) lf)),
    observations with per-RE precision w and matched-filter data b."""
    B = w.shape[0]
    sizes = [len(g[1])*len(g[3]) for g in groups]
    offs = np.concatenate([[0], np.cumsum(sizes)]); r = int(offs[-1])
    sq = torch.cat([torch.outer(g[1], g[3]).reshape(-1) for g in groups]).sqrt().to(C128)
    wf = w.to(torch.float32)[..., None]
    U6 = [(g[0].to(C64), g[2].to(C64)) for g in groups]
    G = torch.empty(B, r, r, dtype=C128)
    for a in range(len(groups)):
        Ut_a, Uf_a = U6[a]
        Wu = (wf * Uf_a.conj()).transpose(-1, -2)                    # [B,14,rfa,72]
        for c in range(a, len(groups)):
            Ut_c, Uf_c = U6[c]
            M = torch.matmul(Wu, Uf_c)                                # [B,14,rfa,rfc]
            TT = Ut_a.conj()[:, :, None] * Ut_c[:, None, :]           # [14,rta,rtc]
            blk = torch.einsum('lac,zljm->zajcm', TT, M).reshape(B, sizes[a], sizes[c]).to(C128)
            G[:, offs[a]:offs[a+1], offs[c]:offs[c+1]] = blk
            if c != a:
                G[:, offs[c]:offs[c+1], offs[a]:offs[a+1]] = blk.conj().transpose(-1, -2)
    G = G * sq[None, :, None] * sq[None, None, :] + torch.eye(r, dtype=C128)[None]
    rhs = torch.cat([torch.einsum('blj,la->baj', torch.matmul(b, g[2].conj()), g[0].conj()).reshape(B, -1)
                     for g in groups], 1) * sq[None]
    L = torch.linalg.cholesky(G)
    u = torch.cholesky_solve(rhs[..., None], L)[..., 0]
    cfull = u * sq[None]
    LASTU[0] = (u.abs()**2).sum(1).real
    h = 0
    cs = []
    for a, g in enumerate(groups):
        ca = cfull[:, offs[a]:offs[a+1]].reshape(B, len(g[1]), len(g[3]))
        cs.append(ca)
        h = h + torch.einsum('la,baj,kj->blk', g[0], ca, g[2])
    if not need_e:
        return h, None
    Gi = torch.cholesky_inverse(L)
    Gi = (Gi * sq[None, :, None] * sq[None, None, :]).to(C64)
    e = torch.zeros(B, NS, NF, dtype=torch.float32)
    dblk = []
    for a in range(len(groups)):
        Ut_a, Uf_a = U6[a]; rta, rfa = len(groups[a][1]), len(groups[a][3])
        for c in range(a, len(groups)):
            Ut_c, Uf_c = U6[c]; rtc, rfc = len(groups[c][1]), len(groups[c][3])
            Gac = Gi[:, offs[a]:offs[a+1], offs[c]:offs[c+1]].reshape(B, rta, rfa, rtc, rfc)
            if c == a:
                dblk.append(Gac)
            S = torch.einsum('la,lc,zajcm->zljm', Ut_a, Ut_c.conj(), Gac)    # [B,14,rfa,rfc]
            t = (torch.matmul(S, Uf_c.conj().T[None, None]) * Uf_a.T[None, None]).sum(2).real
            e = e + (t if c == a else 2*t)
    LAST[0] = (cs, dblk)
    return h, e.clamp(min=0).double()


_S = 1.0/np.sqrt(10)
def _pam_post(z, p):
    a = (2*_S)*z*p; q = (_S*_S)*p
    l0 = -3*a - 9*q; l1 = -a - q; l2 = a - q; l3 = 3*a - 9*q
    m = torch.maximum(torch.maximum(l0, l1), torch.maximum(l2, l3))
    e0 = torch.exp(l0 - m); e1 = torch.exp(l1 - m); e2 = torch.exp(l2 - m); e3 = torch.exp(l3 - m)
    S = e0 + e1 + e2 + e3
    xm = _S*(3*(e3 - e0) + e2 - e1)/S
    m2 = (_S*_S)*(9*(e0 + e3) + e1 + e2)/S
    return xm, m2 - xm*xm


def _demap(y, h, e, n0):
    """Soft 16-QAM demapping (separable 4-PAM per dimension). Returns posterior mean and variance."""
    y = y.to(C64); h = h.to(C64)
    s2 = (n0 + e).float()
    g2 = (h.real**2 + h.imag**2).clamp(min=1e-12)
    z = y * h.conj() / g2
    prec = g2 / s2
    xr, vr = _pam_post(z.real, prec)
    xi, vi = _pam_post(z.imag, prec)
    return torch.complex(xr, xi).to(C128), (vr + vi).clamp(min=0).double()


def _sbl_t(S, ne):
    g = _sbl(S, ne, AT, SBLT_IT)
    Rt = (AT*g) @ AT.conj().T
    return Rt / g.sum()


def _rt_data(q, ne, F=None):
    """Doppler spectrum (time covariance) via SBL over data-aided channel snapshots q [B,14,72]."""
    if F is not None:
        q = torch.matmul(q, F.T)
        ne = ne * float((F.abs()**2).sum()) / NF
    Q = q.permute(0, 2, 1).reshape(-1, NS)
    S = Q.T @ Q.conj() / Q.shape[0]
    return _sbl_t(S, ne)


def _wiener(Rf_g, Rf, nv):
    return Rf_g @ torch.linalg.inv(Rf + nv*torch.eye(Rf.shape[0], dtype=C128))


def _estimate(y, no):
    yt = y.to(C128)
    B = yt.shape[0]
    n0 = max(float(no.double().mean()), 1e-6)
    N0[0] = n0
    hp = yt[:, PS][:, :, KP] / XP[None]                                  # [B,2,36]
    Yp = hp.reshape(-1, NP)
    g = _sbl(Yp.T @ Yp.conj() / Yp.shape[0], n0, A_p, SBL_IT)
    c_all = _pilot_corr(hp, n0)
    # ---- groups ----
    gl = [g]
    if MULTI:
        ip = int(torch.argmax(g))
        win = torch.tensor(np.abs(TAU - TAU[ip]) <= LOS_W)
        frac = float(g[win].sum() / g.sum())
        if LOS_FMIN <= frac < LOS_FMAX:
            gl = [g*win, g*(~win)]
    Rfs = [(A_all*gg) @ A_all.conj().T for gg in gl]
    Rf = sum(Rfs)
    if len(gl) == 1:
        Rts = [_rt_from_c(c_all)]
    else:
        Rfp = Rf[KP][:, KP]
        Rts = []
        for Rg in Rfs:
            Fp = _wiener(Rg[KP][:, KP], Rfp, n0)
            hpg = torch.matmul(hp, Fp.T)
            nf = n0 * float((Fp.abs()**2).sum()) / NP
            Rts.append(_rt_from_c(_pilot_corr(hpg, nf)))
    rmx = [RMAX] if len(gl) == 1 else [RMAX // 4] + [RMAX]*(len(gl) - 1)
    groups = [_basis(Rts[i], Rfs[i], rmx[i]) for i in range(len(gl))]
    # ---- schedule ----
    phc = _phase(c_all)
    fast = float(np.real(c_all*np.exp(-1j*phc))) < FAST_THR
    step = STEP_FAST if fast else STEP_SLOW
    itf = int(np.ceil(float((DIST + EDGE_PEN*EDGE).max())/step - 1 - 1e-9))
    n_iter = itf + 3
    rti = (itf, itf + 1)            # data-SBL Doppler re-estimation
    rfe = (itf + 1,)                # EM update of Rf
    rte = (itf + 2,)                # EM update of Rt
    # ---- data-aided EM ----
    pm = PM_T[None].expand(B, -1, -1)
    px = PX_T[None].expand(B, -1, -1)
    w = torch.where(pm, 1.0/n0, 0.0).double()
    bb = torch.where(pm, px.conj(), torch.zeros((), dtype=C128)) * yt / n0
    hprev = None
    for itr in range(n_iter + 1):
        h, e = _solve(groups, w, bb, need_e=itr < n_iter)
        if e is not None:
            e_last = e
        if itr == n_iter:
            break
        if itr > 0:
            h = hprev + OVR*(h - hprev)
        hprev = h
        xd, v = _demap(yt, h, e*E_INFL, n0)
        xm = torch.where(pm, px, xd)
        w = torch.where(pm, torch.ones_like(v), xd.abs()**2 + v) / n0
        bb = xm.conj() * yt / n0
        act = ((DIST + EDGE_PEN*EDGE) <= (itr + 1)*step)[None, :, None]
        w = torch.where(act, w, torch.zeros_like(w))
        bb = torch.where(act, bb, torch.zeros_like(bb))
        if itr in rfe:                        # EM update of each group's frequency covariance
            cs, dblk = LAST[0]
            for i, (Ut_, lt_, Uf_, lf_) in enumerate(groups):
                Mf = torch.einsum('baj,bam->ajm', cs[i], cs[i].conj()) + \
                    torch.diagonal(dblk[i], dim1=1, dim2=3).permute(0, 3, 1, 2).sum(0).to(C128)
                F = (Mf / lt_[:, None, None]).sum(0) / (B * len(lt_))
                lF, UF = _eigh(F); F = (UF*lF.clamp(min=0)) @ UF.conj().T
                Rfs[i] = Rfs[i] - (Uf_*lf_) @ Uf_.conj().T + Uf_ @ F @ Uf_.conj().T
            groups = [_basis(Rts[i], Rfs[i], rmx[i]) for i in range(len(gl))]
        if itr in rte:                        # EM update of each group's time covariance
            cs, dblk = LAST[0]
            for i, (Ut_, lt_, Uf_, lf_) in enumerate(groups):
                Mt = torch.einsum('baj,bcj->jac', cs[i], cs[i].conj()) + \
                    torch.diagonal(dblk[i], dim1=2, dim2=4).permute(0, 3, 1, 2).sum(0).to(C128)
                T = (Mt / lf_[:, None, None]).sum(0) / (B * len(lf_))
                lT, UT = _eigh(T); T = (UT*lT.clamp(min=0)) @ UT.conj().T
                Rt = Rts[i] - (Ut_*lt_) @ Ut_.conj().T + Ut_ @ T @ Ut_.conj().T
                dg = torch.diagonal(Rt).real.clamp(min=1e-9).sqrt()
                Rts[i] = Rt / torch.outer(dg, dg)
            groups = [_basis(Rts[i], Rfs[i], rmx[i]) for i in range(len(gl))]
        if itr in rti and n0 < RT_N0MAX and bool((w > 0).all()):   # Doppler spectrum from data
            a = w*n0
            q = yt * xm.conj() / a
            ne = float((n0/a).mean())
            if len(gl) == 1:
                Rts = [_rt_data(q, ne)]
            else:
                Rf = sum(Rfs)
                Rts = [_rt_data(q, ne, _wiener(Rg, Rf, ne)) for Rg in Rfs]
            groups = [_basis(Rts[i], Rfs[i], rmx[i]) for i in range(len(gl))]
    if SLIP and n0 < SLIP_N0MAX:
        h = _slip_fix(groups, yt, n0, h)
    return h


def _loglik(yt, h, n0):
    """Per-RE data log-likelihood log sum_s exp(-|y - h s|^2/n0) (16-QAM, separable), [B,14,72]."""
    yt = yt.to(C64); h = h.to(C64)
    g2 = (h.real**2 + h.imag**2).clamp(min=1e-12)
    z = yt * h.conj() / g2
    p = g2 / n0
    tot = 0
    for comp in (z.real, z.imag):
        l = -(comp[..., None] - PAM4)**2 * p[..., None]
        tot = tot + torch.logsumexp(l, -1)
    return tot


def _em_run(groups, yt, n0, step, einfl, dist, n_iter):
    """Plain data-aided EM with fixed statistics; per-entry schedule (step [B], einfl [B], dist [B,14])."""
    B = yt.shape[0]
    pm = PM_T[None].expand(B, -1, -1); px = PX_T[None].expand(B, -1, -1)
    w = torch.where(pm, 1.0/n0, 0.0).double()
    bb = torch.where(pm, px.conj(), torch.zeros((), dtype=C128)) * yt / n0
    hprev = None
    for itr in range(n_iter + 1):
        h, e = _solve(groups, w, bb, need_e=itr < n_iter)
        if itr == n_iter:
            break
        if itr > 0:
            h = hprev + OVR*(h - hprev)
        hprev = h
        xd, v = _demap(yt, h, e*einfl[:, None, None], n0)
        xm = torch.where(pm, px, xd)
        w = torch.where(pm, torch.ones_like(v), xd.abs()**2 + v) / n0
        bb = xm.conj() * yt / n0
        act = (dist <= (itr + 1)*step[:, None])[:, :, None]
        w = torch.where(act, w, torch.zeros_like(w))
        bb = torch.where(act, bb, torch.zeros_like(bb))
    return h, LASTU[0]


_DL = torch.tensor([abs(l - PS[0]) for l in range(NS)]).double()
_DR = torch.tensor([abs(l - PS[1]) for l in range(NS)]).double()
_DD = DIST + EDGE_PEN*EDGE
# restart strategies: (step, einfl, dist, n_iter)
_STRATS = [(0.25, 3.0, _DD, 18), (1.0, 3.0, _DL, 16), (1.0, 3.0, _DR, 16), (0.5, 10.0, _DD, 18)]


def _slip_fix(groups, yt, n0, h):
    """Detect examples whose EM got stuck in a wrong local optimum (outlying data likelihood)
    and re-run EM for them from several alternative schedules; keep the solution with the best
    MAP objective (data log-likelihood minus prior energy)."""
    dm = ~PM_T
    u0 = LASTU[0]
    Jd = _loglik(yt, h, n0)[:, dm].sum(1).double()
    med = Jd.median(); mad = (Jd - med).abs().median() * 1.4826 + 1e-9
    zj = (Jd - med) / mad
    order = torch.argsort(zj)
    flag = [int(b) for b in order[:SLIP_MAX] if float(zj[b]) < SLIP_Z]
    if not flag:
        return h
    fl = torch.tensor(flag); nf = len(flag); ns = len(_STRATS)
    obj0 = (Jd - u0)[fl]
    y_c = yt[fl][None].expand(ns, -1, -1, -1).reshape(-1, NS, NF)
    step = torch.tensor([st[0] for st in _STRATS]).double().repeat_interleave(nf)
    einfl = torch.tensor([st[1] for st in _STRATS]).float().repeat_interleave(nf)
    dist = torch.stack([st[2] for st in _STRATS]).repeat_interleave(nf, 0)
    n_iter = max(st[3] for st in _STRATS)
    h_c, u_c = _em_run(groups, y_c, n0, step, einfl, dist, n_iter)
    obj_c = (_loglik(y_c, h_c, n0)[:, dm].sum(1).double() - u_c).reshape(ns, nf)
    best, arg = obj_c.max(0)
    h = h.clone()
    for i, b in enumerate(flag):
        if float(best[i]) > float(obj0[i]) + SLIP_MARGIN:
            h[b] = h_c[int(arg[i])*nf + i]
    return h


def _fallback(y, no):
    """LS on pilots + linear interpolation in frequency and time (safety net)."""
    yt = y.to(C128)
    hp = (yt[:, PS][:, :, KP] / XP[None]).numpy()
    k = np.arange(NF)
    hf = np.stack([[np.interp(k, KP, hp[b, i].real) + 1j*np.interp(k, KP, hp[b, i].imag)
                    for i in range(2)] for b in range(hp.shape[0])])
    t = (np.arange(NS) - PS[0]) / (PS[1] - PS[0])
    h = hf[:, 0][:, None]*(1 - t)[None, :, None] + hf[:, 1][:, None]*t[None, :, None]
    return torch.tensor(h)


def estimate_channel(y, no):
    nt = torch.get_num_threads()
    torch.set_num_threads(1)          # small matrices: avoid BLAS thread contention
    try:
        with torch.no_grad():
            try:
                h = _estimate(y, no)
                if not bool(torch.isfinite(h.real).all() and torch.isfinite(h.imag).all()):
                    raise ValueError('non-finite estimate')
            except Exception:
                h = _fallback(y, no)
    finally:
        torch.set_num_threads(nt)
    return h.to(torch.complex64)
