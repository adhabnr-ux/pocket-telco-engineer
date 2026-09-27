"""
Statistics-agnostic OFDM channel estimator.

1. Pilot stage: estimate the power-delay profile from the batch (NNLS covariance
   matching at low SNR, multiple-snapshot sparse Bayesian learning at higher SNR),
   the pilot-to-pilot time correlation (Jakes model, optional LOS rotation), and run
   a 2-D Kronecker LMMSE on the 72 pilots.
2. Data-aided stage: variational-Bayes EM over the unknown 16-QAM symbols. Each
   iteration soft-demaps every data RE (exact 4-PAM posteriors per I/Q), turns it
   into a Gaussian pseudo-observation of the channel, and solves the weighted
   Kronecker-prior LMMSE in a reduced eigen-subspace (preconditioned CG).
   Fast channels are processed progressively outward from the pilot symbols with
   an SNR-capped (annealed) noise level; leave-one-out cavities avoid self-confirmation.
3. EM covariance refinement: time and frequency covariances are re-estimated from the
   posterior second moments of the whole grid, followed by further VB iterations.
Only y, no and link_config are used.
"""
import time
import numpy as np, torch
from scipy.optimize import nnls
from scipy.special import j0
from link_config import (PILOT_OFDM_SYMBOLS, PILOT_SYMBOLS, PILOT_MASK, FFT_SIZE,
                         SUBCARRIER_SPACING, OFDM_SYMBOL_DURATION, NUM_OFDM_SYMBOLS)
torch.set_num_threads(1)

OPT = dict(
    tmin=-0.5, tmax=24.0, os=4, floor=1e-5,        # delay grid [Ts], oversampling, PDP floor
    tmax_long=33.0, long_thr=0.003, long_snr=99.0,  # grid extension for very long channels
    sbl_snr=4.0, sbl_it=50,                         # SBL-PDP above this SNR [dB]
    rt_proj=0.1, rotthr=0.8,                        # pilot time-correlation estimate
    prog_thr=0.95, nslow=2, prep=2, nfull=1,        # VB schedule
    eps=1e-2, rmax=200, cg=5,                       # subspace / solver
    loo=True, loo_max=0.8, ann_snr=8.0,             # extrinsic E-step, annealing cap [dB]
    em_cycles=4, em_iters=2,                        # covariance EM refinement
    tbudget=0.25, tbudget_main=0.3)                 # runtime guards [s]

NS, NK = NUM_OFDM_SYMBOLS, FFT_SIZE
DF = SUBCARRIER_SPACING; TS = 1.0/(NK*DF); T = OFDM_SYMBOL_DURATION
PS = list(PILOT_OFDM_SYMBOLS)
KP = np.arange(0, NK, 2); K = np.arange(NK); NP = len(KP)
XP = PILOT_SYMBOLS[PS][:, KP]
M = np.arange(NP)
_xg = np.linspace(0, 3.8317, 4000); _jg = j0(_xg)
PM = torch.tensor(PILOT_MASK); PSYM = torch.tensor(PILOT_SYMBOLS, dtype=torch.complex64)
DIST = torch.tensor(np.min(np.abs(np.arange(NS)[:, None] - np.array(PS)[None]), 1))
_A = float(1/np.sqrt(10))      # 16-QAM = two 4-PAM {-3,-1,1,3}*_A


def _grids(tmax):
    tau = np.arange(OPT['tmin'], tmax + 1e-9, 1.0/OPT['os'])*TS
    w = np.sqrt(NP - M)
    Aw = np.exp(-2j*np.pi*np.outer(2*M*DF, tau))*w[:, None]
    return dict(tau=tau, w=w, A=np.concatenate([Aw.real, Aw.imag]),
                EKt=torch.tensor(np.exp(-2j*np.pi*np.outer(K*DF, tau))),
                Phi=torch.tensor(np.exp(-2j*np.pi*np.outer(KP*DF, tau))))
G_S = _grids(OPT['tmax']); G_L = _grids(OPT['tmax_long'])


def safe_nnls(A, b):
    try:
        p, _ = nnls(A, b, maxiter=50*A.shape[1])
    except Exception:
        p = np.maximum(np.linalg.lstsq(A, b, rcond=None)[0], 0)
    if not np.all(np.isfinite(p)) or p.sum() <= 0:
        p = np.ones(A.shape[1])
    return p

# ---------------- statistics from pilots ----------------
def est_pdp_nnls(hp, no_, G):
    # covariance matching: pilot frequency autocorrelation vs PDP on delay grid
    r = np.array([np.mean(hp[:, :, m:]*np.conj(hp[:, :, :NP-m])) for m in M])
    P0 = max(r[0].real - no_, 1e-3*no_); r[0] = P0
    b = np.concatenate([(r*G['w']).real, (r*G['w']).imag])
    return safe_nnls(G['A'], b), P0

def est_pdp_sbl(hp, no_, G):
    # multiple-snapshot sparse Bayesian learning (type-II ML, MacKay updates)
    Phi = G['Phi']; PhiH = Phi.conj(); J = Phi.shape[1]
    Y = torch.tensor(hp.reshape(-1, NP).T)
    P0 = max(float(np.mean(np.abs(hp)**2)) - no_, 1e-3*no_)
    g = torch.full((J,), P0/J, dtype=torch.float64)
    S = (Y @ Y.conj().T)/Y.shape[1]
    I = torch.eye(NP, dtype=torch.complex128)*no_
    for _ in range(OPT['sbl_it']):
        CiPhi = torch.linalg.solve(I + (Phi*g) @ PhiH.T, Phi)
        q = (CiPhi.conj()*(S @ CiPhi)).sum(0).real
        dg = (PhiH*CiPhi).sum(0).real
        g = (g*q.clamp_min(0)/dg.clamp_min(1e-12)).clamp_min(1e-12)
    g = g.numpy()
    return g, max(g.sum(), 1e-9)

def build_Rf(p, P0, G):
    p = p + OPT['floor']*P0/len(p)
    p = p*(P0/p.sum())
    EK = G['EKt']
    return ((EK*torch.tensor(p)) @ EK.conj().T).numpy()

def est_Rt(hp, Rf, no_):
    # correlation between the two pilot symbols, measured inside the signal subspace
    l, U = torch.linalg.eigh(torch.tensor(Rf[np.ix_(KP, KP)]))
    U = U[:, l > OPT['rt_proj']*no_].numpy()
    hq = hp @ U.conj()
    nr = U.shape[1]
    p0 = np.mean(np.sum(np.abs(hq[:, 0])**2, -1)) - no_*nr
    p1 = np.mean(np.sum(np.abs(hq[:, 1])**2, -1)) - no_*nr
    c = np.mean(np.sum(hq[:, 1]*np.conj(hq[:, 0]), -1))/np.sqrt(max(p0, 1e-9)*max(p1, 1e-9))
    if abs(c) > OPT['rotthr']:          # strong correlation: allow a common Doppler rotation (LOS)
        rho = min(abs(c), 1.0); ph = np.angle(c)
    else:                               # weak correlation: real Jakes, possibly past first zero
        rho = np.clip(c.real, _jg.min(), 1.0); ph = 0.0
    x = _xg[np.argmin(np.abs(_jg - rho))]
    gap = PS[1] - PS[0]
    d = np.arange(NS)[:, None] - np.arange(NS)[None, :]
    return j0(x*d/gap)*np.exp(1j*ph*d/gap)

# ---------------- Kronecker subspace machinery ----------------
def eig2(Rt, Rf):
    lt, Ut = torch.linalg.eigh(torch.tensor(Rt)); lf, Uf = torch.linalg.eigh(torch.tensor(Rf))
    return lt.clamp_min(0).numpy(), Ut.numpy(), lf.clamp_min(0).numpy(), Uf.numpy()

def subspace(E, no_):
    lt, Ut, lf, Uf = E
    thr = OPT['eps']*no_
    it = np.where(lt*lf.max() > thr)[0]; jf = np.where(lf*lt.max() > thr)[0]
    while len(it)*len(jf) > OPT['rmax']:
        thr *= 1.5
        it = np.where(lt*lf.max() > thr)[0]; jf = np.where(lf*lt.max() > thr)[0]
    lam = np.outer(lt[it], lf[jf])
    return (torch.tensor(Ut[:, it], dtype=torch.complex64), torch.tensor(Uf[:, jf], dtype=torch.complex64),
            torch.tensor(lam, dtype=torch.float32))

def post_diag(E, wbar):
    # approximate posterior variance per RE (uniform per-example weight wbar)
    lt, Ut, lf, Uf = E
    lam = torch.tensor(np.outer(lt, lf), dtype=torch.float32)
    g = lam[None]/(1 + torch.as_tensor(wbar, dtype=torch.float32)[:, None, None]*lam[None])
    return (torch.tensor(np.abs(Ut)**2, dtype=torch.float32)[None] @ g
            @ torch.tensor((np.abs(Uf)**2).T, dtype=torch.float32)[None])

def wsolve_cg(z, om, sub, c0, nit):
    # posterior mean of h = Ut (sqrt(lam) * c) Uf^T with c ~ CN(0, I), sites (z, om); PCG in c
    Ut, Uf, lam = sub
    sl = lam.sqrt().to(torch.complex64)[None]
    omc = om.to(torch.complex64)
    UfT = Uf.T.contiguous(); UtH = Ut.conj().T.contiguous(); Ufc = Uf.conj()
    def Aop(c):
        return c + sl*(UtH[None] @ (omc*(Ut[None] @ (sl*c) @ UfT[None])) @ Ufc[None])
    b = sl*(UtH[None] @ (omc*z) @ Ufc[None])
    pre = 1.0/(1.0 + lam[None]*om.mean((1, 2))[:, None, None])
    if c0 is None:
        x = torch.zeros_like(b); r = b.clone()
    else:
        x = c0; r = b - Aop(x)
    zz = pre*r; p = zz.clone(); rz = (r.conj()*zz).sum((1, 2)).real
    for _ in range(nit):
        Ap = Aop(p)
        a = rz/(p.conj()*Ap).sum((1, 2)).real.clamp_min(1e-30)
        x = x + a[:, None, None]*p; r = r - a[:, None, None]*Ap
        zz = pre*r; rzn = (r.conj()*zz).sum((1, 2)).real
        p = zz + (rzn/rz.clamp_min(1e-30))[:, None, None]*p; rz = rzn
    return Ut[None] @ (sl*x) @ UfT[None], x

# ---------------- VB E-step (16-QAM, separable I/Q) ----------------
def pam_post(c, s2):
    # exact posterior mean / 2nd moment of a 4-PAM symbol given c = x + noise (complex var s2)
    inv = 1.0/s2
    au = ((2*_A)*inv)*c.abs()
    t = torch.exp(-2*au)
    dl = 2*au - (8*_A*_A)*inv
    dd = torch.exp(-dl.abs())
    pos = dl > 0
    e1 = torch.where(pos, dd, 1.0); e3 = torch.where(pos, 1.0, dd)
    t3 = t*t*t
    a1 = e1*(1 + t); a3 = e3*(1 + t3)
    den = a1 + a3
    m1 = torch.sign(c)*_A*(e1*(1 - t) + 3*e3*(1 - t3))/den
    m2 = (_A*_A)*(a1 + 9*a3)/den
    return m1, m2

def sites(y, hh, ev, no_d, no_m, z, om, mask=None):
    # VB E-step (with leave-one-out cavity) -> Gaussian pseudo-observations (z, om) of h
    if OPT['loo'] and z is not None:
        g = (om*ev).clamp(0, OPT['loo_max'])
        hh = (hh - g*z)/(1 - g); ev = ev/(1 - g)
    g = hh.abs()**2 + ev
    xe = y*hh.conj()/g
    s2 = no_d/g
    a1, a2 = pam_post(xe.real, s2); b1, b2 = pam_post(xe.imag, s2)
    xb = torch.where(PM, PSYM, torch.complex(a1, b1))
    m2 = torch.where(PM, torch.ones_like(a2), a2 + b2)
    om = m2/no_m
    if mask is not None:
        om = om*mask
    return y*xb.conj()/m2, om

def em_stats(hh, E, wbar):
    # EM update of second-order statistics: E[h h^H | y] ~ hh hh^H + Sigma
    lt, Ut, lf, Uf = E
    B = hh.shape[0]
    lam = torch.tensor(np.outer(lt, lf))
    pm = (lam[None]/(1 + torch.tensor(wbar)[:, None, None]*lam[None])).mean(0)
    Uft = torch.tensor(Uf); Utt = torch.tensor(Ut)
    Hd = hh.to(torch.complex128)
    Hf = Hd.reshape(-1, NK)
    Sf = (Hf.T @ Hf.conj())/(B*NS) + (Uft*(pm.sum(0)/NS)) @ Uft.conj().T
    Ht = Hd.permute(1, 0, 2).reshape(NS, -1)
    St = (Ht @ Ht.conj().T)/(B*NK) + (Utt*(pm.sum(1)/NK)) @ Utt.conj().T
    St = St/St.diagonal().real.mean()
    return St.numpy(), Sf.numpy()

# ---------------- main ----------------
def estimate_channel(y, no):
    t_start = time.perf_counter()
    h = _estimate(y, no, t_start)
    if not bool(torch.isfinite(h).all()):
        h = torch.nan_to_num(h)
    return h

def _estimate(y, no, t_start):
    y = torch.as_tensor(y).to(torch.complex64)
    yn = y.numpy().astype(np.complex128)
    no_ = float(torch.as_tensor(no, dtype=torch.float32).mean())
    snr_db = 10*np.log10(1/no_)
    B = yn.shape[0]
    hp = yn[:, PS][:, :, KP]*np.conj(XP)[None]

    # --- delay profile (with automatic grid extension for very long channels) ---
    pl, P0 = est_pdp_nnls(hp, no_, G_L)
    frac = pl[G_L['tau'] > (OPT['tmax'] + 0.5)*TS].sum()/max(pl.sum(), 1e-12)
    G = G_L if (frac > OPT['long_thr'] and snr_db >= OPT['long_snr']) else G_S
    if snr_db >= OPT['sbl_snr']:
        p, P0 = est_pdp_sbl(hp, no_, G)
    elif G is G_L:
        p = pl
    else:
        p, P0 = est_pdp_nnls(hp, no_, G)
    Rf = build_Rf(p, P0, G)
    Rt = est_Rt(hp, Rf, no_)

    # --- pilot-only 2-D LMMSE (initialisation) ---
    Rtt = torch.tensor(Rt); Rft = torch.tensor(Rf)
    Rpp = torch.kron(Rtt[PS][:, PS], Rft[KP][:, KP]) + no_*torch.eye(2*NP, dtype=torch.complex128)
    Rhp = torch.kron(Rtt[:, PS], Rft[:, KP])
    W = torch.linalg.solve(Rpp.T, Rhp.T).T
    hh = (torch.tensor(hp.reshape(B, -1)) @ W.T).reshape(B, NS, NK).to(torch.complex64)
    ev0 = ((Rtt.diagonal().real[:, None]*Rft.diagonal().real[None, :]).reshape(-1)
           - (W*Rhp.conj()).sum(1).real).numpy()
    ev = torch.tensor(np.broadcast_to(np.maximum(ev0.reshape(NS, NK), 0), (B, NS, NK)).copy(),
                      dtype=torch.float32)

    # --- data-aided VB iterations (progressive for fast channels) ---
    E = eig2(Rt, Rf); sub = subspace(E, no_)
    slow = abs(Rt[PS[0], PS[0] + 4]) > OPT['prog_thr']
    if slow:
        sched = [99]*OPT['nslow']
    else:
        sched = [d for d in range(1, int(DIST.max()) + 1) for _ in range(OPT['prep'])] + [99]*OPT['nfull']
    c = None; z = None; om = None
    for dmax in sched:
        if time.perf_counter() - t_start > OPT['tbudget_main']:
            break
        no_d = max(no_, 10**(-OPT['ann_snr']/10)) if dmax < 99 else no_
        mask = (DIST <= dmax).float()[None, :, None]
        z, om = sites(y, hh, ev, no_d, no_d, z, om, mask)
        hh, c = wsolve_cg(z, om, sub, c, OPT['cg'])
        ev = post_diag(E, om.mean((1, 2)))

    # --- EM refinement of the time/frequency covariances ---
    for _ in range(OPT['em_cycles']):
        if time.perf_counter() - t_start > OPT['tbudget']:
            break
        wbar = om.mean((1, 2)).numpy().astype(np.float64)
        Rt, Rf = em_stats(hh, E, wbar)
        E = eig2(Rt, Rf); sub = subspace(E, no_)
        ev = post_diag(E, wbar); c = None
        for _ in range(OPT['em_iters']):
            z, om = sites(y, hh, ev, no_, no_, z, om)
            hh, c = wsolve_cg(z, om, sub, c, OPT['cg'])
            ev = post_diag(E, om.mean((1, 2)))
    return hh
