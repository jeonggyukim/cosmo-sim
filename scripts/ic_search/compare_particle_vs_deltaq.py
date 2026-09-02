import numpy as np, h5py
import paths
T = paths.ROOT
h, N, L = 0.6711, 64, 500.0            # L in Mpc/h
dx = L/N

with h5py.File(f"{T}/n64_2lpt_dm_z200_L500/ics_dm_n64.hdf5") as f:
    x = f["PartType1/Coordinates"][:].astype(float)*h     # Mpc -> Mpc/h
    pid = f["PartType1/ParticleIDs"][:].astype(np.int64)

# --- displacement: reconstruct the sc lattice site from the particle ID ---
best = None
for order in ("ijk", "kji"):
    idx = pid - pid.min()
    a, b, c = idx//(N*N), (idx//N) % N, idx % N
    ijk = np.stack([a, b, c], 1) if order == "ijk" else np.stack([c, b, a], 1)
    for off in (0.0, 0.5):
        q = (ijk + off)*dx
        psi = x - q
        psi -= L*np.round(psi/L)
        s = psi.std()
        if best is None or s < best[0]:
            best = (s, order, off, psi)
sig1d, order, off, psi = best
print(f"lattice convention: ID order {order}, offset {off}*dx")
print(f"rms displacement per component : {sig1d:.4f} Mpc/h = {sig1d/dx:.4f} cells")
kny = np.pi*N/L
for k in (0.1, 0.2, kny):
    print(f"  k={k:.3f}: k^2 sigma_1d^2 = {(k*sig1d)**2:.3e}")

# --- particle P(k) with CIC + interlacing, deconvolved, on the delta(q) bins ---
def cic(pos):
    g = np.zeros((N, N, N))
    u = pos/dx
    i0 = np.floor(u).astype(int); w1 = u - i0; w0 = 1.0 - w1
    i0 %= N; i1 = (i0 + 1) % N
    for a in (0, 1):
        for b in (0, 1):
            for c in (0, 1):
                wa = (w0[:,0] if a==0 else w1[:,0])*(w0[:,1] if b==0 else w1[:,1])*(w0[:,2] if c==0 else w1[:,2])
                np.add.at(g, ((i0[:,0] if a==0 else i1[:,0]),
                              (i0[:,1] if b==0 else i1[:,1]),
                              (i0[:,2] if c==0 else i1[:,2])), wa)
    return g/g.mean() - 1.0

kf = 2*np.pi/L
kax = np.fft.fftfreq(N, 1.0/N)*kf
d1 = np.fft.fftn(cic(x))/N**3
d2 = np.fft.fftn(cic((x + 0.5*dx) % L))/N**3
ph = np.exp(0.5j*dx*(kax[:,None,None] + kax[None,:,None] + kax[None,None,:]))
dkp = 0.5*(d1 + d2*ph)                                    # interlaced
sinc = lambda u: np.sinc(u/np.pi)
W = (sinc(kax[:,None,None]*dx/2)*sinc(kax[None,:,None]*dx/2)*sinc(kax[None,None,:]*dx/2))**2
Pp = (L**3*np.abs(dkp/W)**2).ravel()

with h5py.File(f"{T}/n64_deltaq_z200_L500/delta_q_n64.hdf5") as f:
    d = f["delta_q"][:].astype(float)
Pq = (L**3*np.abs(np.fft.fftn(d)/N**3)**2).ravel()

kk = np.sqrt(kax[:,None,None]**2 + kax[None,:,None]**2 + kax[None,None,:]**2).ravel()
edges = np.arange(0.5, N/2+1.0)*kf
ib = np.digitize(kk, edges); nb = len(edges)-1
kb = np.array([kk[ib==i+1].mean() for i in range(nb)])
Pbq = np.array([Pq[ib==i+1].mean() for i in range(nb)])
Pbp = np.array([Pp[ib==i+1].mean() for i in range(nb)])
r = Pbp/Pbq

print(f"\nPoisson shot noise if the load were random: V/N_p = {L**3/N**3:.1f} (Mpc/h)^3")
print(f"measured P(k) range: {Pbq.min():.3g} to {Pbq.max():.3g} (Mpc/h)^3\n")
print(f"{'k [h/Mpc]':>10} {'P_delta(q)':>12} {'P_particle':>12} {'ratio':>8}")
for i in range(0, nb, max(1, nb//12)):
    print(f"{kb[i]:10.4f} {Pbq[i]:12.5e} {Pbp[i]:12.5e} {r[i]:8.4f}")
lo = kb < 0.5*kny
print(f"\nsame-bin ratio, k < k_Ny/2 : median {np.median(r[lo]):.4f}, max dev {100*np.abs(r[lo]-1).max():.2f}%")
print(f"same-bin ratio at k_Ny     : {r[-1]:.4f}")
