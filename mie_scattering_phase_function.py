import numpy as np
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
import scipy

# References:
# - https://www.oceanopticsbook.info/view/theory-electromagnetism/level-2/mie-theory-overview
# - Bohren, C.F. and D.R. Huffman, 1983. Absorption and Scattering of Light by Small Particles, John Wiley & Sons
#   https://staff.cs.manchester.ac.uk/~fumie/internal/scattering.pdf

# symbol reference:
# wavelength = wavelength of the light in vacuo
# rho = radius of the scattering particle
# n_s = real index of refraction of the spherical particle
# m_s = complex index of refraction of the spherical particle
# n_m = real index of refraction of the surrounding medium
# m_m = complex index of refraction of the surrounding medium
# theta = scattering angle
# mu = cos(theta)

wavelength = 632.8e-9 # He-Ne laser (red) in vacuo
rho = 50e-9 # 50 nm sphere
n_s = 1.332 # water
k_s = 1.39e-8 # absorption of water
m_s = n_s + 1j*k_s # water with some absorption
n_m = 1 # vacuum
m_m = n_m # no absorption

# Reproduce the first plot on https://www.oceanopticsbook.info/view/theory-electromagnetism/level-2/mie-theory-examples
rho = 0.263e-6
n_s = 1.33
k_s = 1e-8
n_m = 1
wavelength = 550e-9
m_s = n_s + 1j*k_s
m_m = n_m

# Reproduce the second plot
# rho = 0.5e-6
# n_s = 1.37
# k_s = 1.5e-2
# n_m = 1.34
# wavelength = 500e-9
# m_s = n_s + 1j*k_s
# m_m = n_m

# Reproduce figure 2 in https://research.nvidia.com/labs/rtr/approximate-mie/publications/approximate-mie.pdf
# rho = 5e-6
# n_s = 1.33
# k_s = 1e-8
# n_m = 1
# wavelength = 550e-9
# m_s = n_s + 1j*k_s
# m_m = n_m

speed_of_light = 299792458
# wave number
k = 2 * np.pi / wavelength
# radians per second
omega = k * speed_of_light
# wave number inside the sphere
k_1 = k * n_m
# size parameter x
x = k_1 * rho
# relative refractive index
m = m_s / n_m

# A well established rule seems to be that the number of terms
# that must be computed is the integer closest to
N_max = round(x + 4*np.cbrt(x) + 2)

if N_max > 2e4:
    # Numerical precision breaks down at some point, so let's just choose an arbitrary cap
    raise RuntimeError(f"Too many terms to compute: {N_max}. Exiting...")

print(f"Computing {N_max} terms...")

# Less efficient than using Scipy's Riccati-Bessel functions, but supports complex numbers
def riccati_bessel(max_order: int, z: complex):
    orders = np.arange(max_order + 1)
    j_n = scipy.special.spherical_jn(orders, z)
    j_n_dz = scipy.special.spherical_jn(orders, z, derivative=True)
    psi = z * j_n
    psi_dz = j_n + z * j_n_dz

    y_n = scipy.special.spherical_yn(orders, z)
    y_n_dz = scipy.special.spherical_yn(orders, z, derivative=True)
    xi = psi + 1j * z * y_n
    xi_dz = psi_dz + 1j * (y_n + z * y_n_dz)

    return psi, psi_dz, xi, xi_dz

# a_n and b_n are given by equation 7.
# Bohren and Huffman page 101 somewhat clarifies what psi and xi refer to.
# psi corresponds to the Riccati-Bessel function of the first kind,
# which is available as scipy.special.riccati_jn.
# xi(rho) corresponds to rho * h_n^(1)(rho), where h_n^(1) is the spherical
# hankel function of the first kind. This in turn corresponds to:
# rho*J_n(x) + i*rho*Y_a(x)) = riccati_jn + i*riccati_yn, where riccati_yn
# is the Riccati-Bessel function of the second kind, available in Scipy.
# Computing them together saves an extra step of computing Riccati-Bessel
# functions of the first kind.
psi_x, psi_dx, xi_x, xi_dx = riccati_bessel(N_max, x)
psi_mx, psi_dmx, xi_mx, xi_dmx = riccati_bessel(N_max, m * x)
denom_a = m * psi_mx * xi_dx - xi_x * psi_dmx
denom_b = psi_mx * xi_dx - m * xi_x * psi_dmx
a = (m * psi_mx * psi_dx - psi_x * psi_dmx) / denom_a
b = (psi_mx * psi_dx - m * psi_x * psi_dmx) / denom_b

# scattering angles
num_bins = 180
# using same coordinate convention for spherical coordinates as in figure 4.1
theta = np.linspace(0, np.pi, num_bins)
costheta = np.cos(theta)

# S_1 and S_2
S = np.zeros((2, num_bins), dtype=np.complex64)

# initial conditions
n = 1
pi_n_sub1 = 0
pi_n = 1
tau_n = costheta
while n <= N_max:
    # S_1 and S_2 are given by equation 6
    norm = (2*n + 1) / (n * (n + 1))
    S[0] += norm * (a[n] *  pi_n + b[n] * tau_n)
    S[1] += norm * (a[n] * tau_n + b[n] *  pi_n)

    # compute the next pi & tau
    n += 1
    pi_next = (2*n - 1) / (n - 1) * costheta * pi_n - n / (n - 1) * pi_n_sub1
    pi_n_sub1 = pi_n
    pi_n = pi_next
    tau_n = n * costheta * pi_n - (n + 1) * pi_n_sub1

def get_intensity(amplitudes: np.ndarray):
    return np.real(amplitudes * amplitudes.conj())

I_parallel = get_intensity(S[1])
I_perpendicular = get_intensity(S[0])
beta_tilde = (I_parallel + I_perpendicular) / 2 # unpolarized phase function

fig = plt.figure(figsize=(9, 4))

ax = plt.subplot(121)
unpolarized_color = "black"
degrees = np.rad2deg(theta)
ax.plot(degrees, I_parallel, label=r"$I_{||}$", color="C0")
ax.plot(degrees, I_perpendicular, label=r"$I_{\perp}$", color="C1")
ax.plot(degrees, beta_tilde, label=r"$\tilde{\beta}$", color=unpolarized_color)
ax.set_yscale("log")
ax.set_xlabel("Scattering angle in degrees")
ax.xaxis.set_major_formatter(ticker.FormatStrFormatter("%d°"))
ax.set_xticks(np.arange(0, 181, 30))

def mirror_copy(x: np.ndarray):
    return np.hstack([ x, x[1:][::-1] ])

theta = np.linspace(0, 2 * np.pi, num_bins * 2 - 1)
I_parallel = mirror_copy(I_parallel)
I_perpendicular = mirror_copy(I_perpendicular)
beta_tilde = mirror_copy(beta_tilde)
ax = plt.subplot(122, projection="polar")
ax.plot(theta, I_parallel, color="C0")
ax.plot(theta, I_perpendicular, color="C1")
ax.plot(theta, beta_tilde, color=unpolarized_color)
ax.set_rlim(0)
ax.set_rscale("symlog")
ax.set_xlabel("Scattering angle in degrees")

fig.suptitle("Unnormalized scattering intensity")
fig.legend(fontsize=14)

plt.show()