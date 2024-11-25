import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import LogNorm
import scipy
from PIL import Image, ImageDraw, ImageFont
import argparse
import signal

# Windows won't respond to Crtl C when showing plots without this
signal.signal(signal.SIGINT, signal.SIG_DFL)

speed_of_light = 299792458 # m/s
epsilon_0 = 8.8541878188e-12 # electric permittivity of free space (F m^-1)
mu_0 = 1.25663706127e-6 # magnetic permeability of free space (N A^-2)
epsilon_r = 80.2 # relative electric permittivity of water (F m^-1)
mu_r = 0.999992 # relative magnetic permeability of water (N A^-2)

wavelength = 550e-9
rho = 0.263e-6
n_s = 1.33
k_s = 1e-8
n_m = 1
k_m = 0

parser = argparse.ArgumentParser(
    prog="MieScattering",
    description="Render an analytical approximation of electric field for a plane wave interacting with a sphere",
    formatter_class=argparse.ArgumentDefaultsHelpFormatter # show default values
)

parser.add_argument("filename", nargs='?', type=str, default="mie-sim.gif", help="the output path to save the animated gif to")

args_mie = parser.add_argument_group("scattering parameters")
args_mie.add_argument("-w", "--wavelength", type=float, default=wavelength, help="the wavelength of the plane wave in vacuo")
args_mie.add_argument("-r", "--radius", type=float, default=rho, help="the radius of the sphere")
args_mie.add_argument("-n", "--refractive-index-sphere", type=float, default=n_s, help="real refractive index of the sphere")
args_mie.add_argument("-m", "--refractive-index-medium", type=float, default=n_m, help="real refractive index of the surrounding medium")
args_mie.add_argument("-a", "--absorption-sphere", type=float, default=k_s, help="absorption coefficient of the sphere (imaginary component of its refractive index)")
args_mie.add_argument("-b", "--absorption-medium", type=float, default=k_m, help="absorption coefficient of the surrounding medium (imaginary component of its refractive index)")

args_animation = parser.add_argument_group("animation parameters")
args_animation.add_argument("-s", "--scale", type=float, default=10, help="set the scale of the animation in multiples of the sphere diameter. A value of 1 will perfectly contain the sphere within the frame")
args_animation.add_argument("-d", "--dimensions", nargs=2, type=int, default=[512, 512], help="set the pixel width and height of the output animation")
args_animation.add_argument("-l", "--log-scale", type=bool, default=False, action=argparse.BooleanOptionalAction, help="use logarithmic scale for the field intensity")
args_animation.add_argument("-N", "--num-terms", type=int, help="manually specify the number of terms to compute")
args_animation.add_argument("--colormap", type=str, default="viridis", help="the name of a matplotlib colormap to use for the field intensity")
args_animation.add_argument("--show-sphere-boundary", type=bool, default=False, action=argparse.BooleanOptionalAction, help="draw a circle on the sphere's boundary")
arg_show_incident = args_animation.add_argument("--show-incident", type=bool, default=False, action=argparse.BooleanOptionalAction, help="include the incident plane wave in the animation")
args_animation.add_argument("--font", type=str, default="DejaVuSansMono.ttf", help="the path to or name of an installed font to use when drawing text")
args_animation.add_argument("--fps", type=int, default=30, help="frame rate of the animation")
args_animation.add_argument("--duration", type=float, default=3, help="number of seconds to complete one phase cycle (2 pi)")
args_animation.add_argument("--text", type=bool, default=True, action=argparse.BooleanOptionalAction, help="Don't add any text on top of the animation")

args_magnetic = parser.add_argument_group("only relevant when simulating the magnetic field")
args_magnetic.add_argument("--magnetic", type=bool, default=False, action=argparse.BooleanOptionalAction, help="output the magnetic field intensity instead of the electric field")
args_magnetic.add_argument("--permittivity", type=float, default=epsilon_r, help="electric permittivity of the sphere relative to the medium")
args_magnetic.add_argument("--permeability", type=float, default=epsilon_r, help="magnetic permeability of the sphere relative to the medium")
args_magnetic.add_argument("--permittivity-medium", type=float, default=epsilon_0, help="electric permittivity of the surrounding medium (defaults to free space)")
args_magnetic.add_argument("--permeability-medium", type=float, default=mu_0, help="magnetic permeability of the surrounding medium (defaults to free space)")

args = parser.parse_args()

wavelength = args.wavelength
rho = args.radius
n_s = args.refractive_index_sphere
n_m = args.refractive_index_medium
k_s = args.absorption_sphere
k_m = args.absorption_medium
m_s = n_s + 1j * k_s
m_m = n_m + 1j * k_m
epsilon_0 = args.permittivity_medium
mu_0 = args.permeability_medium
epsilon_r = args.permittivity
mu_r = args.permeability
epsilon_1 = epsilon_r * epsilon_0 # electric permittivity of the sphere
mu_1 = mu_r * mu_0 # magnetic permeability of the sphere

scale = 2 * args.scale * rho # width & height of the grid in meters
grid_width = args.dimensions[0]
grid_height = args.dimensions[1]
cmap = plt.get_cmap(args.colormap, 256)
font = ImageFont.truetype(args.font, 16)

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

# wavelength = 1
# n_s = 2
# rho = 1.5
# k_s = 0
# m_s = n_s
# n_m = 1
# m_m = n_m

# wavelength = 632.8e-9 # He-Ne laser (red) in vacuo
# rho = 50e-9 # 50 nm sphere
# n_s = 1.332 # water
# k_s = 1.39e-8 # absorption of water
# m_s = n_s + 1j*k_s # water with some absorption
# n_m = 1 # vacuum
# m_m = n_m # no absorption

# Reproduce the first plot on https://www.oceanopticsbook.info/view/theory-electromagnetism/level-2/mie-theory-examples
# rho = 0.263e-6
# n_s = 1.33
# k_s = 1e-8
# n_m = 1
# wavelength = 550e-9
# m_s = n_s + 1j*k_s
# m_m = n_m

# Reproduce the second plot
# rho = 0.5e-6
# n_s = 1.37
# k_s = 1.5e-2
# n_m = 1.34
# wavelength = 500e-9
# m_s = n_s + 1j*k_s
# m_m = n_m

# Reproduce https://physics.itmo.ru/en/mie#/nearfield
# rho = 100e-9
# n_s = 4
# k_s = 0.01
# n_m = 1
# wavelength = 619.3885178e-9
# m_s = n_s + 1j*k_s
# m_m = n_m

# reproduce https://commons.wikimedia.org/wiki/File:Mie_resonances_vs_Radius.gif
# wavelength = 1
# n_s = 2
# rho = 1.5
# k_s = 0
# m_s = n_s
# n_m = 1
# m_m = n_m

# wave number
k = 2 * np.pi / wavelength
# radians per second
omega = k * speed_of_light
# wave number inside the sphere
k_1 = k * n_m
# size parameter x
x = k * rho
# relative refractive index
m = m_s / m_m
mx = m * x

# A well established rule seems to be that the number of terms
# that must be computed is the integer closest to
N_max = round(x + 4*np.cbrt(x) + 2)

if args.num_terms is not None:
    N_max = args.num_terms
elif N_max > 2e4:
    # Numerical precision breaks down at some point, so let's just choose an arbitrary cap
    raise RuntimeError(f"Too many terms to compute: {N_max}. Exiting...")

print(f"Computing {N_max} terms...")

# Compute the first n orders of the Riccati-Bessel functions using the recurrence relation
# in equation 10.16.1 found here https://dlmf.nist.gov/10.6

# Less efficient than using Scipy's Riccati-Bessel functions, but supports complex numbers
def riccati_bessel(max_order: int, z: complex):
    # scipy.special.spherical_jn([0, 1], z)
    # scipy.special.spherical_jn([0, 1], z, derivative=True)

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
psi_mx, psi_dmx, xi_mx, xi_dmx = riccati_bessel(N_max, mx)
denom_a = m * psi_mx * xi_dx - xi_x * psi_dmx
denom_b = psi_mx * xi_dx - m * xi_x * psi_dmx
a = (m * psi_mx * psi_dx - psi_x * psi_dmx) / denom_a
b = (psi_mx * psi_dx - m * psi_x * psi_dmx) / denom_b

# these produce identical scattering angle plots as the above
N = np.arange(0, N_max + 1)
j_x = scipy.special.spherical_jn(N, x)
j_x_dx = scipy.special.spherical_jn(N, x, derivative=True)
y_x = scipy.special.spherical_yn(N, x)
y_x_dx = scipy.special.spherical_yn(N, x, derivative=True)
h_x = j_x + 1j * y_x
h_x_dx = j_x_dx + 1j * y_x_dx
j_mx = scipy.special.spherical_jn(N, mx)
j_mx_dmx = scipy.special.spherical_jn(N, mx, derivative=True)
y_mx = scipy.special.spherical_yn(N, mx)
y_mx_dmx = scipy.special.spherical_yn(N, mx, derivative=True)
h_mx = j_mx + 1j * y_mx
h_mx_dmx = j_mx_dmx + 1j * y_mx_dmx
denom_a = mu_0 * m ** 2 * j_mx * (h_x + x * h_x_dx) - mu_1 * h_x * (j_mx + mx * j_mx_dmx)
denom_b = mu_1 * j_mx * (h_x + x * h_x_dx) - mu_0 * h_x * (j_mx + mx * j_mx_dmx)
# a = (mu_0 * m ** 2 * j_mx * (j_x + x * j_x_dx) - mu_1 * j_x * (j_mx + mx * j_mx_dmx)) / denom_a
# b = (mu_1 * j_mx * (j_x + x * j_x_dx) - mu_0 * j_x * (j_mx + mx * j_mx_dmx)) / denom_b
c = (mu_1 * j_x * (h_x + x * h_x_dx) - mu_1 * h_x * (j_x + x * j_x_dx)) / denom_b
d = (mu_1 * m * j_x * (h_x + x * h_x_dx) - mu_1 * m * h_x * (j_x + x * j_x_dx)) / denom_a

def get_intensity(vector_field: np.ndarray):
    # return |E|
    # return np.real(np.sqrt(np.einsum('...i,...i', vector_field, vector_field.conj())))

    # Take the real component of the complex representation, and compute the magnitude of the vector squared
    F = np.real(vector_field)
    return np.real(np.sqrt(np.einsum('...i,...i', F, F)))

# plot the electric field
grid_shape = [grid_height, grid_width // 2] # M x N for the right half of the grid
X, Z = np.meshgrid(
    np.linspace(0, scale / 2, grid_shape[1]), # only the right side, since left is identical
    np.linspace(scale / 2, -scale / 2, grid_shape[0]) # bottom and top
)
# from equations 4.18 and 4.19, with phi = 0, [ e_rho, e_theta ]
rhos = np.sqrt(X**2 + Z**2)
costheta = Z / rhos # cos(theta)
sintheta = np.sqrt(1 - costheta**2)

# associated Legendre polynomials of order 1 and their derivatives
P1, P1_dtheta = scipy.special.lpmn(1, N_max, costheta)
# extract order m = 1
P1 = P1[1]
P1_dtheta = P1_dtheta[1]

# compute Bessel & Hankel functions external to and internal to the sphere
# spherical Bessel function of the first kind (inputs specified below equations 4.50)
j_int = scipy.special.spherical_jn(N_max, k_1 * rhos)
j_int_dz = scipy.special.spherical_jn(N_max, k_1 * rhos, derivative=True)
# spherical Bessel function of the second kind
y_int = scipy.special.spherical_yn(N_max, k_1 * rhos)
y_int_dz = scipy.special.spherical_yn(N_max, k_1 * rhos, derivative=True)
# spherical Hankel function of the first kind
h_int = j_int + 1j * y_int
h_int_dz = j_int_dz + 1j * y_int_dz
# same for internal
j_ext = scipy.special.spherical_jn(N_max, k * rhos)
j_ext_dz = scipy.special.spherical_jn(N_max, k * rhos, derivative=True)
y_ext = scipy.special.spherical_yn(N_max, k * rhos)
y_ext_dz = scipy.special.spherical_yn(N_max, k * rhos, derivative=True)
h_ext = j_ext + 1j * y_ext
h_ext_dz = j_ext_dz + 1j * y_ext_dz

# compute the electric field internal and external to the drop,
# using equations 4.40 and 4.45 respectively
E_0 = 1 # amplitude of the incident plane wave
# allocate matrices for computing the electric field, in the same shape as the meshgrid,
# except with 3 complex numbers corresponding to e_rho, e_theta & e_phi basis vector components
shape = (*grid_shape, 3)
dtype = np.complex128
F_internal = np.zeros(shape, dtype=dtype)
F_external = np.zeros(shape, dtype=dtype)

if args.magnetic: # compute the magnetic field
    M_e1n_1 = np.zeros(shape, dtype=dtype)
    N_o1n_1 = np.zeros(shape, dtype=dtype)
    M_e1n_3 = np.zeros(shape, dtype=dtype)
    N_o1n_3 = np.zeros(shape, dtype=dtype)
else: # compute the electric field
    M_o1n_1 = np.zeros(shape, dtype=dtype)
    N_e1n_1 = np.zeros(shape, dtype=dtype)
    M_o1n_3 = np.zeros(shape, dtype=dtype)
    N_e1n_3 = np.zeros(shape, dtype=dtype)

# initial conditions
n = 1
pi_n_sub1 = 0
pi_n = 1
tau_n = costheta
while n <= N_max:
    # populate M and N vector components, with phi = 0 (equations 4.50)

    # TODO: make use of equations 4.88, and recurrence relations on page 87
    # TODO: figure out at which point the accuracy breaks down
    F_n = E_0 * 1j ** n * (2*n + 1) / (n * (n + 1))

    if args.magnetic: # compute the magnetic field
        # inside the sphere
        M_e1n_1[:, :, 2] = -tau_n * j_int[n]
        N_o1n_1[:, :, 2] = pi_n * (j_int[n] / (k_1 * rhos) + j_int_dz[n])
        F_internal += -k_1 / (omega * mu_1) * F_n * (d[n] * M_e1n_1 + 1j * c[n] * N_o1n_1)
        
        # outside the sphere
        M_e1n_3[:, :, 2] = -tau_n * j_ext[n]
        N_o1n_3[:, :, 2] = pi_n * (j_ext[n] / (k * rhos) + j_ext_dz[n])
        F_external += k / (omega * mu_0) * F_n * (1j * b[n] * N_o1n_3 + a[n] * M_e1n_3)
    else: # compute the electric field
        # inside the sphere
        M_o1n_1[:, :, 1] = pi_n * j_int[n]
        N_e1n_1[:, :, 0] = n * (n + 1) * sintheta * pi_n * j_int[n] / (k_1 * rhos)
        N_e1n_1[:, :, 1] = tau_n * (j_int[n] / (k_1 * rhos) + j_int_dz[n])
        F_internal += F_n * (c[n] * M_o1n_1 - 1j * d[n] * N_e1n_1) # equation 4.40
        
        # outside the sphere
        M_o1n_3[:, :, 1] = pi_n * h_ext[n]
        N_e1n_3[:, :, 0] = n * (n + 1) * sintheta * pi_n * h_ext[n] / (k * rhos)
        N_e1n_3[:, :, 1] = tau_n * (j_ext[n] / (k * rhos) + h_ext_dz[n])
        F_external += F_n * (1j * a[n] * N_e1n_3 - b[n] * M_o1n_3) # equation 4.45

    # compute the next pi & tau
    n += 1
    pi_next = (2*n - 1) / (n - 1) * costheta * pi_n - n / (n - 1) * pi_n_sub1
    pi_n_sub1 = pi_n
    pi_n = pi_next
    tau_n = n * costheta * pi_n - (n + 1) * pi_n_sub1

def to_cartesian(field: np.ndarray):
    out = field.copy()
    out[:, :, 0] = field[:, :, 0] * sintheta + field[:, :, 1] * costheta # x
    out[:, :, 1] = field[:, :, 2] # y
    out[:, :, 2] = field[:, :, 0] * costheta - field[:, :, 1] * sintheta # z
    return out

F_internal = to_cartesian(F_internal)
F_external = to_cartesian(F_external)

if args.show_incident:
    if args.magnetic:
        raise argparse.ArgumentError(arg_show_incident, "Showing the incident magnetic field is currently unsupported")
    # add the incident x-polarized plane wave to the external field (equation 4.21)
    E_incident = np.zeros_like(F_external)
    E_incident[:, :, 0] = E_0 * np.exp(1j * k * Z)
    F_external += E_incident * .1 # fudge factor due to normalization issues

internal_mask = rhos.repeat(3).reshape(shape) < rho
F_half = np.where(internal_mask, F_internal, F_external)

# each cell has its own basis vectors, but we can still compute the magnitude
I_half = get_intensity(F_half)
# add a mirrored copy to the left side
I = np.pad(I_half, ((0, 0), (I_half.shape[1], 0)), mode='reflect')

# plt.matshow(I)
# plt.title("Electric field intensity")
# plt.colorbar()

# plt.matshow(I, norm=LogNorm())
# plt.title("Electric field intensity log scale")
# plt.colorbar()

F = np.pad(F_half, ((0, 0), (F_half.shape[1], 0), (0, 0)), mode='reflect')

num_frames = args.fps * args.duration
intensities = np.zeros((num_frames, grid_height, grid_width))
num_waves = scale / wavelength
frame_step = 1 / num_frames
for i in range(0, num_frames):
    # Propagate the field forward in phase
    F *= np.exp(-2j * np.pi * frame_step)
    intensities[i] = get_intensity(F)

if args.log_scale:
    intensities = np.log(intensities + 1e-100)
    vmin, vmax = intensities.min(), intensities.max()
    intensities = (intensities - vmin) / (vmax - vmin)

sphere_boundary = None
sphere_boundary_thickness_squared = 100
if args.show_sphere_boundary:
    r = (rho / scale * grid_width) ** 2
    center_x = grid_width / 2
    center_z = grid_height / 2
    sphere_boundary = np.zeros((*intensities[0].shape, 4)) # rgba
    for z in range(0, sphere_boundary.shape[0]):
        for x in range(0, sphere_boundary.shape[1]):
            dist = np.abs((x - center_x) ** 2 + (z - center_z) ** 2 - r)
            if dist < sphere_boundary_thickness_squared:
                sphere_boundary[z, x, :3] = 1 # rgb
                sphere_boundary[z, x, 3] = 1 - dist / sphere_boundary_thickness_squared # alpha

images = []
for i, I in enumerate(intensities):
    c = cmap(I)

    # add sphere boundary with alpha blending
    if args.show_sphere_boundary:
        c[..., :3] = sphere_boundary[..., :3] * sphere_boundary[..., 3, np.newaxis] + (1 - sphere_boundary[..., 3, np.newaxis]) * c[..., :3]

    image = Image.fromarray(np.uint8(c * 255), mode="RGBA")
    image = image.convert('RGB') # remove alpha to improve color resolution
    draw = ImageDraw.Draw(image)

    if args.text:
        draw.text((0, 0), f"φ = {2 * i * frame_step:.2f} π", font=font)

    images.append(image)

images[0].save(
    fp=args.filename,
    format="GIF",
    append_images=images[1:],
    save_all=True,
    duration=1000 / args.fps, # ms per frame
    loop=0
)

plt.show()