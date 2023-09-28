import numpy as np
import numpy.fft as fft
import random

# Parameters Setting
Nx, Ny = 256, 256  # Spatial resolution
Nt, Dt = 3200, 32  # Temporal resolution
dt = 0.01 / 32  # Time step size
Nsample = 1000  # Number of samples

# Spatial Discretization Parameters
dx, dy = 2 * np.pi / Nx, 2 * np.pi / Ny  
X, Y = np.meshgrid(np.linspace(0, 2 * np.pi, Nx, endpoint=False), np.linspace(0, 2 * np.pi, Ny, endpoint=False))

# Function to generate the initial conditions using Gaussian random fields
def init_condition():
    N = Nx  
    U0_hat = np.random.normal(0, 1, (N, N)) + 1j * np.random.normal(0, 1, (N, N))
    kx, ky = fft.fftfreq(N, d=1./N), fft.fftfreq(N, d=1./N)
    kx, ky = np.meshgrid(kx, ky)
    k_sq = kx**2 + ky**2
    power_spectrum = np.where(k_sq != 0, (k_sq + 30)**-3, 0)
    U0_hat *= np.sqrt(power_spectrum)
    U0 = fft.ifft2(U0_hat).real
    data_new = 0.1 + ((U0 - U0.min()) / (U0.max() - U0.min())) * 1.0
    return data_new

# Function to compute f
def f(U):
    u, v = U[0], U[1]
    return np.array([np.sin(v) * np.cos(5 * X + 5 * Y), np.sin(u) * np.cos(5 * X - 5 * Y)])

# Central Difference Methods
def diff_x_center(U):
    return (-np.roll(U, -2, axis=0) + 8*np.roll(U,-1,axis=0)- 8*np.roll(U,1,axis=0) + np.roll(U,2,axis=0)) / (12*dx)

def diff_y_center(U):
    return (-np.roll(U,-2,axis=1) + 8*np.roll(U,-1,axis=1) - 8*np.roll(U,1,axis=1) + np.roll(U,2,axis=1)) / (12*dy)

# Upwind Difference Methods
def diff_x_upwind(U):
    return np.where(U > 0, 
                    (3*U - 4*np.roll(U, 1, axis=0) + np.roll(U, 2, axis=0)) / (2*dx),
                    (-np.roll(U, -2, axis=0) + 4*np.roll(U, -1, axis=0) - 3*U) / (2*dx))

def diff_y_upwind(U):
    return np.where(U > 0, 
                    (3*U - 4*np.roll(U, 1, axis=1) + np.roll(U, 2, axis=1)) / (2*dy),
                    (-np.roll(U, -2, axis=1) + 4*np.roll(U, -1, axis=1) - 3*U) / (2*dy))

# Function for handling spatial differentiation
def handle_spatial_diff(U, v_new):
    Ux_upwind, Uy_upwind = diff_x_upwind(U[0]), diff_y_upwind(U[1])
    Ux_center, Uy_center = diff_x_center(U[0]), diff_y_center(U[1])
    return -U * np.array([Ux_upwind, Uy_upwind]) + v_new * np.array([diff_x_center(Ux_center), diff_y_center(Uy_center)]) + f(U)

# Arrays for storing results
res1 = np.zeros((Nsample, Nt//Dt+1, 2, Nx, Ny))
v_all1 = np.zeros((Nsample, 1))

# Loop for generating data samples
for s in range(Nsample):
    print("Starting %d th / %d"%(s+1, Nsample))
    U0, V0 = init_condition(), init_condition()
    U = np.stack([U0, V0])
    res1[s, 0] = U
    v = random.uniform(0.001, 0.1)
    v_all1[s, 0] = v
    step = 0
    v_new = v
    print("###############")
    for t in range(Nt):
        # 4th Order Runge-Kutta Method
        k1 = handle_spatial_diff(U,v_new)
        U_temp = U + np.roll(k1 * dt / 2, shift=(1,1), axis=(1,2))
        k2 = handle_spatial_diff(U_temp,v_new)
        U_temp = U + np.roll(k2 * dt / 2, shift=(1,1), axis=(1,2))
        k3 = handle_spatial_diff(U_temp,v_new)
        U_temp = U + np.roll(k3 * dt, shift=(1,1), axis=(1,2))
        k4 = handle_spatial_diff(U_temp,v_new)
        U += dt / 6 * (k1 + 2*k2 + 2*k3 + k4)
        if t % Dt == 0:
            res1[s, 1 + t//Dt] = U
            step += 1
            if step % 20 == 0 or step == 1:
                print("Step: %d/%d"%(t/Dt +1, Nt//Dt))
    print("###############")
