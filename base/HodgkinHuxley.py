import matplotlib.pyplot as plt
import numpy as np

# Constants
C_m = 1.0  # membrane capacitance (uF/cm^2)
g_Na = 120.0  # maximum sodium conductance (mS/cm^2)
g_K = 36.0  # maximum potassium conductance (mS/cm^2)
g_L = 0.3  # leak conductance (mS/cm^2)
E_Na = 50.0  # sodium reversal potential (mV)
E_K = -77.0  # potassium reversal potential (mV)
E_L = -54.387  # leak reversal potential (mV)

# Time parameters
dt = 0.01  # time step (ms)
t = np.arange(0, 50, dt)  # time vector (ms)

# Stimulus
I = 10.0  # applied current (uA/cm^2)
stim_start = 5.0  # start time of current injection (ms)
stim_end = 45.0  # end time of current injection (ms)
stim_duration = stim_end - stim_start

# Initialize voltage and gating variables
V = np.zeros(len(t))  # membrane potential (mV)
m = np.zeros(len(t))  # activation variable for sodium channels
h = np.zeros(len(t))  # inactivation variable for sodium channels
n = np.zeros(len(t))  # activation variable for potassium channels

# Set initial conditions
V[0] = -65.0  # resting membrane potential (mV)
m[0] = 0.05  # initial value for m
h[0] = 0.6  # initial value for h
n[0] = 0.32  # initial value for n

# Helper functions for gating variables' alpha and beta functions


def alpha_m(v):
    return 0.1 * (v + 40.0) / (1.0 - np.exp(-0.1 * (v + 40.0)))


def beta_m(v):
    return 4.0 * np.exp(-0.0556 * (v + 65.0))


def alpha_h(v):
    return 0.07 * np.exp(-0.05 * (v + 65.0))


def beta_h(v):
    return 1.0 / (1.0 + np.exp(-0.1 * (v + 35.0)))


def alpha_n(v):
    return 0.01 * (v + 55.0) / (1.0 - np.exp(-0.1 * (v + 55.0)))


def beta_n(v):
    return 0.125 * np.exp(-0.0125 * (v + 65.0))


# Simulation loop
for i in range(1, len(t)):
    # Calculate sodium and potassium channel conductances
    gNa = g_Na * m[i-1]**3 * h[i-1]
    gK = g_K * n[i-1]**4
    gL = g_L

    # Calculate membrane currents
    INa = gNa * (V[i-1] - E_Na)
    IK = gK * (V[i-1] - E_K)
    IL = gL * (V[i-1] - E_L)

    # Calculate total membrane current
    if stim_start <= t[i] <= stim_end:
        I_inj = I  # applied current during stimulus
    else:
        I_inj = 0.0  # no applied current outside stimulus

    # Update voltage and gating variables using Euler method
    V[i] = V[i-1] + (1.0 / C_m) * (I_inj - INa - IK - IL) * dt
    m[i] = m[i-1] + alpha_m(V[i-1]) * (1.0 - m[i-1]) * \
        dt - beta_m(V[i-1]) * m[i-1] * dt
    h[i] = h[i-1] + alpha_h(V[i-1]) * (1.0 - h[i-1]) * \
        dt - beta_h(V[i-1]) * h[i-1] * dt
    n[i] = n[i-1] + alpha_n(V[i-1]) * (1.0 - n[i-1]) * \
        dt - beta_n(V[i-1]) * n[i-1] * dt


# Plotting the results
plt.figure()
plt.plot(t, V, label='Membrane Potential (mV)')
plt.xlabel('Time (ms)')
plt.ylabel('Membrane Potential (mV)')
plt.title('Hodgkin-Huxley Model')
plt.legend()
plt.show()
