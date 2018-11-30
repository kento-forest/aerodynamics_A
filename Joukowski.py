import numpy as np
import matplotlib.pyplot as plt

# paramater
RADIUS_A = 0.6
ALPHA = np.radians(5)
BETA = np.radians(20)
U_0 = 1
C = 0.5

# Kutta condition
GAMMA = 4 * np.pi * U_0 * RADIUS_A * np.sin(ALPHA + BETA)

# cal potential
## complex
cU_0 = complex(U_0, 0.0)
cRADIUS_A = complex(RADIUS_A, 0.0)
cGAMMA = complex(0.0, GAMMA/(2*np.pi))
cCENT = complex(C, 0.0) + cRADIUS_A * np.exp(complex(0.0, np.pi - BETA))

## grid on Z-plane
mx = 3600
my = 1000
dr = 5/(my)
dtheta = np.radians(360/mx)
theta0 = -BETA

## prepare matrix
x = np.zeros((mx, my))
y = np.zeros((mx, my))
potential = np.zeros((mx, my))
Cp = np.zeros((mx, my))

# zeta-plane
xi = np.zeros((mx, my))
ita = np.zeros((mx, my))
streamfunc = np.zeros((mx, my))


## Start cal
for i in range(mx):
    for j in range(my):
        rrr = RADIUS_A + dr * float(j)
        theta = theta0 + dtheta *float(i)
        x[i, j] = rrr * np.cos(theta)
        y[i, j] = rrr * np.sin(theta)

        cZ = complex(x[i, j], y[i, j]) + cCENT
        cz = (cZ - cCENT) * np.exp(complex(0.0, -ALPHA))
        cf = cU_0 * (cz + cRADIUS_A**2 / cz) + cGAMMA * np.log(cz)
        potential[i, j]  = np.real(cf)
        streamfunc[i, j] = np.imag(cf)

        # zeta-plane
        zeta = cZ + complex(C**2, 0.0)/cZ
        xi[i, j]  = np.real(zeta)
        ita[i, j] = np.imag(zeta)

        # Cp
        cf = np.exp(complex(0.0, -ALPHA)) * (cU_0 * (complex(1.0, 0.0) - cRADIUS_A**2/cz**2) + cGAMMA/cz) \
                    / (complex(1.0, 0.0) - complex(C, 0.0)**2/cZ**2)
        Cp[i, j] = 1.0 - (np.real(cf)**2 + np.imag(cf)**2)/(U_0**2)

# Streamline
fig = plt.figure(figsize=(12, 10), dpi=200)
plt.xlim(-3, 3)
plt.ylim(-3, 3)
plt.contour(xi, ita, streamfunc, levels=[0.1 * x for x in range(-40, 40, 1)])
plt.plot(xi[:, 0], ita[:, 0], "r")
plt.xlabel(r"$\xi$")
plt.ylabel(r"$\eta$")
plt.colorbar()
plt.savefig("output/streamline.png")
plt.close()

# Cp
fig = plt.figure(figsize=(12, 10), dpi=200)
plt.xlim(-3, 3)
plt.ylim(-3, 3)
plt.contour(xi, ita, Cp, levels=[0.1 * x for x in range(-40, 40, 1)])
plt.plot(xi[:, 0], ita[:, 0], "r")
plt.xlabel(r"$\xi$")
plt.ylabel(r"$\eta$")
plt.colorbar()
plt.savefig("output/Cp.png")

# wall pressure
fig = plt.figure(figsize=(10, 12), dpi=200)
ax1 = fig.add_subplot(111)
ax2 = ax1.twinx()
ax1.set_xlim(-2, 2)
ax1.set_ylim(-2, 4)
ax1.set_xlabel(r"$\xi$")
ax1.set_ylabel(r"$\eta$")
ax1.plot(xi[:, 0], ita[:, 0], "r")
ax2.set_ylim(2, -4)
ax2.plot(xi[:, 0], Cp[:, 0], "b")
ax2.set_ylabel(r"$Cp$")
ax1.xaxis.set_major_locator(plt.MultipleLocator(1.0))
ax1.xaxis.set_minor_locator(plt.MultipleLocator(0.1))
ax1.yaxis.set_major_locator(plt.MultipleLocator(1.0))
ax1.yaxis.set_minor_locator(plt.MultipleLocator(0.1))
ax1.grid(which='major', axis='x', linewidth=0.75, linestyle='-', color='0.75')
ax1.grid(which='minor', axis='x', linewidth=0.25, linestyle='-', color='0.75')
ax1.grid(which='major', axis='y', linewidth=0.75, linestyle='-', color='0.75')
ax1.grid(which='minor', axis='y', linewidth=0.25, linestyle='-', color='0.75')
plt.savefig("output/wallpressure.png")
