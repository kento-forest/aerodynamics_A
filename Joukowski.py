import numpy as np
import matplotlib.pyplot as plt

# paramater
RADIUS_A = 0.6
ALPHA = np.radians(5.0)
BETA = np.radians(20.0)
U_0 = 1.0
C = 0.5
DENSITY = 1.0

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

# zeta-plane
xi = np.zeros((mx, my))
eta = np.zeros((mx, my))
streamfunc = np.zeros((mx, my))

# grid prepare
i_mat = np.array([[i for j in range(my)] for i in range(mx)])
j_mat = np.array([[j for j in range(my)] for i in range(mx)])

rrr = RADIUS_A + dr * j_mat
theta = theta0 + dtheta * i_mat
x = rrr * np.cos(theta)
y = rrr * np.sin(theta)

# x and y matrix ==> x+iy matrix
cxy = np.array([complex(xi, yi) for (xi, yi) in zip(np.ravel(x), np.ravel(y))])
cxy = np.reshape(cxy, x.shape)

cZ = cxy + cCENT
cz = (cZ - cCENT) * np.exp(complex(0.0, -ALPHA))
cf = cU_0 * (cz + cRADIUS_A**2 / cz) + cGAMMA * np.log(cz)
potential = np.real(cf)  # potential
streamfunc = np.imag(cf)  # stream function

# zeta-plane
zeta = cZ + complex(C**2, 0.0)/cZ
xi = np.real(zeta)
eta = np.imag(zeta)

# Cp
cf = np.exp(complex(0.0, -ALPHA)) * (cU_0 * (complex(1.0, 0.0) - cRADIUS_A**2/cz**2) + cGAMMA/cz) \
            / (complex(1.0, 0.0) - complex(C, 0.0)**2/cZ**2)
Cp = 1.0 - (np.real(cf)**2 + np.imag(cf)**2)/(U_0**2)

# cal aeroforce on wall
cx_p = 0.0
cy_p = 0.0
for i in range(mx-1):
    d_xi  = xi[i+1, 0] - xi[i, 0]
    d_eta = eta[i+1, 0] - eta[i, 0]
    dnx = d_eta
    dny = -d_xi
    cp_ave = (Cp[i+1, 0] + Cp[i, 0])/2.0
    cx_p = cx_p-cp_ave*dnx
    cy_p = cy_p-cp_ave*dny

cx_p = cx_p / (4.0 * C)
cy_p = cy_p / (4.0 * C)
cdp = cx_p * np.cos(ALPHA) + cy_p * np.sin(ALPHA)
clp = cy_p * np.cos(ALPHA) - cx_p * np.sin(ALPHA)


front_edge = min(xi[:, 0])
back_edge = max(xi[:, 0])

# aerodynamic center
fx = np.array([(-(eta[(i+1), 0] - eta[i, 0]) * (Cp[i, 0] + Cp[(i+1), 0]) / 2) for i in range(eta.shape[0]-1)])
fy = np.array([((xi[(i+1), 0] - xi[i, 0]) * (Cp[i, 0] + Cp[(i+1), 0]) / 2) for i in range(eta.shape[0]-1)])

Xcp = np.sum([xi[i][0] * fy[i] - eta[i][0] * fx[i] for i in range(eta.shape[0]-1)]) / np.sum([fy[i] for i in range(eta.shape[0]-1)])
Xcp_ratio = (Xcp - front_edge) / (back_edge - front_edge)

# Cm
Cm = np.array([(np.sum(2*((xi[j, 0] - xi[i, 0]) * fy[i] + eta[i, 0] * fx[i]) for i in range(eta.shape[0]-1))
                / (DENSITY * U_0** 2 * (back_edge - front_edge)**2)) for j in range(eta.shape[0]-1)])


with open('output/aeroforce.txt', 'w') as f:
    f.write('CL = '+'{0:.3f}'.format(clp)+'\n')
    f.write('CD = '+'{0:.3f}'.format(cdp)+'\n')
    f.write('aerodynamic center = '+"{0:.3f}".format(Xcp_ratio)+' chord length')

print('CL = '+'{0:.3f}'.format(clp))
print('CD = '+'{0:.3f}'.format(cdp))
print('aerodynamic center = '+'{0:.3f}'.format(Xcp_ratio)+' chord length')


# Streamline
fig = plt.figure(figsize=(12, 10), dpi=200)
plt.xlim(-3, 3)
plt.ylim(-3, 3)
plt.contour(xi, eta, streamfunc, levels=[0.1 * x for x in range(-40, 40, 1)])
plt.plot(xi[:, 0], eta[:, 0], "r")
plt.xlabel(r"$\xi$")
plt.ylabel(r"$\eta$")
plt.colorbar()
plt.savefig("output/streamline.png", bbox_inches='tight')
plt.close()

# Cp
fig = plt.figure(figsize=(12, 10), dpi=200)
plt.xlim(-3, 3)
plt.ylim(-3, 3)
plt.contour(xi, eta, Cp, levels=[0.1 * x for x in range(-40, 40, 1)])
plt.plot(xi[:, 0], eta[:, 0], "r")
plt.xlabel(r"$\xi$")
plt.ylabel(r"$\eta$")
plt.colorbar()
plt.savefig("output/Cp.png", bbox_inches='tight')

# Cp distribution
fig = plt.figure(figsize=(10, 12), dpi=200)
ax1 = fig.add_subplot(111)
ax2 = ax1.twinx()
ax1.set_xlim(-2, 2)
ax1.set_ylim(-2, 4)
ax1.set_xlabel(r"$\xi$")
ax1.set_ylabel(r"$\eta$")
ax1.plot(xi[:, 0], eta[:, 0], "r")
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
plt.savefig("output/wallpressure.png", bbox_inches='tight')

# Cm
fig = plt.figure(figsize=(10, 10), dpi=200)
plt.xlim(-1, 1)
plt.plot(xi[:-1, 0], Cm, lw = 1)
plt.xlabel(r"$\xi$")
plt.ylabel(r"$C_m$")
plt.savefig('output/Cm.png', bbox_inches='tight')
