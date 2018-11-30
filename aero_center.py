import numpy as np
import matplotlib.pyplot as plt
import matplotlib.cm as cm

# paramater
RADIUS_A = 0.6
ALPHA_list = [-10, -6, -2, 0, 2, 5, 8, 12, 15]
BETA = np.radians(20)
U_0 = 1
C = 0.5
DENSITY = 1.0

def cal_cm(alpha):
    alpha = np.radians(alpha)
    GAMMA = 4 * np.pi * U_0 * RADIUS_A * np.sin(alpha + BETA)
    # cal potential
    ## complex
    cU_0 = complex(U_0, 0.0)
    cRADIUS_A = complex(RADIUS_A, 0.0)
    cGAMMA = complex(0.0, GAMMA/(2*np.pi))
    cCENT = complex(C, 0.0) + cRADIUS_A * np.exp(complex(0.0, np.pi - BETA))

    ## grid on Z-plane
    mx = 360
    my = 100
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
    cz = (cZ - cCENT) * np.exp(complex(0.0, -alpha))
    cf = cU_0 * (cz + cRADIUS_A**2 / cz) + cGAMMA * np.log(cz)
    potential = np.real(cf)  # potential
    streamfunc = np.imag(cf)  # stream function

    # zeta-plane
    zeta = cZ + complex(C**2, 0.0)/cZ
    xi = np.real(zeta)
    eta = np.imag(zeta)

    # Cp
    cf = np.exp(complex(0.0, -alpha)) * (cU_0 * (complex(1.0, 0.0) - cRADIUS_A**2/cz**2) + cGAMMA/cz) \
        / (complex(1.0, 0.0) - complex(C, 0.0)**2/cZ**2)
    Cp = 1.0 - (np.real(cf)**2 + np.imag(cf)**2)/(U_0**2)

    # cal aeroforce on wall
    cx_p = 0.0
    cy_p = 0.0
    for i in range(mx-1):
        d_xi = xi[i+1, 0] - xi[i, 0]
        d_eta = eta[i+1, 0] - eta[i, 0]
        dnx = d_eta
        dny = -d_xi
        cp_ave = (Cp[i+1, 0] + Cp[i, 0])/2.0
        cx_p = cx_p-cp_ave*dnx
        cy_p = cy_p-cp_ave*dny

    cx_p = cx_p / (4.0 * C)
    cy_p = cy_p / (4.0 * C)
    cdp = cx_p * np.cos(alpha) + cy_p * np.sin(alpha)
    clp = cy_p * np.cos(alpha) - cx_p * np.sin(alpha)


    front_edge = min(xi[:, 0])
    back_edge = max(xi[:, 0])

    # aerodynamic center
    fx = np.array([(-(eta[(i+1), 0] - eta[i, 0]) * (Cp[i, 0] + Cp[(i+1), 0]) / 2)
                for i in range(eta.shape[0]-1)])
    fy = np.array([((xi[(i+1), 0] - xi[i, 0]) * (Cp[i, 0] + Cp[(i+1), 0]) / 2)
                for i in range(eta.shape[0]-1)])

    Xcp = np.sum([xi[i][0] * fy[i] - eta[i][0] * fx[i]
                for i in range(eta.shape[0]-1)]) / np.sum([fy[i] for i in range(eta.shape[0]-1)])
    Xcp_ratio = (Xcp - front_edge) / (back_edge - front_edge)

    # Cm
    Cm = np.array([(np.sum(2*((xi[j, 0] - xi[i, 0]) * fy[i] + eta[i, 0] * fx[i]) for i in range(eta.shape[0]-1))
                    / (DENSITY * U_0 ** 2 * (back_edge - front_edge)**2)) for j in range(eta.shape[0]-1)])
    
    return xi, Cm

# Cm plot change alpha
count = 0
color_list = ['b', 'g', 'r', 'c', 'm', 'y', 'k', '', 'pink']
fig = plt.figure(figsize=(10, 10), dpi=200)
plt.xlim(-1, 1)
for alpha in ALPHA_list:
    xi, Cm = cal_cm(alpha)
    plt.plot(xi[:-1, 0], Cm, lw=0.5, c=cm.hsv(count/len(ALPHA_list)))
    count += 1
plt.savefig('output/cm_alpha.png', bbox_inches='tight')
plt.close()