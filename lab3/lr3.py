import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
import math
from scipy.integrate import odeint


def odesys(y, t, P, l, c, g, nu):
    dy = np.zeros(4)
    dy[0] = y[2]
    dy[1] = y[3]
    a11 = 1
    a12 = 0
    a21 = 0
    a22 = (l + y[0])
    b1 = -(nu * g / P) * y[2] - (c * g / P) * y[0] + (l + y[0]) * y[3] ** 2 + g * np.cos(y[1])
    b2 = -(y[2] * y[3]) - g * (l + y[0]) * np.sin(y[1])
    dy[2] = (b1 * a22 - b2 * a12) / (a11 * a22 - a12 * a21)
    dy[3] = (b2 * a11 - b1 * a21) / (a11 * a22 - a12 * a21)
    return dy


Steps = 1001
t_fin = 20
t = np.linspace(0, t_fin, Steps)

l_kernel = 20  # длина стержня
s_0 = 8  # длина O-О1
P_1 = 0.1  # вес стержня
P = 0.1  # вес колечка
l = 0.5  # длина недеформированной пружины
c = 200  # жесткость пружины
g = 9.81  # ускорение свободного падения
nu = 1  # трение
phi_0 = math.pi/10  # начальный угол
S_0 = 0  # начальная длина пружины
dphi_0 = 0
dS_0 = 0

y0 = [S_0, phi_0, dS_0, dphi_0]

Y = odeint(odesys, y0, t, (P, l, c, g, nu))

s = Y[:, 0]
phi = Y[:, 1]
ds = Y[:, 2]
dphi = Y[:, 3]
dds = np.array([odesys(y, t, P, l, c, g, nu)[2] for y, t in zip(Y, t)])
ddphi = np.array([odesys(y, t, P, l, c, g, nu)[3] for y, t in zip(Y, t)])

N = P * np.sin(phi) + (P / g) * ((l + s) * ddphi + 2 * dphi * ds)

M_x = s_0 * np.sin(phi) * s + np.sin(phi)*s_0
M_y = (-1) * s_0 * np.cos(phi) * s - s_0*np.cos(phi)

Kernel_x = l_kernel * np.sin(phi)
Kernel_y = (-1) * l_kernel * np.cos(phi)

SprX_0 = 4
K = 19
Sh = 0.3
b = 1 / (K - 2)
X_Spr = np.zeros(K)
Y_Spr = np.zeros(K)
X_Spr[0] = 0
Y_Spr[0] = 0
X_Spr[K - 1] = 0
Y_Spr[K - 1] = 1
for i in range(K - 2):
    Y_Spr[i + 1] = b * ((i + 1) - 1 / 2)
    X_Spr[i + 1] = Sh * (-1) ** i

L_Spr = (s_0 * np.cos(phi) * s * (-1)) - s_0 * np.cos(phi)
X_U = np.linspace(0, 1, K)

fig_for_graphs = plt.figure(figsize=[13, 7])
ax_for_graphs = fig_for_graphs.add_subplot(2, 2, 1)
ax_for_graphs.plot(t, s, color='Blue')
ax_for_graphs.set_title('s(t)')
ax_for_graphs.set(xlim=[0, t_fin])
ax_for_graphs.grid(True)

ax_for_graphs = fig_for_graphs.add_subplot(2, 2, 3)
ax_for_graphs.plot(t, phi, color='Red')
ax_for_graphs.set_title('phi(t)')
ax_for_graphs.set(xlim=[0, t_fin])
ax_for_graphs.grid(True)

ax_for_graphs = fig_for_graphs.add_subplot(2, 2, 2)
ax_for_graphs.plot(t, N, color='Orange')
ax_for_graphs.set_title('N')
ax_for_graphs.set(xlim=[0, t_fin])
ax_for_graphs.grid(True)

fig = plt.figure(figsize=[12, 7])
ax = fig.add_subplot(1, 1, 1)
ax.axis('equal')
ax.set(xlim=[-20, 20], ylim=[-30, 10])

Kernel = ax.plot([0, Kernel_x[0]], [0, Kernel_y[0]], linewidth=5, color="#7ED7C1")[0]
ax.plot([0, 0], [-20, 20], "k--")
ax.plot([-50, 50], [0, 0], "k--")
ax.plot(0, 0, marker='o')
ax.annotate("O", xy=(0, 0), xytext=(-1, -1.5))

ax.plot([-1, 0, 1], [1, 0, 1], "black")
ax.plot([-1, 1], [1, 1], "black")

Point_M = ax.plot(M_x[0], M_y[0], marker='o', color='r')[0]

Drawed_Spring = ax.plot(X_Spr + X_U * np.sin(phi)[0] * (s_0 * s[0])+np.sin(phi)[0]*s_0*X_U, Y_Spr * L_Spr[0])[0]


def anima(i):
    if s[i] < 0:
        tmp = -1
    else:
        tmp = 1

    Point_M.set_data([M_x[i] * tmp], [M_y[i]*tmp])
    Kernel.set_data([0, Kernel_x[i]], [0, Kernel_y[i]])

    Drawed_Spring.set_data((X_Spr + X_U * np.sin(phi)[i]*(s_0 * s[i])+np.sin(phi)[i]*s_0*X_U)*tmp, Y_Spr * L_Spr[i]*tmp)

    return [Point_M, Kernel, Drawed_Spring]


anim = FuncAnimation(fig, anima, frames=len(t), interval=40, repeat=False)

plt.show()
