import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
import math
from scipy.integrate import odeint

Steps = 1001
t_fin = 20
t = np.linspace(0, t_fin, Steps)

l_kernel = 20  # длина стержня
s_0 = 8  # длина O-О1
P_1 = 20  # вес стержня
P = 10  # вес колечка
l = 0.5  # длина недеформированной пружины
c = 20  # жесткость пружины
g = 9.81  # ускорение свободного падения
nu = 1  # трение
phi_0 = math.pi/10  # начальный угол
S_0 = 0  # начальная длина пружины
dphi_0 = 0
dS_0 = 0

s = np.sin(t)
phi = np.cos(t)

M_x = s_0 * np.sin(phi) * s
M_y = (-1) * s_0 * np.cos(phi) * s

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

L_Spr = (s_0 * np.cos(phi) * s * (-1))
X_U = np.linspace(0, 1, K)

fig = plt.figure(figsize=[12, 7])
ax = fig.add_subplot(1, 1, 1)
ax.axis('equal')
ax.set(xlim=[-20, 20], ylim=[-20, 5])

Kernel = ax.plot([0, Kernel_x[0]], [0, Kernel_y[0]], linewidth=5, color="#7ED7C1")[0]
ax.plot([0, 0], [-20, 20], "k--")
ax.plot([-50, 50], [0, 0], "k--")
ax.plot(0, 0, marker='o')
ax.annotate("O", xy=(0, 0), xytext=(-1, -1.5))

ax.plot([-1, 0, 1], [1, 0, 1], "black")
ax.plot([-1, 1], [1, 1], "black")

Point_M = ax.plot(M_x[0], M_y[0], marker='o', color='r')[0]

Drawed_Spring = ax.plot(X_Spr + X_U * np.sin(phi)[0] * (s_0 * s[0]), Y_Spr * L_Spr[0])[0]


def anima(i):
    if s[i] < 0:
        tmp = -1
    else:
        tmp = 1
    Point_M.set_data([M_x[i] * tmp], [M_y[i] * tmp])
    Kernel.set_data([0, Kernel_x[i]], [0, Kernel_y[i]])

    Drawed_Spring.set_data((X_Spr + X_U * np.sin(phi)[i] * (s_0 * s[i])) * tmp, Y_Spr * L_Spr[i]*tmp)

    return [Point_M, Kernel, Drawed_Spring]


anim = FuncAnimation(fig, anima, frames=len(t), interval=40, repeat=False)

plt.show()
