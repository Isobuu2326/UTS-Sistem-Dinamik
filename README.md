import numpy as np
import matplotlib.pyplot as plt

# Fungsi turunan untuk metode numerik
def f1(t, y, z):
    return z

def f2(t, y, z):
    return -4 * z - 5 * y

# Solusi analitik
def analytic_solution(t):
    return np.exp(t / 2) * (np.sin(t) + 7 * np.cos(t))

# Inisialisasi variabel
h = 0.1
t_values = np.arange(0, 2.1, h)
n = len(t_values)

# Inisialisasi array
y_euler = np.zeros(n)
z_euler = np.zeros(n)

y_rk4 = np.zeros(n)
z_rk4 = np.zeros(n)

y_exact = analytic_solution(t_values)

# Kondisi awal
y_euler[0] = y_rk4[0] = 5
z_euler[0] = z_rk4[0] = 10

# Metode Euler
for i in range(n - 1):
    y_euler[i + 1] = y_euler[i] + h * f1(t_values[i], y_euler[i], z_euler[i])
    z_euler[i + 1] = z_euler[i] + h * f2(t_values[i], y_euler[i], z_euler[i])

# Metode Runge-Kutta Orde 4
for i in range(n - 1):
    t, y, z = t_values[i], y_rk4[i], z_rk4[i]
    
    k1 = h * f1(t, y, z)
    l1 = h * f2(t, y, z)
    
    k2 = h * f1(t + h/2, y + k1/2, z + l1/2)
    l2 = h * f2(t + h/2, y + k1/2, z + l1/2)
    
    k3 = h * f1(t + h/2, y + k2/2, z + l2/2)
    l3 = h * f2(t + h/2, y + k2/2, z + l2/2)
    
    k4 = h * f1(t + h, y + k3, z + l3)
    l4 = h * f2(t + h, y + k3, z + l3)
    
    y_rk4[i + 1] = y + (k1 + 2*k2 + 2*k3 + k4) / 6
    z_rk4[i + 1] = z + (l1 + 2*l2 + 2*l3 + l4) / 6

# Plot hasil
plt.figure(figsize=(10, 6))
plt.plot(t_values, y_exact, label='Analitik', linestyle='-', marker='o')
plt.plot(t_values, y_euler, label='Euler', linestyle='--', marker='x')
plt.plot(t_values, y_rk4, label='Runge-Kutta Orde 4', linestyle='-.', marker='s')
plt.title('Perbandingan Solusi y(t)')
plt.xlabel('Waktu (t)')
plt.ylabel('y(t)')
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.show()
