import h5py
import numpy as np
from scipy.integrate import quad
from scipy.integrate import romberg 
from scipy.integrate import simps
import matplotlib
#matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.colors import hsv_to_rgb
import matplotlib.cm as cm


# some constants
l_s = 3.086e21 # length scale, centimeters in a kiloparsec
m_s = 1.99e33 # mass scale, g in a solar mass
t_s = 3.154e10 # time scale, seconds in a kyr
d_s = m_s / l_s**3 # density scale, M_sun / kpc^3
v_s = l_s / t_s # velocity scale, kpc / kyr
p_s = d_s*v_s**2 # pressure scale, M_sun / kpc kyr^2
G = 6.67259e-8 # in cm^3 g^-1 s^-2
mp = 1.67e-24 # proton mass in grams
G = G / l_s**3 * m_s * t_s**2 # in kpc^3 / M_sun / kyr^2
KB = 1.3806e-16 # boltzmann constant in cm^2 g / s^2 K
M_vir = 1e12
M_d = 6.5e10
M_h = M_vir - M_d
R_vir = 261 # MW viral radius in kpc
c_vir = 20 
R_h = R_vir / c_vir # halo scale radius in kpc
R_d = 3.5 # stellar disk scale length in kpc
Z_d = 3.5/5.0 # disk scale height in kpc
R_g = 2*R_d # gas disk scale length in kpc
T = 1e4 # gas temperature, 10^4 K
cs = np.sqrt(KB*T/(0.6*mp))*t_s/l_s # isothermal sound speed
v_to_kmps = l_s/t_s/100000
kmps_to_kpcpkyr = 1.0220122e-6
gamma = 1.001


phi_0_h = G * M_h / (np.log(1.0+c_vir) - c_vir / (1.0+c_vir))


N = 0 
dname='.'

for i in range(0,N+1):

  print(i)
  f = h5py.File(dname+'/hdf5/'+str(i)+'.h5', 'r')
  head = f.attrs
  nx = head['dims'][0]
  d  = np.array(f['density'])
  mx = np.array(f['momentum_x'])
  my = np.array(f['momentum_y'])
  mz = np.array(f['momentum_z'])
  E  = np.array(f['Energy'])
  vx = mx/d
  vy = my/d
  vz = mz/d
  p  = (E - 0.5*d*(vx*vx + vy*vy + vz*vz)) * (gamma - 1.0)
  e  = p/d/(gamma - 1.0)

  xmin, xmax = -10, 10
  dx = (xmax - xmin) / nx;
  x_pos = np.linspace(xmin+0.5*dx,xmax-0.5*dx,nx)

  r_disk = 13.9211647546
  #r_disk = 6.85009694274
  #r_disk = 0.220970869121
  r_halo = np.sqrt(x_pos*x_pos + r_disk*r_disk)
  x = r_halo / R_h

  # calculate acceleration due to NFW halo & Miyamoto-Nagai disk
  a_halo = - phi_0_h * (np.log(1+x) - x/(1+x)) / (r_halo*r_halo)
  a_disk_z = - G * M_d * x_pos * (R_d + np.sqrt(x_pos*x_pos + Z_d*Z_d)) / ( (r_disk*r_disk + (R_d + np.sqrt(x_pos*x_pos + Z_d*Z_d))**2)**1.5 * np.sqrt(x_pos*x_pos + Z_d*Z_d) )

  # total acceleration is the sum of the halo + disk components
  gx = (x_pos/r_halo)*a_halo + a_disk_z

  force = np.gradient(p)
  #force = cs**2 * np.gradient(d)

  fig = plt.figure(figsize=(4, 4), dpi=100)
  a0 = plt.axes([0.2,0.15,0.75,0.8])
  a0.plot(x_pos[0:nx//2], force[0:nx//2], 'ro')
  a0.plot(x_pos[0:nx//2], d[0:nx//2]*gx[0:nx//2], 'bo')
  plt.xlabel('z [kpc]')
  plt.xlim([-10,0])
#  plt.ylim([1e-10, 1e-1])
  a0.set_yscale('log')
#  a0.text(-1.8,1e-2,"r = 6.9")
#  a0.text(-1.8,1e-3,r'$\rho g_z$',color='blue')
#  a0.text(-1.8,1e-4,r'$\nabla p$',color='red')
  plt.savefig(dname+'/png/balance.png', dpi=100)
  plt.close(fig)
