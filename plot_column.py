import h5py
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from scipy import ndimage
from matplotlib.colors import hsv_to_rgb
import matplotlib.cm as cm


# some constants
l_s = 3.086e21 # length scale, centimeters in a kiloparsec
m_s = 1.99e33 # mass scale, g in a solar mass
t_s = 3.154e10 # time scale, seconds in a kyr
G = 6.67259e-8 # in cm^3 g^-1 s^-2
G = G / l_s**3 * m_s * t_s**2 # in kpc^3 / M_sun / kyr^2
M_vir = 1e12
M_d = 6.5e10
M_b = 1e10
M_h = M_vir - M_d #- M_b
R_vir = 261 # MW viral radius in kpc
c_vir = 20 
r_s = R_vir / c_vir # halo scale radius in kpc
R_d = 3.5 # disk scale length in kpc
R_b = 0.7 # bulge scale length in kpc
Sigma_0 = 8.636e2
v_to_kmps = l_s/t_s/100000
kmps_to_kpcpkyr = 1.0220122e-6


N = 0 
dname='.'

for i in range(0,N+1):

  print(i)
  f = h5py.File(dname+'/hdf5/'+str(i)+'.h5', 'r')
  head = f.attrs
  gamma = head['gamma'][0]
  t = head['t']
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
  x = np.linspace(xmin+0.5*dx,xmax-0.5*dx,nx)


  fig = plt.figure(figsize=(4, 4), dpi=100)
  a0 = plt.axes([0.2,0.15,0.75,0.8])
  a0.plot(x, d, 'r.')
  plt.xlabel('z [kpc]')
  plt.ylabel(r'$\rho$ [$M_{\odot}/kpc^3$]')
  plt.ylim([0.1,1e8])
  a0.set_yscale('log')
  a0.text(5,1e7,"r = 13.9")
  plt.savefig(dname+'/png/'+str(i)+'.png', dpi=100)
  plt.close(fig)

