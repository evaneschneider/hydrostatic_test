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
M_b = 1e10
M_h = M_vir - M_d #- M_b
R_vir = 261 # MW viral radius in kpc
c_vir = 20 
R_h = R_vir / c_vir # halo scale radius in kpc
R_d = 3.5 # stellar disk scale length in kpc
z_d = 3.5/5.0 # disk scale height in kpc
R_g = 2*R_d # gas disk scale length in kpc
T = 1e4 # gas temperature, 10^4 K
v_to_kmps = l_s/t_s/100000
kmps_to_kpcpkyr = 1.0220122e-6
cs = np.sqrt(KB*T/(0.6*mp))*t_s/l_s
gamma = 1.001

N = 0
dname='./128'

for i in range(0,N+1):
  
  print(i)
  f = h5py.File(dname+'/hdf5/'+str(i)+'.h5', 'r')
  head = f.attrs
  gamma = head['gamma'][0]
  t = head['t']
  nx = head['dims'][0]
  ny = head['dims'][1]
  nz = head['dims'][2]
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
  
  dx = 40 / nx
  dy = 40 / ny
  dz = 20 / nz

  pdz = np.sum(d*dz, axis=2)
  pdy = np.sum(d*dy, axis=1)
  log_pdz = np.log10(pdz)
  log_pdy = np.log10(pdy)


  #make color image
  cmin=100/360.
  cmax=300/360.

  #xy projection
  #set the density scale
  min_pd = np.min(pdz) 
  max_pd = np.max(pdz) 
  cpd = (np.clip(pdz, min_pd, max_pd) - min_pd + 0.01) / (max_pd-min_pd)
  H = (cmin-cmax)*cpd + cmax
  H = H.T # gotta transpose to get the plot right

  #set the projected density scale
  rho_pd_min = np.min(log_pdz) 
  rho_pd_max = 1.1*np.max(log_pdz) 

  #produce baseline density plot
  rpd = (log_pdz - rho_pd_min)/(rho_pd_max - rho_pd_min)
  V = rpd
  V = V.T

  #eliminate accidental negative values
  V = np.clip(V, 0.01, 0.99)
  S = 1.0 - V

  HSV = np.dstack((H,S,V))
  RGB = hsv_to_rgb(HSV)
  xy = RGB

  #xz projection
  #set the density scale
  min_pd = np.min(pdy) 
  max_pd = np.max(pdy) 
  cpd = (np.clip(pdy, min_pd, max_pd) - min_pd + 0.01) / (max_pd-min_pd)
  H = (cmin-cmax)*cpd + cmax
  H = H.T

  #set the projected density scale
  rho_pd_min = np.min(log_pdy) 
  rho_pd_max = 1.1*np.max(log_pdy)

  #produce baseline density plot
  rpd = (log_pdy - rho_pd_min)/(rho_pd_max - rho_pd_min)
  V = rpd
  V = V.T

  #eliminate accidental negative values
  V = np.clip(V, 0.01, 0.99)
  S = 1.0 - V

  HSV = np.dstack((H,S,V))
  RGB = hsv_to_rgb(HSV)
  xz = RGB

  fig = plt.figure(figsize=(4,6), dpi=100)
  a0 = plt.axes([0.,0.333,1.,0.667])
  for child in a0.get_children():
    if isinstance(child, matplotlib.spines.Spine):
      child.set_visible(False)  
  a0.set_xticks(400*np.arange(0.1, 1, 0.1))
  a0.set_yticks(400*np.arange(0.1, 1, 0.1)+200)
  a0.tick_params(axis='both', which='both', color='white', length=5, direction='in', top='on', right='on', labelleft='off', labelbottom='off')  
  a0.imshow(xy, origin='lower', extent=(0, 400, 201, 600), interpolation='bilinear')
  a0.autoscale(False)
  a0.hlines(240, 280, 320, color='white')
  a0.text(325, 235, '5 kpc', color='white')
  a1 = plt.axes([0.,0.,1.,0.333])
  for child in a1.get_children():
    if isinstance(child, matplotlib.spines.Spine):
      child.set_visible(False)  
  a1.set_xticks(400*np.arange(0.1, 1, 0.1))
  a1.set_yticks(200*np.arange(0.2, 1, 0.2))
  a1.tick_params(axis='both', which='both', color='white', length=5, direction='in', top='on', right='on', labelleft='off', labelbottom='off')  
  a1.imshow(xz, origin='lower', extent=(0, 400, 0, 200), interpolation='bilinear')
  #pretty white border
  a0.axvline(x=0, color='white')
  a1.axvline(x=0, color='white')
  a0.axvline(x=399, color='white')
  a1.axvline(x=399, color='white')
  a1.hlines(200, 0, 400, color='white')
  a1.hlines(1, 0, 400, color='white')
  plt.savefig(dname+'/png/'+str(i)+'.png', dpi=300)
  plt.close(fig)
"""
  for zi in range(nz):
    fig = plt.figure(figsize=(8,4), dpi=100)
    a0 = plt.axes([0,0,0.5,1])
    vx_plot = mx[:,:,zi]/d[:,:,zi]
    vy_plot = my[:,:,zi]/d[:,:,zi]
    plt.imshow(vx_plot.T, origin='lower', interpolation='none')
    plt.axis('off')
    a0.text(3, 35, r'$v_x$', color='white')
    a1 = plt.axes([0.5,0,0.5,1])
    plt.imshow(vy_plot.T, origin='lower', interpolation='none')
    plt.axis('off')
    a1.text(3, 35, '$v_y$', color='white')
    plt.savefig(dname+'/png/v_'+str(i)+str(zi)+'.png', dpi=300)
    plt.close(fig)
"""

