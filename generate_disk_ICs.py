import h5py
import numpy as np
from scipy.integrate import quad
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
M_vir = 1.0e12
M_d = 6.5e10
M_h = M_vir - M_d
R_vir = 261.0 # MW viral radius in kpc
c_vir = 20.0 
R_h = R_vir / c_vir # halo scale radius in kpc
R_d = 3.5 # stellar disk scale length in kpc
Z_d = 3.5/5.0 # disk scale height in kpc
R_g = 2*R_d # gas disk scale length in kpc
T = 1.0e5 # gas temperature, 10^4 K
cs = np.sqrt(KB*T/(0.6*mp))*t_s/l_s # isothermal sound speed
v_to_kmps = l_s/t_s/100000
kmps_to_kpcpkyr = 1.0220122e-6
gamma = 1.001


dnamedata='/Volumes/data1/evan/data/disk_3D/128/hdf5/'
dname='./128'

# function for halo concentration, NFW profile
def func(xx):
  return np.log(1+xx) - xx / (1+xx)

i = 0

# set up the grid
nx = 128 
ny = 128 
nz = 128 
xmin, xmax = -20,20
ymin, ymax = -20,20
zmin, zmax = 0,10
dx = (xmax - xmin) / nx;
dy = (ymax - ymin) / ny;
dz = (zmax - zmin) / nz;
x = np.linspace(xmin+0.5*dx,xmax-0.5*dx,nx)
y = np.linspace(ymin+0.5*dy,ymax-0.5*dy,ny)
z = np.linspace(zmin+0.5*dz,zmax-0.5*dz,nz)
r = np.sqrt(x**2 + y**2)
x_pos, y_pos = np.meshgrid(x, y, indexing='ij')
r_pos = np.sqrt(x_pos**2 + y_pos**2)


# define the surface density distribution (exponential disk)
sigma = 0.25*M_d*np.exp(-r_pos/R_g) / (2*np.pi*R_g*R_g)
sigma_e = 0.25*M_d*np.exp(-r/R_g) / (2*np.pi*R_g*R_g)

# define phi_0_h
phi_0_h = G * M_h / (R_h * func(c_vir))
B_h = phi_0_h / cs**2 

# NFW halo potential with Phi_0 divided out
def phi_halo(z, r, r_h):
  return -np.log(1 + np.sqrt(r**2 + z**2)/r_h)/(np.sqrt(r**2 + z**2)/r_h)

# define phi_0_d
phi_0_d = G * M_d / R_d
B_d = phi_0_d / cs**2 

# Miyamoto-Nagai disk with Phi_0 divided out
def phi_disk(z, r, r_d, z_d):
  return - 1.0 / np.sqrt((r/r_d)**2 + (1 + np.sqrt((z/r_d)**2 + (z_d/r_d)**2))**2)

# set the constant in the exponential argument
# this is an array with a unique value for each r_pos
A_h = phi_halo(0, r_pos, R_h)
A_d = phi_disk(0, r_pos, R_d, Z_d)

def gr_halo(z, r, G, M_h, R_h):
  return 

def gr_disk(z, r, G, M_d, R_d, z_d):
  return G * M_d * r / (r**2 + (R_d + np.sqrt(z*z + z_d*z_d))**2)**(1.5)


def integrand_h(z, r, A, B, r_h):
  return np.exp(B * (A + np.log(1+(np.sqrt(r**2 + z**2)/r_h))/(np.sqrt(r**2 + z**2)/r_h)))
def integrand_d(z, r, A, B, r_d, z_d):
  return np.exp(B * (A + 1.0 / np.sqrt((r/r_d)**2 + (1 + np.sqrt((z/r_d)**2 + (z_d/r_d)**2))**2)))
def integrand(z, r, A_h, A_d, B_h, B_d, r_h, r_d, z_d):
  return np.exp(B_h*(A_h + np.log(1+(np.sqrt(r**2 + z**2)/r_h))/(np.sqrt(r**2 + z**2)/r_h)) + B_d*(A_d + 1.0 / np.sqrt((r/r_d)**2 + (1 + np.sqrt((z/r_d)**2 + (z_d/r_d)**2))**2)))


sol = np.ones_like(r_pos)
for ii in range(0, nx):
  for jj in range(0, ny):
    #sol[ii,jj],err = quad(integrand_h, 0, 10, args=(r_pos[ii,jj],A_h[ii,jj],B_h,R_h))
    sol[ii,jj],err = quad(integrand_d, 0, 10, args=(r_pos[ii,jj],A_d[ii,jj],B_d,R_d,Z_d))
    #sol[ii,jj],err = quad(integrand, 0, 10, args=(r_pos[ii,jj],A_h[ii,jj],A_d[ii,jj],B_h,B_d,R_h,R_d,Z_d))


rho_0 = 0.5 * sigma / sol
d_temp = np.empty([nx, ny, 2*nz])

# set densities (symmetric about z=0)
for kk in range(0, nz):
  #d_temp[:,:,nz+kk] = d_temp[:,:,nz-kk-1] = rho_0*np.exp(B_h*(A_h - phi_halo(z[kk],r_pos,R_h)))
  d_temp[:,:,nz+kk] = d_temp[:,:,nz-kk-1] = rho_0*np.exp(B_d*(A_d - phi_disk(z[kk],r_pos,R_d,Z_d)))
  #d_temp[:,:,nz+kk] = d_temp[:,:,nz-kk-1] = rho_0*np.exp(B_h*(A_h - phi_halo(z[kk],r_pos,R_h)) + B_d*(A_d - phi_disk(z[kk],r_pos,R_d,Z_d)))

d = np.empty([nx, ny, nz/2])
v_cell = dx*dy*dz
M = d_temp*v_cell
dz_new = 4*dz
v_cell_new = dx*dy*dz_new
for kk in range(0, 2*nz, 4):
  d[:,:,kk/4] = (M[:,:,kk] + M[:,:,kk+1] + M[:,:,kk+2] + M[:,:,kk+3])/v_cell_new

# now redefine everything in terms of the new z array
nz = nz/2
dz = dz_new
z = np.linspace(-zmax+0.5*dz,zmax-0.5*dz,nz)

"""
xi=64
yi=64
zz = np.linspace(0,dz,1001)
#plt.plot(zz, integrand_h(zz, r_pos[xi,yi], A_h[xi,yi], B_h, R_h))
#plt.plot(zz, integrand_d(zz, r_pos[xi,yi], A_d[xi,yi], B_d, R_d, Z_d))
#plt.plot(zz, np.exp(B_h*(A_h[xi,yi] - phi_halo(zz,r_pos[xi,yi],R_h))))
#plt.plot(zz, np.exp(B_d*(A_d[xi,yi] - phi_disk(zz,r_pos[xi,yi],R_d,Z_d))))
plt.plot(zz, np.exp(B_h*(A_h[xi,yi] - phi_halo(zz,r_pos[xi,yi],R_h)) + B_d*(A_d[xi,yi] - phi_disk(zz,r_pos[xi,yi],R_d,Z_d))))
plt.yscale('log')
plt.show()
"""
# don't let densities get too low
lowd = np.where(d < 1.0)
d[lowd] = 1.0


# calculate pressure derivs
P = d*(cs**2)/gamma
dPdx = np.gradient(P, dx, axis=0)
dPdy = np.gradient(P, dy, axis=1)
#dPdx[:,:,:] = 0.0
#dPdy[:,:,:] = 0.0
#for kk in range(0, nz):
#  plt.plot(dPdy[64,:,kk])
#  plt.ylim(-0.02, 0.02)
#  plt.text(2,0.0008,str(kk))
#  plt.show()

mx = np.empty([nx, ny, nz])
my = np.empty([nx, ny, nz])
mz = np.empty([nx, ny, nz])
E = np.empty([nx, ny, nz])
vr = np.empty([nx, ny, nz])
for kk in range(0, nz):
  for jj in range(0, ny):
    for ii in range(0, nx):
      xp = x[ii]
      yp = y[jj]
      zp = z[kk]
      rp = r_pos[ii,jj] 
      dp = d[ii,jj,kk]
      #calculate radial acceleration due to potential
      a_d = gr_disk(zp, rp, G, M_d, R_d, Z_d)
      #calculate radial pressure gradient
      dPdr = xp*dPdx[ii,jj,kk]/rp + yp*dPdy[ii,jj,kk]/rp
      #circular velocity from combined radial acceleration and pressure gradient
      a = a_d + dPdr/dp
      v = np.sqrt(rp*a)
      vr[ii,jj,kk] = v
      #vx = -np.sin(phi)*v
      #vy = np.cos(phi)*v
      vx = -(yp/rp)*v
      vy = (xp/rp)*v
      #vx = 0.0
      #vy = 0.0
      vz = 0.0
      mx[ii,jj,kk] = dp*vx
      my[ii,jj,kk] = dp*vy
      mz[ii,jj,kk] = dp*vz
      E[ii,jj,kk] = P[ii,jj,kk]/(gamma-1.0) + 0.5*dp*(vx*vx + vy*vy)

print(np.max(vr)*v_to_kmps)

pdz = np.sum(d*dz, axis=2)
pdy = np.sum(d*dy, axis=1)
log_pdz = np.log10(pdz)
log_pdy = np.log10(pdy)

#plt.plot(r_pos, pdz, 'ro')
#plt.plot(r, sigma_e)
#plt.xlim([0,30])
#plt.ylim([1e5,1e7])
#plt.yscale('log')
#plt.show()


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
#V = np.clip(V, 0.01, 0.99)
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
#V = np.clip(V, 0.01, 0.99)
S = 1.0 - V

HSV = np.dstack((H,S,V))
RGB = hsv_to_rgb(HSV)
xz = RGB

fig = plt.figure(figsize=(4,6), dpi=100)
a0 = plt.axes([0.,0.333,1.,0.667])
for child in a0.get_children():
  if isinstance(child, matplotlib.spines.Spine):
    child.set_visible(False)  
a0.set_xticks(400*np.arange(0, 1, 0.125))
a0.set_yticks(400*np.arange(0, 1, 0.125)+200)
a0.tick_params(axis='both', which='both', color='white', length=5, direction='in', top='on', right='on', labelleft='off', labelbottom='off')  
a0.imshow(xy, origin='lower', extent=(0, 400, 201, 600), interpolation='bilinear')
a0.autoscale(False)
a0.hlines(250, 250, 300, color='white')
a0.text(305, 245, '5 kpc', color='white')
a1 = plt.axes([0.,0.,1.,0.333])
for child in a1.get_children():
  if isinstance(child, matplotlib.spines.Spine):
    child.set_visible(False)  
a1.set_xticks(400*np.arange(0, 1, 0.125))
a1.set_yticks(200*np.arange(0, 1, 0.25))
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
for zi in range(0, nz):
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

# write the array to an hdf5 file
f = h5py.File(dnamedata+'0.h5.0', 'w')
f.attrs['gamma'] = gamma
f.attrs['t'] = 0.0
f.attrs['dt'] = 0.1
f.attrs['n_step'] = 0
f.create_dataset('density', data=d)
f.create_dataset('momentum_x', data=mx)
f.create_dataset('momentum_y', data=my)
f.create_dataset('momentum_z', data=mz)
f.create_dataset('Energy', data=E)
f.close()
