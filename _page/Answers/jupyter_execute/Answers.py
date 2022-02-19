#!/usr/bin/env python
# coding: utf-8

# # Question 1 - single particle trajectory in magnetic field

# 1. Plot the trajectory of a deuterium ion with arbitrary initial velocity in a uniform magnetic field $\vec{B} = B \hat{z}$ with $B = 3\mathrm{T}$. Assume $\vec{E}=0$.

# In[1]:


# If using Jupyter notebook instead of VS code, change the line below to '%matplotlib notebook'
get_ipython().run_line_magic('matplotlib', 'widget')


# In[2]:


import numpy as np

el_charge = 1.6e-19
el_mass = 9.10938356e-31

# Update particle position and velocity
def particle_pusher(x_old,v_old,q,m,dt,E,B):
    
    x_new = x_old + v_old*dt
    
    B_mag = np.linalg.norm(B)
    t = np.tan(q * B_mag * dt * 0.5 / m) * B / B_mag
    s = 2.0 * t / (1.0 + (np.linalg.norm(t) ** 2))
    v_minus = v_old + q * E * dt * 0.5 / m
    v_prime = v_minus + np.cross(v_minus,t)
    v_plus = v_minus + np.cross(v_prime,s)
    
    v_new = v_plus + q * E * dt * 0.5 / m
    
    return x_new, v_new

# Launch a particle with an initial position and velocity x_0 and v_0, traversing uniform E and B fields
def launch_particle(q, m, v_0, x_0, E_field, B_field, timestep, num_timesteps):
  trajectory = np.zeros([num_timesteps,3])
  velocities = np.zeros([num_timesteps,3])
  trajectory[0,:] = x_0
  velocities[0,:] = v_0

  x = x_0
  v = v_0
  for i in range(1,num_timesteps):

      x_new, v_new = particle_pusher(x,v,q,m,timestep,E_field,B_field)
      x = x_new
      v = v_new
      
      trajectory[i,:] = x
      velocities[i,:] = v
  
  return trajectory, velocities


# In[3]:


# Define a few constants
electron_charge = -1.6e-19
amu = 1.66053906660e-27

# Define the deuterium ion charge and mass
ion_charge = -electron_charge
ion_mass = 2 * amu 

# Initialise position, velocity, fields and timesteps
v_0 = 1e5*np.array([0.0,1.0,1.0])
x_0 = np.array([0.5,0.5,0.0])
E = np.array([0.0,0.0,0.0])
B = np.array([0.0,0.0,3.0])
dt = 1e-9
N_t = 100

ion_trajectory, ion_velocities = launch_particle(ion_charge,ion_mass,v_0,x_0,E,B,dt,N_t)


# In[4]:


import matplotlib.pyplot as plt

# Define a function to plot a particle trajectory in 3D
def plot_trajectory3d(trajectory):
  # Create figure
  fig = plt.figure()
  ax = plt.axes(projection='3d')
  ax.set_xlabel('x [m]'); ax.set_ylabel('y [m]'); ax.set_zlabel('z [m]')
      
  # Plot the 3D trajectory of the particle
  ax.plot(trajectory[:,0],trajectory[:,1],trajectory[:,2])


# In[5]:


plot_trajectory3d(ion_trajectory)


# 2. Check that kinetic energy and magnetic moment of the particle are conserved.

# In[6]:


# Compute kinetic energy at each timestep
kin_en = 0.5 * ion_mass * (ion_velocities[:,0] ** 2 + ion_velocities[:,1] ** 2 + ion_velocities[:,2] ** 2 )
timestamps = np.linspace(0,dt*N_t,N_t)

# Plot kinetic energy over time
fig,ax = plt.subplots(1)
ax.plot(timestamps,kin_en)
ax.set_xlabel('Time [s]')
ax.set_ylabel('Kinetic energy [J]')


# In[7]:


# Calculate the magnetic moment for an array of particle velocities over time, with mass m, assuming constant B-field
def compute_magnetic_moments(velocities,B,m):
  mu = np.zeros(velocities.shape[0])

  # To compute magnetic moment at each timestep, first extract perpendicular component of velocity
  for i,v in enumerate(velocities):
    v_par = np.dot(B,v) * (B / np.linalg.norm(B)**2)
    v_perp = v - v_par  

    # Compute the magnetic moment
    mu[i] = 0.5 * m * np.linalg.norm(v_perp)**2 / np.linalg.norm(B)
  
  return mu

# Compute the magnetic moments and plot over time
magnetic_moments = compute_magnetic_moments(ion_velocities,B,ion_mass)
fig,ax = plt.subplots(1)
ax.plot(timestamps,magnetic_moments)
ax.set_xlabel('Time [s]')
ax.set_ylabel('Magnetic moment [Am$^2$]')


# 3. Compare the deuterium ion trajectory to an electron orbiting in the same magnetic field. Calculate their Larmor radii. 

# In[8]:


# Calculate the trajectory for an electron
el_charge = -1.6e-19
el_mass = 9.10938356e-31
dt = 2.5e-13
el_trajectory, el_velocities = launch_particle(el_charge,el_mass,v_0,x_0,E,B,dt,N_t)

plot_trajectory3d(el_trajectory)


# In[9]:


# Calculate the radius of both particles using min/max x positions (could equally use y positions in this example)
ion_xmin = np.min(ion_trajectory[:,0])
ion_xmax = np.max(ion_trajectory[:,0])
ion_lrad = 0.5 * (ion_xmax - ion_xmin)

el_xmin = np.min(el_trajectory[:,0])
el_xmax = np.max(el_trajectory[:,0])
el_lrad = 0.5 * (el_xmax - el_xmin)

print('Ion Larmor radius: {:1.2f}'.format(ion_lrad*1e3) + 'mm')
print('Electron Larmor radius: {:1.2f}'.format(el_lrad*1e6) + 'μm')


# In[10]:


# Compare to analytical value
def calculate_larmor_radius(q,B,m,v):
  v_par = np.dot(B,v) * (B / np.linalg.norm(B)**2)
  v_perp = v - v_par

  return m * np.linalg.norm(v_perp) /(q * np.linalg.norm(B))

ion_lrad_th = calculate_larmor_radius(abs(el_charge), B, ion_mass, ion_velocities[0])
el_lrad_th = calculate_larmor_radius(abs(el_charge), B, el_mass, el_velocities[0])

print('Ion Larmor radius (analytical): {:1.2f}'.format(ion_lrad_th*1e3) + 'mm')
print('Electron Larmor radius (analytical): {:1.2f}'.format(el_lrad_th*1e6) + 'μm')


# # Question 2 - drifts

# 1. Using the code you wrote for the first question, now plot the trajectory of a deuterium ion in the presence of a uniform magnetic field $\vec{B} = B\hat{z}$ and electric field $\vec{E} = E\hat{y}$ (experiment with different values of $E$ and $B$). Calculate the expected $\vec{E} \times \vec{B}$ drift velocity $\vec{v}_d$, and use this to plot the guiding centre alongside the particle trajectory you have computed (remembering $\vec{v}_{gc} = \vec{v}_{\parallel} + \vec{v}_d$). Does your ion's drift match speed and direction to that expected?

# In[11]:


# Define new E and B fields and plot trajectory of the ion
v_0 = 1e5*np.array([0.0,1.0,1.0])
x_0 = np.array([0.5,0.5,0.0])
E = np.array([0.0,10000.0,0.0])
B = np.array([0.0,0.0,1.0])
dt = 5e-9
N_t = 100

ion_trajectory, ion_velocities = launch_particle(ion_charge,ion_mass,v_0,x_0,E,B,dt,N_t)

plot_trajectory3d(ion_trajectory)


# In[12]:



# Compute the ExB drift and guiding centre velocity
v0 = ion_velocities[0]
v_par0 = np.dot(B,v0) * (B / np.linalg.norm(B)**2)
v_d0 = np.cross(E,B) / np.linalg.norm(B) ** 2
v_gc = v_par0 + v_d0

# Compute the guiding centre trajectory, making an educated guess for the initial 
# guiding centre position (this could be done more rigorously!)
gc_trajectory = np.zeros(ion_trajectory.shape)
gc_trajectory[0,:] = np.array([0.502,0.5005,0.0])
for i in range(N_t):
  gc_trajectory[i,:] = gc_trajectory[0,:] + i * dt * v_gc

# Plot both trajectories together
fig = plt.figure()
ax = plt.axes(projection='3d')
ax.set_xlabel('x [m]'); ax.set_ylabel('y [m]'); ax.set_zlabel('z [m]')
ax.plot(ion_trajectory[:,0],ion_trajectory[:,1],ion_trajectory[:,2],label='Full trajectory')
ax.plot(gc_trajectory[:,0],gc_trajectory[:,1],gc_trajectory[:,2],label='Guiding centre')
ax.legend()


# 2. For magnetic field gradients perpendicular to the field lines, the drift velocity is $\vec{v}_g = \frac{\frac{1}{2}mv_{\perp}^2}{q} \frac{\vec{B} \times \nabla B}{B^3}$. Set up a non-uniform magnetic field of the form $\vec{B} = B(y)\hat{z}$, where $B(y) = B_0 + Cy$, with $C$ being some constant which defines the magnetic field gradient. Compare the trajectory of your particle with the predicted guiding centre drift. 

# In[13]:


# Modify the 'launch_particle' function to accept fields which vary with position
def launch_particle(q, m, v_0, x_0, E_field, B_field, timestep, num_timesteps):
  trajectory = np.zeros([num_timesteps,3])
  velocities = np.zeros([num_timesteps,3])
  trajectory[0,:] = x_0
  velocities[0,:] = v_0

  x = x_0
  v = v_0
  for i in range(1,num_timesteps):
      # E_func and B_func can now be functions
      if callable(E_field):
        E = E_field(x)
      else:
        E = E_field
      if callable(B_field):
        B = B_field(x)
      else:
        B = B_field 
        
      x_new, v_new = particle_pusher(x,v,q,m,timestep,E,B)
      x = x_new
      v = v_new
      
      trajectory[i,:] = x
      velocities[i,:] = v
  
  return trajectory, velocities


# In[14]:


# Define a B-field which varies with position
E = np.array([0.0,0.0,0.0])
B_0 = 1.0; C = 10.0
def get_B(x):
  B_z = B_0 + C*(x[1] - 0.49)
  return np.array([0.0,0.0,B_z])

# Initial conditions and setup
v_0 = 1e5*np.array([0.0,1.0,1.0])
x_0 = np.array([0.5,0.5,0.0])
dt = 5e-10
N_t = 1000

# Compute trajectory
ion_trajectory, ion_velocities = launch_particle(ion_charge,ion_mass,v_0,x_0,E,get_B,dt,N_t)


# In[15]:


# Compute the grad-B drift velocity and guiding centre trajectory
gc_trajectory = np.zeros(ion_trajectory.shape)
gc_trajectory[0,:] = np.array([0.502,0.5,0.0])
for i in range(N_t):
  v = ion_velocities[i]
  B = get_B(ion_trajectory[i,:])
  v_par = np.dot(B,v) * (B / np.linalg.norm(B)**2)
  v_perp = v - v_par
  grad_B = np.array([0,C,0])
  v_d = 0.5 * ion_mass * np.linalg.norm(v_perp) ** 2 * np.cross(B,grad_B) / ( abs(el_charge) * np.linalg.norm(B) ** 3 ) 
  v_gc = v_par + v_d

  gc_trajectory[i,:] = gc_trajectory[0,:] + i * dt * v_gc

# Plot both trajectories together
fig = plt.figure()
ax = plt.axes(projection='3d')
ax.set_xlabel('x [m]'); ax.set_ylabel('y [m]'); ax.set_zlabel('z [m]')
ax.plot(ion_trajectory[:,0],ion_trajectory[:,1],ion_trajectory[:,2],label='Full trajectory')
ax.plot(gc_trajectory[:,0],gc_trajectory[:,1],gc_trajectory[:,2],label='Guiding centre')
ax.legend()


# # Question 3 - magnetic mirror

# 1. As shown in the lectures, an expression for $B_r$ can be obtained by writing $\nabla \cdot \vec{B} = 0$ in cylindrical coordinates. This depends on $\frac{\partial B_z}{\partial z}|_{r_0}$, which we can assume is constant along $z$. Write a function which returns all components of $\vec{B}$ for a given position, assuming $B_z$ increases from some $B_{min}$ at $z=0$. 

# In[16]:


# Define B-field parameters
B_min_z = 1.0
grad_B_r0 = 15.0

# Define a function which takes position x and return mirror device B components in cartesian coordinates
def get_mirror_B(x):
  r = np.sqrt(x[0]**2 + x[1]**2)
  z = x[2]
  theta = np.arctan2(x[1], x[0])
  B_z = B_min_z + grad_B_r0 * abs(z)
  B_r = -0.5 * r * grad_B_r0
  return np.array([B_r * np.cos(theta), B_r * np.sin(theta), B_z])


# 2. Setting $B_{min}=1\mathrm{T}$ and $\frac{\partial B_z}{\partial z}|_{r_0}=15\mathrm{Tm}^{-1}$, launch a test particle (deuterium ion) from the origin, orbiting about the z-axis, and see if you can observe the particle reverse direction in the z-axis. You may wish to use to following initial conditions
#     - $\vec{v}_0 = 15\times 10^{4} (\hat{x} + \hat{y} + \hat{z})\mathrm{ms}^{-1}$ for the initial velocity
#     - $\vec{r}_0 = (-0.003\hat{x} + 0.003\hat{y})\mathrm{m}$ for the initial position
#     - $\vec{E} = 0$
#     - timesteps of around 1 nanosecond

# In[17]:


# First, define a function to plot the particle trajectory in 2D
def plot_trajectory2d(trajectory,axes=[0,2]):
  # Create figure
  fig,ax = plt.subplots(1)
  axis_labels = ['x [m]','y [m]','z [m]']
  ax.set_xlabel(axis_labels[axes[0]]); ax.set_ylabel(axis_labels[axes[1]])
      
  # Plot the 3D trajectory of the particle
  ax.plot(trajectory[:,axes[0]],trajectory[:,axes[1]])


# In[18]:


# Now launch a test particle to see if the mirror effect can be achieved
v_0 = 1.5e5*np.array([1.0,1.0,1.0])
x_0 = np.array([-0.003,0.003,0.0])
E = np.array([0.0,0.0,0.0])
dt = 1e-9
N_t = 750

ion_trajectory, ion_velocities = launch_particle(ion_charge,ion_mass,v_0,x_0,E,get_mirror_B,dt,N_t)

plot_trajectory2d(ion_trajectory,axes=[2,0])


# # Question 4 (optional extension) - magnetic mirror continued

# 1. Write a new function which launches a particle into such a device. Now, we are not worried about storing the full trajectory - we should just track the particle until either a) the particle has passed some point $z_0$ on the z-axis (this may be the position of a magnetic coil of the device) or b) the particle's z-component of velocity has reversed direction. The function should return information on whether the particle escaped (condition a) or was trapped (condition b).

# In[19]:


# Write a new launch_particle function for checking particle confinement in a mirror device
def launch_particle_mirror(q, m, v_0, x_0, B_field, timestep, z_0):
  
  E = np.array([0.0,0.0,0.0])
  x = x_0
  v = v_0

  escaped = False
  mirrored = False
  max_steps = 1000000
  step = 0
  while step < max_steps:
      B = B_field(x)
      x_new, v_new = particle_pusher(x,v,q,m,timestep,E,B)
      x = x_new
      v = v_new
      

      # print(x[2])
      if x[2] > z_0:
        escaped = True 
        break
      elif v[2] < 0:
        mirrored = True
        break
      
      step +=1 
        
  if step >= max_steps and (escaped is False and mirrored is False):
    print('Max steps reached while launching particle.')
    
  
  return escaped, mirrored


# 2. Let us imagine that the magnetic coils for this device are placed at $\pm z_0$. We will continue to consider only half of the device in the positive z domain, checking for particles which can escape past the coil placed at $z_{0}=1\mathrm{m}$ . Modify the B-field so that $\frac{\partial B_z}{\partial z}|_{r_0}=1\mathrm{Tm}^{-1}$ (keeping $B_{min}=1\mathrm{T}$). Now, vary the balance of perpendicular and parallel velocity in your particle's initial condition, keeping the magnitude $v_0$ the same, $v_0=2.5\times 10^5\mathrm{ms}^{-1}$. Launch a number of particles from the origin, and make a record of which ones are trapped and which ones escape (as well as their initial $v_{\perp,0}$ and $v_{\parallel,0}$).

# In[20]:


# Modify the B-field
grad_B_r0 = 1.0
B_min = 1.0

# Launch an array of particles with differing fractions of v_perp and v_parallel
x_0 = np.array([0.0,0.0,0.0])
v_mag = 2.5e5
E = np.array([0.0,0.0,0.0])
num_pcles = 50
z_0 = 1.0
dt = 1e-9
v_perps = np.zeros(num_pcles)
v_pars = np.zeros(num_pcles)
pcl_escaped = np.zeros(num_pcles)

for i in range(num_pcles):
  
  eta = np.random.random() # Eta is the squared ratio of perpendicular to total velocity

  v_perp0 = v_mag*np.sqrt(eta)
  v_par0 = v_mag*np.sqrt(1.0 - eta)
  v_0 = np.array([v_perp0,0.0,v_par0])

  escaped, mirrored = launch_particle_mirror(ion_charge,ion_mass,v_0,x_0,get_mirror_B,dt,z_0)
  
  # Track which particle escape/are trapped
  if escaped:
    pcl_escaped[i] = 1

  # Store initial parallel and perpendicular velocities
  v_perps[i] = v_perp0
  v_pars[i] = v_par0


# 3. Produce a plot of the particles in the $v_{\parallel,0},v_{\perp,0}$ plane, using their initial values, showing which ones are trapped and which ones escape. Estimate (by eye is sufficient) the cutoff value for $v_{\parallel,0}^2/v_{\perp,0}^2$ which determines whether a particle is trapped. Does this match the formula derived in the lectures, $v_{\parallel,0}^2/v_{\perp,0}^2 < (B_{z_0}/B_{min} - 1)$, where $B_{z_0}$ is the B-field strength at $z_0$?

# In[21]:


# Plot the v_perps and v_pars as well as whether or not particles are trapped
fig,ax = plt.subplots(1)
ax.plot(v_perps[np.where(pcl_escaped == 0)], v_pars[np.where(pcl_escaped == 0)], 'x', label='Trapped')
ax.plot(v_perps[np.where(pcl_escaped == 1)], v_pars[np.where(pcl_escaped == 1)], 'x', label='Escaped')
ax.set_xlabel('$v_{\perp,0}$ [ms$^{-1}$]')
ax.set_ylabel('$v_{\parallel,0}$ [ms$^{-1}$]')
ax.legend()


# In[22]:


# Estimate v_perp and v_par at the cutoff between trapped/escaped
v_perp_cutoff = 1.82e5
v_par_cutoff = 1.73e5
cutoff_ratio_sim = v_perp_cutoff / v_par_cutoff

# Calculate predicted value
B_min = np.linalg.norm(get_mirror_B([0,0,0]))
B_z0 = np.linalg.norm(get_mirror_B([0,0,z_0]))
cutoff_ratio_th = np.sqrt(B_z0/B_min - 1)

print('Cutoff ratio from simulation = {:1.2f}'.format(cutoff_ratio_sim) )
print('Theoretical cutoff ratio = {:1.2f}'.format(cutoff_ratio_th) )


# 
