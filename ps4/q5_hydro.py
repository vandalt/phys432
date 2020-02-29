"""
1-D Hydro solver for a sound wave in uniform density gas without gravity.
The IC is a small Gaussian perturbation in density and/or velocity (see below).

This solver is partially adapted from the Heidelberg lectures, which I used to
understand in more detail the different parts of the assignment.
(http://www.ita.uni-heidelberg.de/~dullemond/lectures/num_fluid_2011/)

Author: Thomas Vandal
Feb 27th 2020
"""
import numpy as np
from matplotlib import pyplot as plt


def hydrosol(x, f1, f2, dx, dt, cs):
    """Single step Hydrodynamics
    Function performing one step of donnor cell advection for a Hydro solver.
    The function acts directly on the array and returns nothing.

    Args:
        x    (array): cell centers
        f1   (array): f1 (mass density) at cell centers
        f2   (array): f2 (momentum density) at cell centers
        dx   (float): grid spacing
        dt   (float): timestep size
        cs   (array): isothermal (constant) sound speed
    """
    npts = x.size

    # Compute velocity at interfaces (needed for both fluxes)
    ui = np.zeros(npts+1)  # one more interface if count the two boundaries
    ui[1:-1] = 0.5 * (f2[1:]/f1[1:] + f2[:-1]/f1[:-1])

    # Compute the f1 flux
    ff1 = np.zeros(npts+1)
    ff1[1:-1] = np.where(ui[1:-1] > 0, f1[:-1], f1[1:]) * ui[1:-1]

    # Calculate f1
    f1[:] -= dt/dx * (ff1[1:] - ff1[:-1])

    # Compute the f2 flux
    ff2 = np.zeros(npts+1)
    ff2[1:-1] = np.where(ui[1:-1] > 0, f2[:-1], f2[1:]) * ui[1:-1]

    # Calculate f2
    f2[:] -= dt/dx * (ff2[1:] - ff2[:-1])

    # Add the source (pressure) term for f2, except at boundary cells
    f2[1:-1] -= dt/dx * cs**2 * (f1[2:] - f1[:-2])

    # Enforce reflective BC using source term
    # (as explained in Heidelberg lectures)
    f2[0] = f2[0] - 0.5 * dt/dx * cs**2 * (f1[1] - f1[0])
    f2[-1] = f2[-1] - 0.5 * dt/dx * cs**2 * (f1[-1] - f1[-2])


# Setting up the problem
npts = 1000                 # number grid cells
nsteps = 4000               # number of timesteps
xmin, xmax = 0.0, 1000.0    # min and max position values on grid
dt = 0.25                   # initial timestep (adjusted with speed)
cfl = 0.5                   # CFL factor used to control timestep
x, dx = np.linspace(xmin, xmax, num=npts, retstep=True)  # centers and spacing
f1, f2 = np.zeros(npts), np.zeros(npts)  # initialize f1, f2
cs = 50  # some constant factor playing role of sound speed

# Setting Gaussian Perturbation in otherwise constant to 1 f1
mu = 0.5 * (xmax + xmin)
s = 0.1 * (xmax - xmin)
a = 0.1
f1 = 1.0 + a * np.exp(-(x-mu)**2/s**2)

# Setting up figure
plt.ion()
fig = plt.figure(figsize=[6.4, 4.8])
p1, = plt.plot(x, f1)
plt.title(r'Motion of a sound wave')
plt.xlabel(r'Position $x$')
plt.ylabel(r'Density $f1$')
fig.canvas.draw()

# Do hydro steps
print('Starting hydro solver...')
for i in range(nsteps):
    print('Completed {} steps.'.format(i), end='\r')
    tmp = dx/(cs+np.abs(f2/f1))
    dt = cfl*np.min(tmp)  # timestep from velocity and dx

    # Call hydro function
    hydrosol(x, f1, f2, dx, dt, cs)

    # Update plots
    if not i % 10:
        p1.set_ydata(f1)
        fig.canvas.draw()
        plt.pause(0.001)

print('Completed {} steps.'.format(nsteps))
print('Hydro solver done.')
