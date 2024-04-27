import numpy as np
import healpy as hp 
from healpy.newvisufunc import projview, newprojplot
from healpy.visufunc import projscatter
from healpy.visufunc import projplot

import matplotlib.pyplot as plt
from matplotlib import cm 
from matplotlib.colors import ListedColormap, LinearSegmentedColormap

green = 0.2 
red = 0.2
blue = 0.4

N = 20
vals = np.ones((N, 4))
vals[:, 0] = red
vals[:, 1] = green
vals[:, 2] = blue
NEWCMP = ListedColormap(vals[::-1])

def plot_skymap(moon_zeniths, moon_azimuths):
    
    moon_theta = np.pi - np.radians(moon_zeniths)
    moon_phi = (np.radians(moon_azimuths) - np.pi)

    n = 25
    cmap = np.arange(len(moon_theta))

    pix0 = hp.ang2pix(n, moon_theta, moon_phi)
    m = np.ones(hp.nside2npix(n))

    hp.projview(
        m, coord = ["C"], title="The Moon's Trajectory", 
        cmap=NEWCMP, cbar=False, min=0, max=4,
        graticule=True, graticule_labels=True, alpha=0.5
    )
    plt.scatter(moon_phi, moon_theta-np.pi/2, c=np.arange(len(moon_theta)), cmap="binary")
    # plt.scatter(moon_phi, moon_theta-np.pi/2, c=dates-15000, cmap="binary")
    plt.colorbar(orientation="horizontal", label="days", pad=0.05) #need to put dates in the correct format
    plt.show()


# <!-- Just to get a sense, let's plot the moon's position in the sky at many different times and see how much it moves. How much does it move in a month? Can you figure out how much does it move in a year? (You will need to modify `start_date` and `end_date`)!
# We can do this for an observer located at the South Pole, or one here in Cambridge (just remember to update `observer_lat` and `observer_long`)! -->

# from src.plot_skymap import plot_skymap

# # change me if you want to look at a different range of dates/times!
# start_date = "2023-01-01"
# end_date = "2023-02-01"
# dates = np.arange(
#     start_date, end_date, dtype='datetime64[D]')

# # change me if you want to change the observer's position!
# observer_lat = icecube_lat
# observer_long = icecube_long

# moon_zeniths = []
# moon_azimuths = []

# # Now we iterate over all the dates, and calculate the moon position for each.
# # At each step, we will `append` the zenith and azimuth to our lists
# for date in dates:
#     azi, zen = get_moon_position_at( 
#         str(date),
#         observer_lat,
#         observer_long
#     )

#     moon_zeniths.append(zen)
#     moon_azimuths.append(azi)


# plot_skymap(moon_zeniths, moon_azimuths)