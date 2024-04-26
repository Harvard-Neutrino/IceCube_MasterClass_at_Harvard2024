
import numpy as np

"""
bound the azimuth angle between [-pi, pi]
"""
def bound_azi( azi ): 
    if azi < -np.pi:    return bound_azi( azi + 2*np.pi ) 
    elif azi > np.pi:   return bound_azi( azi - 2*np.pi )
    else:               return azi

"""
bound the zenith angle between [0, pi]
"""
def bound_zen( zen ):
    if zen < 0:         return bound_zen( abs(zen) )
    elif zen > np.pi:   return bound_zen( zen - 2*np.pi )
    else:               return zen

def normalize(x): return x / np.linalg.norm(x)

# spherical coordinate transformations for directions:

def get_direction_vector_from_angles( azi, zen ):
    return np.array([
        np.cos(azi) * np.sin(zen),
        np.sin(azi) * np.sin(zen),
        np.cos(zen)
    ])

def get_direction_angles_from_vector( dir ):
    azi = np.arctan2( dir[1], dir[0] )
    zen = np.arccos( dir[2] )
    return np.array([azi, zen])


def calc_zenith_diff( zen_1, zen_2 ):
  return np.abs( zen_1 - zen_2 )

def calc_azimuth_diff( azi_1, azi_2 ):
  # this is a little trickier, because 0° and 360° (or, in radians, 0 and 2pi) mean the same thing!
  # we need to remember that the biggest possible difference between two angles is 180° (pi radians).
  abs_diff =  np.abs( azi_1 - azi_2 )
  return np.minimum( abs_diff, 2*np.pi - abs_diff )


def great_circle_distance( azi_1, zen_1, azi_2, zen_2  ): 

    dot_prod = \
        np.cos(zen_1) * np.cos(zen_2) + \
        np.sin(zen_1) * np.sin(zen_2) * np.cos( azi_1 - azi_2 )
        
    return np.arccos(dot_prod)