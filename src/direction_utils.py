

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
