
# CREDIT: Felix Yu, felixyu7@github.com
# #@title Run me!

import numpy as np
from IPython.display import display, clear_output
from ipywidgets import interact, FloatSlider, Dropdown, Button, Layout

from src.plot_event import *
from src.direction_utils import *



PRESEL_TRACK_EVENTS = [24, 18973, 25456, 80336, 18901, 172, 47547, 5717, 831, 696, 82031, 3458, 755, 58125, 385]
TRACK_EVENT_DICT = { str(idx+1): v for idx, v in enumerate(PRESEL_TRACK_EVENTS) }

BASIC_EVT_DICT = { str(idx+1): idx for idx in range(10) }

def display_evt_and_arrow( evt, zen, azi, calc_dist=False):

    clear_output()
    fig = display_evt( evt )
    fig.add_traces( plot_direction(
        get_direction_vector_from_angles(azi, zen),
        calc_center_of_gravity(evt.hits_xyz),
        color="black"
    ))

    if calc_dist:
        mean_dist = calc_mean_perpendicular_distance( 
            np.array([azi, zen]),
            evt.hits_xyz,
            calc_center_of_gravity( evt.hits_xyz )
        ) 
        print("\n")
        print( f"the mean distance is {mean_dist:.2f} meters \n" )
        print("\n")

    fig.show()


"""
starts a new game given existing widgets. 
"""
def start_new_game( event_id, zenith, azimuth, events, EVT_DICT, calc_dist=False ):

    submit_button = Button(description='Submit')

    g = lambda button: reco_results( events, EVT_DICT, button, event_id, zenith, azimuth, calc_dist)
    submit_button.on_click(g)
    
    f = lambda x, y, z: display_evt_and_arrow( get_evt(x, EVT_DICT, events), y, z, calc_dist )
    interact(f, x=event_id, y=zenith, z=azimuth, continuous_update=False)

    display(submit_button)


"""
displays true values + angular error
"""
def reco_results( events, EVT_DICT, button, event_id, zenith, azimuth, calc_dist):

    clear_output()

    evt = get_evt( event_id.value, EVT_DICT, events )

    fig = display_evt( evt )
    pivot_pt = calc_center_of_gravity( evt.hits_xyz )

    true_zenith = evt.true_muon_zenith
    true_azimuth = evt.true_muon_azimuth
    true_dir_vec = get_direction_vector_from_angles( true_azimuth, true_zenith )
    pred_dir_vec = get_direction_vector_from_angles( azimuth.value, zenith.value )

    fig.add_traces( plot_direction( pred_dir_vec, pivot_pt, color="black" ) )
    fig.add_traces( plot_direction( true_dir_vec, pivot_pt, color="red" ) )
    fig.show()

    ad = np.rad2deg( great_circle_distance(true_zenith, true_azimuth, zenith.value, azimuth.value) )

    print("\n")
    print( f"Your estimate was {ad:.2f}° off the true direction.")
    print("\n")
    button.close()

    return_button = Button(description='Return')
    f = lambda button: return_to_game( button, event_id, zenith, azimuth, events, EVT_DICT, calc_dist )    
    return_button.on_click(f)
    display(return_button)


"""
clears output, closes the return button, and starts a new game. 
"""
def return_to_game(button, event_id, zenith, azimuth, events, EVT_DICT, calc_dist):

    clear_output()
    button.close()

    start_new_game( event_id, zenith, azimuth, events, EVT_DICT, calc_dist=calc_dist )


"""
initializes the persistent game widgets.
"""
def init_game_widgets():

    zenith = FloatSlider(
        min=0, max=3.14, step=0.01, 
        value=np.pi * 2/3, 
        description='zenith',
        layout=Layout(width='75%')
    )

    azimuth = FloatSlider(
        min=-3.14, max=3.14, step=0.01, 
        value = 0, 
        description='azimuth',
        layout=Layout(width='75%')
    )

    event_id = Dropdown(
        value='1', options=[str(x+1) for x in range(10)], 
        description='event_id', disabled=False
    )

    return event_id, zenith, azimuth


"""
main function
"""
def reco_game( events, event_type="track", calc_dist=False ):

    event_id, zenith, azimuth = init_game_widgets()

    if event_type=="track":
        EVT_DICT = TRACK_EVENT_DICT

    # elif event_type=="cascade":
    #     events = events[ PRESEL_CASCADE_EVENTS

    else: EVT_DICT = BASIC_EVT_DICT
    return start_new_game( event_id, zenith, azimuth, events, EVT_DICT, calc_dist )



# event handling utility 
def get_evt( num, EVT_DICT, events ): return events[ EVT_DICT[num] ]

def calc_center_of_gravity( hits ):
    return hits.mean(axis=0)


"""
This function uses the equation above to calculate the perpendicular distance between a single point and a pivot.
"""
def calc_perpendicular_distance( hit_pt, dir_vec, pivot_pt ):
    dist_vec = hit_pt - pivot_pt
    return np.sqrt( np.dot( dist_vec, dist_vec ) -  np.dot( dist_vec, dir_vec )**2 )


"""
Given a set of hits and a pair of guess angles, calculate the mean of all the perpendicular distances!
"""
def calc_mean_perpendicular_distance( dir_angles, hit_pts, pivot_pt ):
    
    dir_vec = get_direction_vector_from_angles( dir_angles[0], dir_angles[1] )
    return np.mean([ calc_perpendicular_distance(hit, dir_vec, pivot_pt) for hit in hit_pts ])