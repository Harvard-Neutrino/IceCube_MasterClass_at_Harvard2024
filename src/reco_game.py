
# CREDIT: Felix Yu, felixyu7@github.com
# #@title Run me!

import numpy as np
from IPython.display import display, clear_output
from ipywidgets import interact, FloatSlider, Dropdown, Button, Layout

from src.plot_event import *
from src.direction_utils import *


# TRACK_EVENT_DICT = {str(idx+1): idx for idx in range(10)}

EVENT_LIST = [24, 18973, 25456, 80336, 18901, 172, 47547, 5717, 831, 696, 82031, 3458, 755, 58125, 385]
TRACK_EVENT_DICT = {str(idx+1): v for idx, v in enumerate(EVENT_LIST)}

CASCADE_EVENT_DICT = {str(idx+1): idx for idx in range(10)}

def intermediate_plot_fn(events, x, y, z):
    ed = TRACK_EVENT_DICT
    # if events._photon_info["mc_truth_initial", "initial_state_type", 0]==12:
    #     ed = CASCADE_EVENT_DICT
    layout = get_3d_layout()
    plot_det = plot_I3det()
    fig = go.FigureWidget(data=plot_det, layout=layout)

    plot_evt = plot_first_hits( events[ed[x]] )
    fig.add_trace(plot_evt)
    
    _, pivot_pt = get_dir_and_pt(events[ed[x]])
    dir_vec = np.array([np.cos(z) * np.sin(y), 
                        np.sin(z) * np.sin(y), 
                        np.cos(y)])
    
    fig.add_traces( plot_direction( dir_vec, pivot_pt, color="dodgerblue" ) )
    fig.show()

def return_to_game(button, event_id, zenith, azimuth, events):

    clear_output()

    submit_button = Button(description='Submit')
    g = lambda button: reco_results(events, button, event_id, zenith, azimuth)
    submit_button.on_click(g)
    
    f = lambda x,y,z: intermediate_plot_fn(events, x, y, z)
    interact(f, x=event_id, y=zenith, z=azimuth, continuous_update=False)
    
    display(submit_button)

def reco_results(events, button, event_id, zenith, azimuth):

    # global event_id, zenith, azimuth
    x = event_id.value
    y = zenith.value
    z = azimuth.value

    clear_output()
    EVT_DICT = TRACK_EVENT_DICT
    # if events._photon_info["mc_truth_initial", "initial_state_type", 0]==12:
    #     ed = CASCADE_EVENT_DICT
    
    # layout = get_3d_layout()
    # plot_det = plot_I3det()
    # fig = go.FigureWidget(data=plot_det, layout=layout)

    # plot_evt = plot_first_hits(events[ed[x]])
    # fig.add_trace(plot_evt)

    evt = get_evt( event_id.value, EVT_DICT, events )

    fig = display_evt( evt )
    
    pivot_pt = calc_center_of_gravity( evt.hits_xyz )

    true_zenith = evt.true_muon_zenith
    true_azimuth = evt.true_muon_azimuth
    true_dir_vec = get_direction_vector_from_angles( true_azimuth, true_zenith )
    pred_dir_vec = get_direction_vector_from_angles( azimuth.value, zenith.value )

    fig.add_traces( plot_direction( pred_dir_vec, pivot_pt, color="dodgerblue" ) )
    fig.add_traces( plot_direction( true_dir_vec, pivot_pt, color="red" ) )
    fig.show()
    
    true_zenith = events[ed[event_id.value]].true_muon_zenith
    true_azimuth = events[ed[event_id.value]].true_muon_azimuth

    ad = np.rad2deg( great_circle_distance(true_zenith, true_azimuth, zenith.value, azimuth.value) )

    print( f"Your estimate was {ad:.2f}° off the true direction.")
    button.close()

    return_button = Button(description='Return')
    f = lambda button: return_to_game(button, event_id, zenith, azimuth, events)    
    return_button.on_click(f)
    display(return_button)

    

def reco_game( events ):

    zenith = FloatSlider(
        min=0, max=3.14, step=0.01, 
        value=0, 
        description='zenith',
        layout=Layout(width='75%')
    )

    azimuth = FloatSlider(
        min=-3.14, max=3.14, step=0.01, 
        value=0, 
        description='azimuth',
        layout=Layout(width='75%')
    )

    event_id = Dropdown(
        value='1', options=[str(x+1) for x in range(10)], 
        description='event_id', disabled=False
    )

    submit_button = Button(description='Submit')


    g = lambda button: reco_results(events, button, event_id, zenith, azimuth)
    submit_button.on_click(g)
    
    f = lambda x, y, z: intermediate_plot_fn(events, x, y, z)
    interact(f, x=event_id, y=zenith, z=azimuth, continuous_update=False)

    display(submit_button)


# event handling utility 

def get_evt( num, EVT_DICT, events ): return events[ EVT_DICT[num] ]

def display_evt( evt ):

    fig = go.FigureWidget( data=plot_I3det(), layout=get_3d_layout() )
    fig.add_trace( plot_first_hits( evt ) )    
    return fig

# def get_dir_and_pt(event):

#     xs, ys, zs = [ event.hits_xyz[:, i] for i in (0, 1, 2) ]
    
#     muon_zen = event.true_muon_zenith
#     muon_azi = event.true_muon_azimuth
    
#     dir_vec = get_direction_vector_from_angles( muon_azi, muon_zen )

#     # get center of gravity point
#     pivot_pt = calc_center_of_gravity( evt.hits_xyz )
    
#     return dir_vec, pivot_pt

def calc_center_of_gravity( hits ):
    return hits.mean(axis=0)