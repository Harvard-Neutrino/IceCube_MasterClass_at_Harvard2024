
# CREDIT: Felix Yu, felixyu7@github.com

# @title Run me!

import numpy as np

from IPython.display import display, clear_output
from ipywidgets import interact, FloatSlider, Dropdown, Button, Layout

from src.plot_event import *
from src.direction_utils import *


EVENT_LIST = [24, 18973, 25456, 80336, 18901, 172, 47547, 5717, 831, 696, 82031, 3458, 755, 58125, 385]
TRACK_EVENT_DICT = {str(idx+1): v for idx, v in enumerate(EVENT_LIST)}

CASCADE_EVENT_DICT = {str(idx+1): idx for idx in range(10)}

def calc_center_of_gravity( hits ):
    return hits.mean(axis=0)

def show_results( button, fig, evt ):
    pass

def reco_game( event_selection ):

    # setup widget objects
    zenith = FloatSlider(
        min=0, max=3.14, step=0.01, 
        value=2/3 * np.pi, description='zenith',
        layout=Layout(width='75%')
    )

    azimuth = FloatSlider(
        min=-3.14, max=3.14, step=0.01,
        value=np.pi/6, description='azimuth',
        layout=Layout(width='75%')
    )

    camera_phi = FloatSlider(
        min=-180, max=180, step=5,
        value=0, description='camera: rotate left/right',
        layout=Layout(width='75%')
    )

    camera_theta = FloatSlider(
        min=-90, max=90, step=5, 
        value=20, description='camera: rotate up/down',
        layout=Layout(width='75%')
    )

    event_id = Dropdown(
        value='1', options=[str(x+1) for x in range(10)], 
        description='event_id', disabled=False
    )

    submit_button = Button(description='Submit')

    # constants:
    EVT_DICT = TRACK_EVENT_DICT
    get_evt = lambda num: event_selection[ EVT_DICT[num] ]

    # # variable:  
    # current_evt_num = event_id.value
    # fig = go.FigureWidget( data=plot_I3det(), layout=get_3d_layout() )

    # # initialize 
    # evt = get_evt( current_evt_num )
    # pivot = calc_center_of_gravity( evt.hits_xyz )
    # fig.add_trace( plot_first_hits(evt) )
    # fig.add_traces( 
    #     plot_direction(
    #         get_direction_vector_from_angles(azimuth.value, zenith.value), 
    #         pivot,
    #         color="black" 
    #     )
    # )

    # setup action functions 
    def update_fig_display( num, azi, zen, cam_phi, cam_th ):   

        # nonlocal current_evt_num    
        # nonlocal fig 

        clear_output()

        evt = get_evt(num)

        fig = go.FigureWidget( data=plot_I3det(), layout=get_3d_layout() )
        fig.layout["scene"]["camera"]["eye"] = get_eye_xyz( np.deg2rad(cam_phi), np.deg2rad(cam_th) )

        fig.add_trace( plot_first_hits(evt) )
        fig.add_traces( 
            plot_direction(
                get_direction_vector_from_angles( azi, zen ), 
                calc_center_of_gravity(evt.hits_xyz),
                color="black" 
            )
        )
        fig.show()

        # if current_evt_num != num:

        #     # remove arrow and evt 
        #     fig.data = fig.data[:-3]

        #     # re-plot evt
        #     fig.add_trace( plot_first_hits(evt) )
        #     current_evt_num = num

        # else:

        #     # remove arrow 
        #     fig.data = fig.data[:-2]

        # # re-plot arrow
        # fig.add_traces( 
        #     plot_direction(
        #         get_direction_vector_from_angles(azi, zen), 
        #         calc_center_of_gravity(evt.hits_xyz) ,
        #         color="black" )
        # )

        # fig.layout["scene"]["camera"]["eye"] = get_eye_xyz( np.deg2rad(cam_phi), np.deg2rad(cam_th) )
        # fig.show()
        
    interact(
        lambda num, azi, zen, cam_phi, cam_th: update_fig_display( num, azi, zen, cam_phi, cam_th ),
        num = event_id,
        azi = azimuth,
        zen = zenith,
        cam_phi = camera_phi,
        cam_th = camera_theta,
        continuous_update = False
    )

    submit_button.on_click(
        lambda b: print("submitted!")
        # lambda button: show_results( button, fig, get_evt(num) )
    )

    # fig.show()
    display(submit_button)
    return None