
# CREDIT: Felix Yu, felixyu7@github.com

# #@title Run me!
import numpy as np

from IPython.display import display, clear_output
from ipywidgets import interact, FloatSlider, Dropdown, Button, Layout

# from google.colab import output
# output.enable_custom_widget_manager()

from src.plot_event import *
from src.direction_utils import *


TRACK_EVENT_DICT = {str(idx+1): idx for idx in range(10)}
CASCADE_EVENT_DICT = {str(idx+1): idx for idx in range(10)}


# def calculate_angular_difference(zenith1, azimuth1, zenith2, azimuth2):
#     # Convert angles to Cartesian coordinates
#     cartesian1 = np.array([
#         np.sin(zenith1) * np.cos(azimuth1),
#         np.sin(zenith1) * np.sin(azimuth1),
#         np.cos(zenith1)
#     ])
#     cartesian2 = np.array([
#         np.sin(zenith2) * np.cos(azimuth2),
#         np.sin(zenith2) * np.sin(azimuth2),
#         np.cos(zenith2)
#     ])

#     # Normalize vectors to unit length
#     cartesian1 /= np.linalg.norm(cartesian1)
#     cartesian2 /= np.linalg.norm(cartesian2)

#     # Calculate dot product
#     dot_product = np.dot(cartesian1, cartesian2)

#     # Calculate angular difference (in radians)
#     angular_difference_rad = np.arccos(dot_product)

#     # Convert to degrees
#     angular_difference_deg = np.degrees(angular_difference_rad)

#     return angular_difference_deg




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
    ed = TRACK_EVENT_DICT
    # if events._photon_info["mc_truth_initial", "initial_state_type", 0]==12:
    #     ed = CASCADE_EVENT_DICT
    
    layout = get_3d_layout()
    plot_det = plot_I3det()
    fig = go.FigureWidget(data=plot_det, layout=layout)

    plot_evt = plot_first_hits(events[ed[x]])
    fig.add_trace(plot_evt)
    
    dir_vec, pivot_pt = get_dir_and_pt(events[ed[x]])
    pred_dir_vec = np.array([np.cos(z) * np.sin(y), 
                            np.sin(z) * np.sin(y), 
                            np.cos(y)])
    fig.add_traces(plot_direction( pred_dir_vec, pivot_pt, color="dodgerblue" ))
    fig.add_traces(plot_direction( dir_vec, pivot_pt, color="red" ))
    fig.show()
    
    true_zenith = events[ed[event_id.value]].true_muon_zenith
    true_azimuth = events[ed[event_id.value]].true_muon_azimuth
    # true_azimuth = events._photon_info[int(event_id.value)].mc_truth_initial.initial_state_azimuth
    ad = np.rad2deg( great_circle_distance(true_zenith, true_azimuth, zenith.value, azimuth.value) )
    print('Your estimate was ' + str(ad) + ' degrees off the true direction.')
    button.close()
    return_button = Button(description='Return')
    f = lambda button: return_to_game(button, event_id, zenith, azimuth, events)
    
    return_button.on_click(f)
    display(return_button)

def reco_game(events):
    zenith = FloatSlider(min=0, max=3.14, step=0.01, value=0, description='zenith',
                     layout=Layout(width='75%'))
    azimuth = FloatSlider(min=-3.14, max=3.14, step=0.01, value=0, description='azimuth',
                     layout=Layout(width='75%'))
    event_id = Dropdown(value='1', options=[str(x+1) for x in range(10)], description='event_id', disabled=False)
    submit_button = Button(description='Submit')
    g = lambda button: reco_results(events, button, event_id, zenith, azimuth)
    submit_button.on_click(g)
    # submit_button.on_click(reco_results)
    f = lambda x, y, z: intermediate_plot_fn(events, x, y, z)
    interact(f, x=event_id, y=zenith, z=azimuth, continuous_update=False)
    display(submit_button)

def get_dir_and_pt(event):

    # xs = np.array(list(event.hits.to_dict()['sensor_pos_x'].values()))
    # ys = np.array(list(event.hits.to_dict()['sensor_pos_y'].values()))
    # zs = np.array(list(event.hits.to_dict()['sensor_pos_z'].values()))
    xs, ys, zs = [ event.hits_xyz[:, i] for i in (0, 1, 2) ]
    
    muon_zen = event.true_muon_zenith
    muon_azi = event.true_muon_azimuth

    # dir_vec = np.array( [np.cos(muon_azi) * np.sin(muon_zen), 
    #                     np.sin(muon_azi) * np.sin(muon_zen), 
    #                     np.cos(muon_zen)] )
    
    dir_vec = get_direction_vector_from_angles( muon_azi, muon_zen )

    # get center of gravity point
    pivot_pt = np.mean( event.hits_xyz, axis=0 ) 
    
    # pivot_pt = np.array( [xs.mean(), ys.mean(), zs.mean()])
    
    return dir_vec, pivot_pt