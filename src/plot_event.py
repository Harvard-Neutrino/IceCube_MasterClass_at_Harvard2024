
import plotly.graph_objs as go
import numpy as np

"""
cleaner plotly.graph_objs.Layout object for 3d plots.
"""
def get_3d_layout():
    axis_settings = dict(
        showgrid=False, 
        showticklabels=False,
        backgroundcolor='whitesmoke', 
        title='',
        showspikes=False
        # tickfont=dict(size=10)
    )
    return go.Layout(
            scene=dict(
            xaxis=axis_settings,
            yaxis=axis_settings,
            zaxis=axis_settings,
            camera=dict(
                up=dict(x=0, y=0, z=1),
                center=dict(x=0, y=0, z=0),
                eye=dict(x=1.5, y=1.5, z=0.1)
            )
        ),
        margin=dict(l=20, r=20, t=40, b=40),
    )


"""
plot objects for the strings and the boundaries of the detector.
"""
def plot_I3det():

    I3_dom_z = np.loadtxt("../resources/detector_info/I3_dom_zpos.txt")
    I3_str_xy = np.loadtxt("./resources/detector_info/I3_string_xypos.txt")
    N_strings = I3_str_xy.shape[0]

    I3_strings = [ go.Scatter3d( 
            x=[x,x], 
            y=[y,y], 
            z=[I3_dom_z[0], I3_dom_z[-1]],
            mode='lines',
            line=dict(color='lightgrey', width=1),
            showlegend=False,
            hoverinfo="skip"
        ) for (x,y) in I3_str_xy 
    ]

    boundary_strs = np.array([75, 31, 1, 6, 50, 74, 72, 78, 75])
    I3_borders = [ go.Scatter3d(
            x=I3_str_xy[boundary_strs-1, 0],
            y=I3_str_xy[boundary_strs-1, 1],
            z=np.full(N_strings, z),
            mode='lines',
            line=dict(color='grey', width=1),
            showlegend=False,
            hoverinfo="skip"
        ) for z in I3_dom_z[[-1, 0]]
    ]

    return I3_strings + I3_borders


def plot_first_hits(evt):

    hits = evt.hits[["t", "sensor_pos_x", "sensor_pos_y", "sensor_pos_z"]].to_numpy()

    # sort hits by time
    sorted_hits = hits[ np.argsort( hits[:,0] ) ]

    # np.unique returns the sorted array, the indices of the unique items, and the counts 
    _, unique_inds, n_hits = np.unique( sorted_hits[:, 1:4], axis=0, return_index=True, return_counts=True )
    first_hits = sorted_hits[unique_inds]

    hits = go.Scatter3d(
            x = first_hits[:, 1], 
            y = first_hits[:, 2], 
            z = first_hits[:, 3],
            customdata = first_hits[:,0], 
            mode = 'markers',
            marker = dict(
                size = 4,
                color = first_hits[:,0],
                colorscale = 'Rainbow_r',   
            ),
            showlegend=False,
            hoverinfo=['x','y','z','text'], 
            hovertext=['%.2f ns' % t for t in first_hits[:,3]], 
            hovertemplate="x: %{x} m, y: %{y} m, z: %{z} m, t: %{customdata} ns"
    )

    return hits
