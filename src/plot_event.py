
import plotly.graph_objs as go
import numpy as np

"""
for modifying the plotly layout.scene.camera.eye attribute.
"""
def get_eye_xyz( camera_phi, camera_th, zoom=2 ):
    return dict(
        x = zoom * np.cos(camera_phi) * np.cos(camera_th),
        y = zoom * np.sin(camera_phi) * np.cos(camera_th),
        z = zoom * np.sin(camera_th)
    )

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

                # 'which direction is up'?
                up=dict(x=0, y=0, z=1),

                # 'translate?'
                center=dict(x=0, y=0, z=0),

                # 
                eye=get_eye_xyz( np.pi/6, np.pi/6, 2 )
            )
        ),
        margin=dict(l=20, r=20, t=40, b=40),
    )


"""
plot objects for the strings and the boundaries of the detector.
"""
def plot_I3det():

    I3_dom_z = np.loadtxt("./IceCube_MasterClass_at_Harvard2024/resources/detector_info/I3_dom_zpos.txt")
    I3_str_xy = np.loadtxt("./IceCube_MasterClass_at_Harvard2024/resources/detector_info/I3_string_xypos.txt")
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

"""
visualize an event by making a scatter plot of the unique DOMs hit.
Based on code prepared by Felix Yu (felixyu7) and Jeffrey Lazar (jlazar17) for the 2023 MasterClass.
"""
def plot_first_hits( evt ):

    # hits = evt.hits[["t", "sensor_pos_x", "sensor_pos_y", "sensor_pos_z"]].to_numpy()
    hits = np.column_stack( [evt.hits_t, evt.hits_xyz] )

    # sort hits by time
    sorted_hits = hits[ np.argsort( evt.hits_t ) ]

    # np.unique returns the sorted array, the indices of the unique items, and the counts 
    _, unique_inds, n_hits = np.unique( sorted_hits[:, 1:4], axis=0, return_index=True, return_counts=True )
    first_hits = sorted_hits[unique_inds]

    hits = go.Scatter3d(
            x = first_hits[:, 1], 
            y = first_hits[:, 2], 
            z = first_hits[:, 3],
            customdata = first_hits[:, 0], 
            mode = 'markers',
            marker = dict(
                size = 4,
                color = first_hits[:,0],
                colorscale = 'Rainbow_r',   
            ),
            showlegend=False,
            hoverinfo=['x','y','z','text'], 
            hovertext=['%.2f ns' % t for t in first_hits[:,3]], 
            hovertemplate="x: %{x} m, y: %{y} m, z: %{z} m, t: %{customdata} ns",
            name="current_evt"
    )

    return hits

def plot_direction( dir_vec, pivot_pt, color="black" ):

    pt_0 = pivot_pt - 500 * dir_vec
    pt_1 = pivot_pt + 500 * dir_vec
    arrow_vec = 20 * dir_vec

    plot_dir_line = go.Scatter3d(
            x = [ pt_0[0], pt_1[0] ],
            y = [ pt_0[1], pt_1[1] ],
            z = [ pt_0[2], pt_1[2] ],
            mode ='lines',
            line = dict( color=color, width=6 ),
            showlegend=False,
            name="arrow_line",
            # marker = dict( color='black', size=4 )
        )

    plot_dir_arrow = go.Cone(
        x = [ pt_1[0] ],
        y = [ pt_1[1] ],
        z = [ pt_1[2] ],
        u = [ arrow_vec[0], ],
        v = [ arrow_vec[1], ],
        w = [ arrow_vec[2], ],
        anchor="center",
        showscale=False,
        sizemode="absolute",
        sizeref=100, 
        colorscale=[[0, color], [1, color]],
        name="arrow_head"
    )

    return [ plot_dir_line, plot_dir_arrow ]


def display_evt( evt ):
    fig = go.FigureWidget( data=plot_I3det(), layout=get_3d_layout() )
    fig.add_trace( plot_first_hits( evt ) )    
    return fig

# def display_event( fig, evt ):
#     fig.add_trace( plot_first_hits(evt) )
#     return fig