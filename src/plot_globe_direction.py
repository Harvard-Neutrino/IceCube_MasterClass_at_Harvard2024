
import plotly
import plotly.graph_objs as go
import numpy as np

# utility functions: 
from ipywidgets import interact, interactive, FloatSlider, Layout
from IPython.display import display

# assumes theta, phi in degrees.
def unitsphere_to_cart(theta, phi):
    theta *= np.pi/180
    phi   *= np.pi/180

    x = np.cos(theta) * np.cos(phi)
    y = np.cos(theta) * np.sin(phi)
    z = np.sin(theta)

    return np.array([x,y,z])

# assumes azimuth, zenith in degrees.
def local_to_global_dir(zen=0.0, azi=0.0, lat=-90.0, long=0.0):

    azi *= np.pi/180
    zen *= np.pi/180

    # fix the local direction of the north pole
    zen_NP = (90.0 - lat) * np.pi/180 
    azi_NP = (180.0 - long) * np.pi/180

    # fix the global direction of the local x-axis.
    phi_x = 0.0   * np.pi/180

    sz_NP, cz_NP = np.sin(zen_NP), np.cos(zen_NP)

    sin_theta = cz_NP * np.cos(zen) + sz_NP * np.sin(zen) * np.cos(azi - azi_NP) 
    sin_phi = np.sin(zen) * np.sin(azi - azi_NP)
    cos_phi = np.cos(zen) * sz_NP - np.sin(zen) * cz_NP * np.cos(azi - azi_NP)

    phi = phi_x - np.arctan2(sin_phi, cos_phi)
    theta = np.arcsin(sin_theta)

    theta *= 180/np.pi
    phi *= 180/np.pi

    return theta, phi


def get_earth_object():

    # load from downloaded files
    longs = np.loadtxt("earthmap_longs.csv", delimiter=',')
    lats = np.loadtxt("earthmap_lats.csv", delimiter=',')
    zcolor = np.loadtxt("earthmap_colors.csv", delimiter=',')  

    xs = np.cos(longs) * np.cos(lats)
    ys = np.sin(longs) * np.cos(lats)
    zs = np.sin(lats)

    # define colormap
    ctopo = [
        [0, 'rgb(0, 0, 70)'],[0.2, 'rgb(0,90,150)'], 
        [0.4, 'rgb(150,180,230)'], [0.5, 'rgb(210,230,250)'],
        [0.50001, 'rgb(0,120,0)'], [0.57, 'rgb(220,180,130)'], 
        [0.65, 'rgb(120,100,0)'], [0.75, 'rgb(80,70,0)'], 
        [0.9, 'rgb(200,200,200)'], [1.0, 'rgb(255,255,255)']
    ]
    cmin = -8000
    cmax = 8000

    globe = dict(
        type='surface',
        x=xs, y=ys, z=zs, 
        colorscale=ctopo, surfacecolor=zcolor,
        cmin=cmin, cmax=cmax,
        showscale=False
    )

    return globe

def get_layout():
    noaxis=dict(
        showbackground=False,
        showgrid=False,
        showline=False,
        showticklabels=False,
        ticks='',
        title='',
        zeroline=False,
        range=(-2, 2)
    )

    layout = go.Layout(
        autosize=False, width=400, height=400,
        showlegend = False,
        margin=dict(l=10, r=10, t=10, b=10),
        scene = dict(
            xaxis = noaxis,
            yaxis = noaxis,
            zaxis = noaxis,
            aspectmode='manual',
            aspectratio=go.layout.scene.Aspectratio(
            x=1, y=1, z=1)),
    )
    return layout


def plot_direction_from_earth(lat=0.0, long=-90.0, azi=0.0, zen=0.0):

    earth = get_earth_object()
    layout = get_layout()

    pt_obs = unitsphere_to_cart(lat, long)
    theta, phi = local_to_global_dir(azi=azi, zen=zen, lat=lat, long=long)
    vec_dir = unitsphere_to_cart(theta, phi)

    cz = np.cos(zen * np.pi/180)
    vec_len = -cz + np.sqrt( cz**2 + 3 )
    pt_sky = pt_obs + vec_len * vec_dir

    pts = [ [pt_obs[i], pt_sky[i]] for i in range(3) ]
    arrow_line = go.Scatter3d(
        x = pts[0], y = pts[1], z = pts[2],
        marker=dict(size=2, color="dodgerblue", line=dict(width=0)),
        line=dict(color="dodgerblue", width=6)
    )

    observer = go.Scatter3d(
        x = [pt_obs[0],], y = [pt_obs[1],], z = [pt_obs[2],],
        marker=dict(size=6, color="dodgerblue"),
    ) 

    ah = 0.2
    arrow_vec = ah * vec_dir
    arrow_head = go.Cone(
        x = [pt_sky[0],], y = [pt_sky[1],], z = [pt_sky[2],],
        u = [arrow_vec[0],], v = [arrow_vec[1],], w = [arrow_vec[2],],
        anchor="center", showscale=False, sizemode='absolute', sizeref=0.3, colorscale=[[0, "dodgerblue"], [1, "dodgerblue"]]
    )

    plot_data=[earth, observer, arrow_line, arrow_head]
    fig = go.Figure(data=plot_data, layout=layout)
    display(fig)
    # return fig


# zenith = FloatSlider(min=0, max=3.14, step=0.01, value=0, description='zenith',
#                      layout=Layout(width='75%'))
# azimuth = FloatSlider(min=-3.14, max=3.14, step=0.01, value=0, description='azimuth',
#                      layout=Layout(width='75%'))

# zenith_ = FloatSlider(min=0, max=190, step=1.0, value=0, description='zenith',
#                      layout=Layout(width='75%'))
# azimuth_ = FloatSlider(min=0, max=360, step=1.0, value=0, description='azimuth',
#                      layout=Layout(width='75%'))

# def interactive_earth_direction():
#     zenith = FloatSlider(min=0, max=190, step=1.0, value=0, description='zenith',
#                      layout=Layout(width='75%'))
#     azimuth = FloatSlider(min=0, max=360, step=1.0, value=0, description='azimuth',
#                      layout=Layout(width='75%'))
#     f = lambda azi, zen: plot_direction_from_earth(lat=-90.0, long=-90, azi=azi, zen=zen)
#     interact(f, azi=azimuth, zen=zenith, continuous_update=True)