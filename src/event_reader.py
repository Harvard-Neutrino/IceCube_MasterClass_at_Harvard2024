
from pandas import read_parquet, Series, DataFrame, RangeIndex
import numpy as np

from .direction_utils import bound_azi, bound_zen


class Event():

    def __init__( self, photon_hits, mc_truth ):

        self.true_muon_energy = mc_truth["final_state_energy"][0]
        self.true_muon_zenith = mc_truth["final_state_zenith"][0]
        self.true_muon_azimuth = mc_truth["final_state_azimuth"][0]

        self.hits_t = photon_hits["t"]
        self.hits_xyz = np.vstack([ photon_hits[f"sensor_pos_{a}"] for a in ("x", "y", "z") ]).T
        self.string_id = photon_hits["string_id"]
        self.sensor_id = photon_hits["sensor_id"] 

        return None
    
    def __repr__(self):
        print(f"IceCube Event with {len(self.hits_t)} hits: \n")
        return repr( 
            DataFrame.from_dict( dict(
                t=self.hits_t,
                dom=[ t for t in zip(self.string_id, self.sensor_id) ]
                # x=self.hits_xyz[:, 0],
                # y=self.hits_xyz[:, 1],
                # z=self.hits_xyz[:,2]
            ))
        ) 

"""
wrapper around the combined Prometheus, LI output,
with two kinds of indexing:
    integer based for indexing events,
    string based for indexing a given property of all events.
"""
class EventSelection():

    def __init__( self, mc_truth, prometheus_photons=None ):

        self.mc_truth = mc_truth 
        self.mc_keys = next(iter(mc_truth)).keys()
        self.photons = prometheus_photons

        self.indices = mc_truth.index
        self.N_events = len(mc_truth)

        return None

    def __len__(self): return self.N_events

    def __getitem__( self, idxs ):

        # handle string-based indexing for evt attributes:
        if isinstance(idxs, str):
            key = idxs
            if key in self.mc_keys:
                return np.array( [ evt[key] for evt in self.mc_truth] )
            else:
                raise AttributeError(key)

        # handle slices
        elif isinstance(idxs, slice):
            if idxs.step is None:
                return [ self[idx] for idx in range(idxs.start, idxs.stop) ]
            else:
                return [ self[idx] for idx in range(idxs.start, idxs.stop, idxs.step) ]
        
        elif isinstance(idxs, tuple):
            return [ self[idx] for idx in idxs ]
        
        # base case: 
        else:
            idx = idxs
            if idx in self.indices:
                return Event( 
                    self.photons[idx], 
                    self.mc_truth[idx],
                )
            else: raise IndexError(idx)



def load_sim_events(fpath="."):

    if "moonshadow" in fpath:
        # Jeff's moon shadow simulation files from last year:
        out = read_parquet(fpath)
        mc_truth = Series(
            [ out["mc_truth_initial"][i] | out["mc_truth_final"][i] | out["reco_quantities"][i] | dict(mjd_time=out["times"][i]) \
                for i in out["mc_truth_initial"].index ]
        )

    else:
        # Nick's simulation files: 
        out = read_parquet(fpath)
        out.set_index( RangeIndex(start=0, stop=len(out), step=1) )
        mc_truth =  out["mc_truth"]

    return EventSelection( mc_truth, out["photons"] )