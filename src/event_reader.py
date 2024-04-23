
from pandas import read_parquet
from pandas import DataFrame

# just a wrapper with a __repr__ attribute...
class Event():

    def __init__( self, photon_hits, true_q  ):

        self.true_muon_zenith = true_q["final_state_zenith"][0]
        self.true_muon_azimuth = true_q["final_state_azimuth"][0]

        # self.hits = photon_hits
        self.hits = DataFrame( photon_hits )[[
            "t", "string_id", "sensor_id",
            "sensor_pos_x", "sensor_pos_y", "sensor_pos_z",
        ]]

        return None
    
    def __repr__(self): 
        print( type(self) )
        return repr( self.hits )
        # return repr( DataFrame( self.hits )[[
        #     "t", "string_id", "sensor_id",
        #     "sensor_pos_x", "sensor_pos_y", "sensor_pos_z",
        # ]] )

# a wrapper around the output of read_parquet,
# with specialized get
class EventSelection():

    def __init__( self, fpath ):

        self.fpath = fpath
        out = read_parquet( fpath )

        self.N_events = len( out["times"] )
        self.event_mjd_times = out["times"]

        self.event_hit_info = out["photons"]
        self.mc_truth = out["mc_truth_final"]

        return None
    
    def __getitem__(self, idx):
        if (idx < 0) or (idx > self.N_events):
            raise IndexError
        return Event( 
            self.event_hit_info[idx], 
            self.mc_truth[idx]
        )