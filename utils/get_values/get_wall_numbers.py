from utils.pandas_apply_functions import order_walls
import pandas as pd
from utils.get_values.get_trial_walls import get_trial_walls

def get_wall1_wall2(octpy_metadata, grating_high=True):
    """ get the first and second walls, where wall1 is high if 
        and wall2 is low for choice trials
      
     return pandas series that can be added into a dataframe or 
      converted to a numpy array """
    
  
    om = octpy_metadata.copy()
    om['trial_walls'] = om.apply(lambda x: get_trial_walls(x['trial_type'], x['wall']), axis=1)

    # get trial walls function puts grating wall number in idx 0 (assuming octpy trial data
    # still stores walls as "light wall num, grating wall num")
    # Here, reverse this order so light wall number is in idx 0, if light == High
    if grating_high:
      om['wall_1'] = om.apply(lambda x: x['trial_walls'][0], axis=1)
      om['wall_2'] = om.apply(lambda x: x['trial_walls'][1], axis=1)
    elif not grating_high:
      om['wall_1'] = om.apply(lambda x: x['trial_walls'][1], axis=1)
      om['wall_2'] = om.apply(lambda x: x['trial_walls'][0], axis=1)
       

    return om['wall_1'].values, om['wall_2'].values
