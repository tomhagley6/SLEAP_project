import h5py

""" Use these getters if manually running a single trial trajectory extraction """

def get_node_names(trajectory_file_path):
    """ import node names from .h5 sleap export file
        to avoid time cost of repeating this each loop
        iteration when extracting trajectories """
    
    with h5py.File(trajectory_file_path, 'r') as f:
        node_names = [name.decode() for name in f['node_names'][:]] # decode string UTF-8 encoding

        return node_names
    
def get_locations(trajectory_file_path):
    """ import node names from .h5 sleap export file
        to avoid time cost of repeating this each loop
        iteration when extracting trajectories """
    
    with h5py.File(trajectory_file_path, 'r') as f:
        locations = f['tracks'][:].T

        return locations