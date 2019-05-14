from .config import cfg
import numpy as np
import h5py as h5
from obspy.core import UTCDateTime as udt

# this is only an example with one function to read data
# custom functions should be added to this module to adapt
# to the architecture you are using to store your data

def ReadData(date, band):
    """
    ReadData(date, band)
    """
    date = udt(date)
    filename = 'data_{:d}_{:d}/data_{}.h5'.format(int(band[0]), int(band[1]), date.strftime('%Y%m%d'))
    data = {}
    data.update({'metadata' : {}})
    with h5.File(cfg.input_path + filename, mode='r') as f:
        data.update({'waveforms' : f['waveforms'][()]})
        for item in f['metadata'].keys():
            if len(f['metadata'][item][()]) == 1:
                data['metadata'].update({item : f['metadata'][item][()][0]})
            else:
                data['metadata'].update({item : f['metadata'][item][()]})
    data['metadata'].update({'date' : udt(data['metadata']['date'])})
    for item in ['stations', 'components']:
        data['metadata'][item] = data['metadata'][item].astype('U')
    return data

