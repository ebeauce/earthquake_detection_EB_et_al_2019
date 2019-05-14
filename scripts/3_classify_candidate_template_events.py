import sys
sys.path.append('/nobackup1/ebeauce/automatic_detection/')
import automatic_detection as autodet
import numpy as np
import matplotlib.pyplot as plt
import h5py as h5
from obspy.core import UTCDateTime as udt
from time import time as give_time
from keras import models

path_classification  = './classification/'
path_template_db     = './template_event_database/'

classifier = models.load_model(path_classification + 'classifier.h5')

features_dataset = {}
with h5.File(path_classification + 'features_dataset.h5', mode='r') as f:
    for item in list(f.keys()):
        features_dataset[item] = f[item][()]

features     = features_dataset['features']
n_detections = features.shape[0]
features     = features.reshape(n_detections, -1)

t1 = give_time()
predictions = classifier.predict(features)
t2 = give_time()

print("{:.2f}sec to classify the {:d} detections".format(t2-t1, n_detections))

I = np.where(predictions[:,0] > 0.5)[0]

from keras.models import Model

layer_name = classifier.layers[1].name
intermediate_layer_model = Model(inputs=classifier.input,
                                         outputs=classifier.get_layer(layer_name).output)
intermediate_output = intermediate_layer_model.predict(features[I,:])

print('{:d} detections were classified as acceptable template events.'.format(I.size))

template_event_database = {}

#=================================================================================
#                FIRST CONCATENATION

n_stations   = 20
n_components = 3
n_samples    = np.int32(autodet.cfg.template_len * autodet.cfg.sampling_rate)

origin_times = np.zeros(0,                                         dtype=np.float64)
locations    = np.zeros((0, 3),                                    dtype=np.float32)
source_index = np.zeros(0,                                         dtype=np.int32)
moveouts     = np.zeros((0, n_stations, n_components),             dtype=np.float32)
stations     = np.zeros((0, n_stations),                           dtype=np.dtype('|S4'))
components   = np.zeros((0, n_components),                         dtype=np.dtype('|S4'))
waveforms    = np.zeros((0, n_stations, n_components, n_samples),  dtype=np.float32)

for i, idx in enumerate(I):
    filename = 'detections_{}_'.format(udt(features_dataset['days'][idx]).strftime('%Y%m%d'))
    index    = features_dataset['indexes'][idx]
    #----------------------------
    with h5.File(autodet.cfg.dbpath + filename + 'wav.h5', mode='r') as f:
        waveforms_  = f['waveforms'][index,:,:,:]
    with h5.File(autodet.cfg.dbpath + filename + 'meta.h5', mode='r') as f:
        stations_   = f['stations'][index,:]
        components_ = f['components'][()]
        moveouts_   = f['moveouts'][index,:,:]
        location_   = f['locations'][index,:]
        origin_t_   = f['origin_times'][index]
        source_idx_ = f['test_source_indexes'][index]
    waveforms = np.concatenate( (waveforms, waveforms_.reshape(1, n_stations, n_components, n_samples)), axis=0)
    locations = np.concatenate( (locations, location_.reshape(1, -1)), axis=0)
    moveouts  = np.concatenate( (moveouts, moveouts_.reshape(1, n_stations, n_components)), axis=0)
    stations  = np.concatenate( (stations, stations_.reshape(1, -1)), axis=0)
    components = np.concatenate( (components, components_.reshape(1, -1)), axis=0)
    origin_times = np.hstack( (origin_times, origin_t_))
    source_index = np.hstack( (source_index, source_idx_))

#--------------------------------------------
#    SAVE THE TEMPLATE DATABASE
with h5.File(path_template_db + 'template_event_database.h5', mode='w') as f:
    f.create_dataset('waveforms',    data=waveforms, compression='gzip')
    f.create_dataset('locations',    data=locations, compression='gzip')
    f.create_dataset('moveouts',     data=moveouts, compression='gzip')
    f.create_dataset('stations',     data=stations, compression='gzip')
    f.create_dataset('components',   data=components, compression='gzip')
    f.create_dataset('origin_times', data=origin_times, compression='gzip')
    f.create_dataset('source_index', data=source_index, compression='gzip')

def plot_template_event(idx):
    filename = 'detections_{}_'.format(udt(features_dataset['days'][idx]).strftime('%Y%m%d'))
    index    = features_dataset['indexes'][idx]
    #----------------------------
    with h5.File(autodet.cfg.dbpath + filename + 'wav.h5', mode='r') as f:
        waveforms_ = f['waveforms'][index,:,:,:]
    with h5.File(autodet.cfg.dbpath + filename + 'meta.h5', mode='r') as f:
        stations_   = f['stations'][index,:]
        components_ = f['components'][()]
        moveouts_   = np.int32(f['moveouts'][index,:,:] * autodet.cfg.sampling_rate)
        location_   = f['locations'][index,:]
        origin_t_   = udt(f['origin_times'][index])
    n_stations   = len(stations_)
    n_components = len(components_)
    t_min = 0
    t_max = moveouts_.max() + waveforms_.shape[-1]
    fig = plt.figure('detection_{:d}'.format(idx), figsize=(27,17))
    plt.suptitle('Detection on {} from {:.2f}|{:.2f}|{:.2f}km'.format(origin_t_.strftime('%Y,%m,%d %H:%M:%S'),
                                                                      location_[0], location_[1], location_[2]))
    for s in range(n_stations):
        for c in range(n_components):
            plt.subplot(n_stations, n_components, s*n_components + c + 1)
            time = np.arange(waveforms_.shape[-1]) + moveouts_[s,c]
            plt.plot(time, waveforms_[s,c,:], label='{}.{}'.format(stations_[s].astype('U'), components_[c].astype('U')))
            plt.xlim(t_min, t_max)
            plt.legend(loc='upper left', frameon=False)
    plt.subplots_adjust(top=0.94, bottom=0.08, left=0.05, right=0.95)
    plt.show()

#======================================================================================

