import automatic_detection as autodet
import numpy as np
import matplotlib.pyplot as plt
import h5py as h5
from obspy.core import UTCDateTime as udt
from scipy import stats as scistats

net = autodet.dataset.Network('network.in')
net.read()

#-----------------------------------------------
from obspy.geodetics.base import calc_vincenty_inverse
D = np.zeros((net.n_stations(), net.n_stations()), dtype=np.float32)
for s1 in range(D.shape[0]):
    for s2 in range(D.shape[0]):
        D[s1,s2], az, baz = calc_vincenty_inverse(net.latitude[s1], net.longitude[s1], net.latitude[s2], net.longitude[s2])
for s in range(D.shape[0]):
    D[s,s] += D[s, D[s,:] != 0.].min()/10.
#-----------------------------------------------

def feature_max_kurto(detections, sliding_window=1.):
    sliding_window = np.int32(sliding_window * autodet.cfg.sampling_rate)
    n_detections = detections['waveforms'].shape[0]
    n_stations   = detections['waveforms'].shape[1]
    n_components = detections['waveforms'].shape[2]
    max_kurto    = np.zeros((n_detections, n_stations, n_components, 1), dtype=np.float32)
    for d in range(n_detections):
            max_kurto[d, :, :, 0]  = np.max(autodet.clib.kurtosis(detections['waveforms'][d, :, :, :], W=sliding_window), axis=-1)
    return max_kurto

def feature_peaks_autocorr(detections, max_lag=1.):
    max_lag = np.int32(max_lag * autodet.cfg.sampling_rate)
    n_detections = detections['waveforms'].shape[0]
    n_stations   = detections['waveforms'].shape[1]
    n_components = detections['waveforms'].shape[2]
    statistical_moment_peaks_1 = np.zeros((n_detections, n_stations, n_components, 1), dtype=np.float32)
    statistical_moment_peaks_2 = np.zeros((n_detections, n_stations, n_components, 1), dtype=np.float32)
    statistical_moment_peaks_3 = np.zeros((n_detections, n_stations, n_components, 1), dtype=np.float32)
    average_cross_A            = np.zeros((n_detections, n_stations, n_components, 1), dtype=np.float32)
    #---------------------------------------------------------------
    for d in range(n_detections):
        A = np.zeros((n_stations, n_components, 2*max_lag), dtype=np.float32)
        indexes_stations = np.int32(net.stations_idx(detections['stations'][d,:].astype('|U4')))
        distances = D[indexes_stations, :]
        distances = distances[:, indexes_stations]
        weights   = np.power(distances, -1./2)
        weights_rs = np.repeat(weights, A.shape[-1]).reshape(weights.size, -1)
        for s in range(n_stations):
            for c in range(n_components):
                for n in range(A.shape[-1]):
                    A[s,c,n] = np.sum(detections['waveforms'][d, s, c, n: -A.shape[-1]+n] * detections['waveforms'][d, s, c, max_lag:-max_lag])
                max_auto  = A[s,c,:].max()
                if max_auto != 0.:
                    A[s,c,:] /= max_auto
                peaks = detections['waveforms'][d, s, c, autodet.template_search._detect_peaks(A[s,c,:], mpd=np.int32(max_lag/20))]
                if peaks.size == 0:
                    continue
                statistical_moment_peaks_1[d, s, c, 0] = np.std(peaks)
                statistical_moment_peaks_2[d, s, c, 0] = scistats.skew(peaks,     bias=False)
                statistical_moment_peaks_3[d, s, c, 0] = scistats.kurtosis(peaks, bias=False)
    return np.concatenate( (statistical_moment_peaks_1, statistical_moment_peaks_2, statistical_moment_peaks_3) , axis=-1)

def features_max_amp(detections):
    n_detections = detections['waveforms'].shape[0]
    n_stations   = detections['waveforms'].shape[1]
    n_components = detections['waveforms'].shape[2]
    max_amp      = np.zeros((n_detections, n_stations, n_components, 1), dtype=np.float32)
    for d in range(n_detections):
            max_amp[d, :, :, 0]  = np.max(detections['waveforms'][d, :, :, :], axis=-1)
    return max_amp

#dates = net.datelist()
dates = [udt('2013,03,17')]

av_features = []
features    = []
filenames   = []
days        = []
indexes     = []

for date in dates:
    print('Extract features for day {}'.format(date.strftime('%d-%m-%Y')))
    filename      = 'detections_{}_'.format(date.strftime('%Y%m%d'))
    day           = date.strftime('%Y,%m,%d')
    detections    = autodet.db_h5py.read_detections(filename, attach_waveforms=True)
    #-----------------------------------------
    #        NORMALIZE THE TRACES
    for s in range(detections['waveforms'].shape[1]):
        for c in range(detections['waveforms'].shape[2]):
            mad = np.median(np.abs(detections['waveforms'][:,s,c,:] - np.median(detections['waveforms'][:,s,c,:])))
            if mad == 0.:
                continue
            detections['waveforms'][:,s,c,:] /= mad
    #-----------------------------------------
    features_1day = np.concatenate( (feature_max_kurto(detections),
                                     feature_peaks_autocorr(detections),
                                     features_max_amp(detections)) , axis=-1)
    features.append(features_1day)
    days.append([day] * features_1day.shape[0])
    filenames.append([filename] * features_1day.shape[0])
    indexes.append(np.arange(features_1day.shape[0]))

array_features    = np.array(features[0], copy=True)
array_days        = np.array(days[0], copy=True).astype('|S10')
array_indexes     = np.array(indexes[0], copy=True)
for i in range(1, len(features)):
    array_features = np.concatenate( (array_features, 
                                      features[i]), axis=0)
    array_days = np.concatenate( (array_days, 
                                  np.asarray(days[i]).astype('|S10')), axis=0)
    array_indexes = np.concatenate( (array_indexes,
                                     indexes[i]), axis=0)
array_features[np.isnan(array_features)] = 0.

features_dataset = {}
features_dataset['features']         = array_features
features_dataset['days']             = array_days
features_dataset['indexes']          = array_indexes
path = autodet.cfg.base + 'classification/'
with h5.File(path + 'features_dataset.h5', mode='w') as f:
    for item in list(features_dataset.keys()):
        f.create_dataset(item, data=features_dataset[item], compression='gzip')

