import automatic_detection as autodet
import sys
sys.path.append(autodet.cfg.base + 'libraries/')
import my_library as persolib
import picker_lib as pl
import numpy as np
from scipy.stats import kurtosis
from scipy.signal import hilbert
from scipy.interpolate import interp1d
import time
import matplotlib.pyplot as plt
import os
from scipy import optimize
from scipy.ndimage.filters import gaussian_filter1d
from obspy import Trace
from scipy.signal import wiener
from matplotlib.colors import Normalize
from matplotlib.cm import ScalarMappable
import matplotlib.colorbar as clb
from obspy.geodetics.base import calc_vincenty_inverse
from mpl_toolkits.mplot3d import Axes3D
from cartopy.crs import PlateCarree
from mpl_toolkits.axes_grid1 import make_axes_locatable

font = {'family' : 'serif', 'size' : 20}
plt.rc('font', **font)
plt.rcParams['pdf.fonttype'] = 42 #TrueType
plt.rcParams.update({'ytick.labelsize'  :  15})

device='cpu'

figsize = (28, 17)

#===================================================================
#============= PROCESS ID: VERSION / STEP / SUBPART ================
type_thrs = 'RMS'

db_path_T          = 'template_db_1/'
db_path_T_to_write = 'template_db_2/'
db_path_M          = 'matched_filter_1/'

# read the database of template events
templates = autodet.db_h5py.read_template_list('database_index', db_path=autodet.cfg.dbpath+db_path_T)
tids      = []
n_templates = len(templates)
for t in range(n_templates):
    tids.append(templates[t].metadata['template_idx'])
tids = np.int32(tids)

n_templates = tids.size

Nparts = 13 # to run the code on Nparts different nodes
L_1part = n_templates // Nparts + 1
try:
    idx_part = int(sys.argv[1])
    id1 = (idx_part-1) * L_1part
    id2 = idx_part * L_1part
    if id2 > n_templates:
        id2 = n_templates
    tids = tids[id1:id2]
except:
    idx_part = -1
    tids = tids[::-1]

print("Relocation part {:d} / {:d}".format(idx_part, Nparts))

band = [autodet.cfg.min_freq, autodet.cfg.max_freq]

#===================================================================
#===================================================================
W = 2. #temporal window for the sliding kurtosis
gaussian_width = 10. #temporal width (in samples) of the smoothing window

def baseline(X, w):
    n_windows = np.int32(np.ceil(X.size/w))
    minima      = np.zeros(n_windows, dtype=X.dtype)
    minima_args = np.zeros(n_windows, dtype=np.int32)
    for i in range(n_windows):
        minima_args[i] = i*w + X[i*w:(i+1)*w].argmin()
        minima[i]      = X[minima_args[i]]
    #----------------------------------------
    #--------- build interpolation ----------
    interpolator = interp1d(minima_args, minima, kind='linear', fill_value='extrapolate')
    bline = interpolator(np.arange(X.size))
    return bline

def dist_az(lat_source, long_source, depth_source, lat_station, long_station, depth_station):
    """
    dist_az(lat_source, long_source, lat_station, long_station)\n
    Compute the epicentral distance and the azimuth from source to station.
    """
    dist, az, baz = calc_vincenty_inverse(lat_source, long_source, lat_station, long_station)
    dist /= 1000.
    dist = np.sqrt(dist**2 + (depth_station - depth_source)**2)
    return dist

def get_delta_r(NR, source_indexes_1D):
    #---------------------
    relevant_interval = get_half_peak_interval(NR)
    #---------------------
    # generate probability distribution from the composite network response
    nr_max = NR.max()
    sigma   = np.std(NR[relevant_interval])
    rho  = np.exp(-(nr_max - NR[relevant_interval])**2/(4.*sigma))
    rho /= rho.sum()
    #---------------------
    mean_delta_r = 0.
    coords_max = np.array([ MV.latitude[ source_indexes_1D[NR.argmax()]], \
                            MV.longitude[source_indexes_1D[NR.argmax()]], \
                            MV.depth[    source_indexes_1D[NR.argmax()]]])
    for i, idx in enumerate(relevant_interval):
        coords_n = np.array([ MV.latitude[ source_indexes_1D[idx]], \
                              MV.longitude[source_indexes_1D[idx]], \
                              MV.depth[    source_indexes_1D[idx]]])
        mean_delta_r += dist_az(coords_max[0], coords_max[1], coords_max[2], coords_n[0], coords_n[1], coords_n[2]) * rho[i]
    #---------------------
    return rho, mean_delta_r

def get_half_peak_interval(NR, plot=False):
    idx_max  = NR.argmax()
    #threshold = (NR.max() + np.median(NR))/2.
    threshold = 3./4. * NR.max()
    idx_left = np.int32(idx_max)
    nr_left  = np.float32(NR.max())
    while nr_left >= threshold:
        idx_left -= 1
        nr_left   = NR[idx_left]
        if idx_left == 0:
            break
    #----------------------
    idx_right = np.int32(idx_max)
    nr_right  = np.float32(NR.max())
    while nr_right >= threshold:
        idx_right += 1
        nr_right   = NR[idx_right]
        if idx_right == NR.size-1:
            break
    interval = np.arange(idx_left, idx_right)
    if plot:
        plt.figure('threshold')
        plt.plot(NR)
        plt.plot(interval, NR[interval], color='C3')
        plt.axhline(threshold, lw=2, ls='--', color='k')
        plt.subplots_adjust(top=0.88,
                bottom=0.3,
                left=0.5,
                right=0.9,
                hspace=0.2,
                wspace=0.2)
        plt.show()
    return interval

def plot_NR_3D(NR, source_indexes_1D):
    n_samples = NR.size
    #---------------------
    cNorm = Normalize(vmin=NR.min(), vmax=NR.max())
    cmap  = plt.get_cmap('jet')
    scalarMap = ScalarMappable(norm=cNorm, cmap=cmap)
    #---------------------
    relevant_interval = get_half_peak_interval(NR)
    rho, mean_delta_r = get_delta_r(NR, source_indexes_1D)
    #---------------------
    time = np.linspace(0., NR.size/autodet.cfg.sampling_rate, NR.size)
    fig = plt.figure('weighting_function', figsize=figsize)
    ax1 = fig.add_subplot(111)
    ax1.plot(time, NR)
    ax1.tick_params('y', color='C0', labelcolor='C0')
    ax1.set_ylabel('Composite Network Response', color='C0')
    ax2 = ax1.twinx()
    ax2.plot(time[relevant_interval], rho, color='C3')
    ax2.set_ylabel('Weighting Function', color='C3')
    ax2.tick_params('y', color='C3', labelcolor='C3')
    plt.subplots_adjust(top=0.88,
            bottom=0.3,
            left=0.5,
            right=0.9,
            hspace=0.2,
            wspace=0.2)
    plt.xlim(time.min(), time.max())
    ax1.set_xlabel('Time (s)')
    #---------------------
    colors = scalarMap.to_rgba(NR)
    colors[np.setdiff1d(np.arange(NR.size), relevant_interval), -1] = 0.05
    #---------------------
    #    PROJECTION
    fake_map = plt.figure()
    fake_map_axis = plt.axes(projection = PlateCarree())
    X = np.zeros(n_samples, dtype=np.float64)
    Y = np.zeros(n_samples, dtype=np.float64)
    for n in range(n_samples):
        X[n], Y[n] = fake_map_axis.transLimits.transform((MV.longitude[source_indexes_1D[n]], MV.latitude[source_indexes_1D[n]]))
    plt.close(fig=fake_map)
    #---------------------
    fig = plt.figure('NR_map', figsize=figsize)
    ax = fig.add_subplot(111, projection='3d')
    Z = MV.depth[source_indexes_1D]
    ax.scatter(X, Y, zs=Z, c=colors)
    ax.set_xlabel('Longitude')
    ax.set_ylabel('Latitude')
    ax.set_zlabel('Depth (km)')
    ax.set_xlim(X.min(), X.max())
    ax.set_ylim(Y.min(), Y.max())
    ax.set_zlim(Z.min(), Z.max())
    ax.invert_zaxis()
    #---------------------
    ax2, _ = clb.make_axes(plt.gca(), shrink=0.8, orientation='vertical', pad=0.15, aspect=40, anchor=(1.1,0.75))
    cbmin = cNorm.vmin
    cbmax = cNorm.vmax
    ticks_pos = np.arange(np.round(cbmin, decimals=1), np.round(cbmax, decimals=1), 10.)
    cbar = clb.ColorbarBase(ax2, cmap = cmap, norm=cNorm, \
                            label='Composite Network Response', orientation='vertical', \
                            boundaries=np.linspace(cbmin, cbmax, 100), \
                            ticks=ticks_pos)
    ax.set_title(r'Location uncertainty $\Delta r$ = {:.2f}km'.format(mean_delta_r))
    #---------------------

def plot_NR_2D(NR, source_indexes_1D):
    n_samples = NR.size
    #---------------------
    cNorm = Normalize(vmin=NR.min(), vmax=NR.max())
    cmap  = plt.get_cmap('jet')
    scalarMap = ScalarMappable(norm=cNorm, cmap=cmap)
    #---------------------
    relevant_interval = get_half_peak_interval(NR)
    rho, mean_delta_r = get_delta_r(NR, source_indexes_1D)
    #---------------------
    time = np.linspace(0., NR.size/autodet.cfg.sampling_rate, NR.size)
    #---------------------
    colors = scalarMap.to_rgba(NR)
    colors[np.setdiff1d(np.arange(NR.size), relevant_interval), -1] = 0.05
    #---------------------
    #    PROJECTION
    fake_map = plt.figure()
    fake_map_axis = plt.axes(projection = PlateCarree())
    X = np.zeros(n_samples, dtype=np.float64)
    Y = np.zeros(n_samples, dtype=np.float64)
    for n in range(n_samples):
        X[n], Y[n] = fake_map_axis.transLimits.transform((MV.longitude[source_indexes_1D[n]], MV.latitude[source_indexes_1D[n]]))
    Z = MV.depth[source_indexes_1D]
    plt.close(fig=fake_map)
    #---------------------
    fig = plt.figure('NR_cross_sections', figsize=figsize)
    ax1 = plt.subplot(2, 1, 1)
    ax1.set_title(r'Location uncertainty $\Delta r$ = {:.2f}km'.format(mean_delta_r))
    ax1.scatter(X, Z, c=colors)
    ax1.set_xlabel('Longitude')
    ax1.set_ylabel('Depth (km)')
    ax1.set_xlim(X.min(), X.max())
    ax1.set_ylim(Z.min(), Z.max())
    ax1.invert_yaxis()
    #---------------------
    divider = make_axes_locatable(ax1)
    cax     = divider.append_axes('right', size='2%', pad=0.08)
    cbar    = clb.ColorbarBase(cax, cmap = cmap, norm=cNorm, label='Composite Network Response', orientation='vertical')
    #---------------------
    ax2 = plt.subplot(2, 1, 2)
    ax2.scatter(Y, Z, c=colors)
    ax2.set_xlabel('Latitude')
    ax2.set_ylabel('Depth (km)')
    ax2.set_xlim(Y.min(), Y.max())
    ax2.set_ylim(Z.min(), Z.max())
    ax2.invert_yaxis()
    #---------------------
    divider = make_axes_locatable(ax2)
    cax     = divider.append_axes('right', size='2%', pad=0.08)
    cbar    = clb.ColorbarBase(cax, cmap = cmap, norm=cNorm, label='Composite Network Response', orientation='vertical')
    #---------------------

def reloc(tid, I, relocation_traces, svd, test_points, plot=False):
    """
    reloc(tid, I, relocation_traces, plot=False) \n
    Relocation function: computes the modified composite network response on
    the stack of the sliding kurtosis.
    """
    moveoutsP = MV.p_relative_samp[:,I]
    moveoutsS = MV.s_relative_samp[:,I]
    window      = np.int32(0.1 * 60. * autodet.cfg.sampling_rate)
    buffer_length = int(window/10)
    ##-----------------------------
    composite, where = autodet.clib.network_response_SP(np.mean(relocation_traces[I,:-1,:], axis=1), \
                                                                relocation_traces[I,-1,:], \
                                                                moveoutsP, \
                                                                moveoutsS, \
                                                                10, \
                                                                device=device, \
                                                                test_points=test_points)
    composite /= np.float32(I.size)
    smoothed    = gaussian_filter1d(composite, np.int32(0.2 * autodet.cfg.sampling_rate))
    base = np.hstack( (baseline(composite[:window], int(window/20)), baseline(composite[window:], window)) )
    comp_baseline = composite - base
    detection_trace = smoothed
    Q = detection_trace.max() / np.sqrt(np.var(detection_trace)) # quality factor of the relocation
    T0 = detection_trace[buffer_length:].argmax() + buffer_length
    mv_min = moveoutsP[where[T0],:].min()
    I2 = I[:12]
    #==================================================
    moveoutsP = MV.p_relative_samp[:,I2]
    moveoutsS = MV.s_relative_samp[:,I2]
    ##-----------------------------
    composite2, where2 = autodet.clib.network_response_SP(np.mean(relocation_traces[I2,:-1,:], axis=1), \
                                                                  relocation_traces[I2,-1,:], \
                                                                  moveoutsP, \
                                                                  moveoutsS, \
                                                                  10, \
                                                                  device=device, \
                                                                  test_points=test_points)
    composite2 /= np.float32(I2.size)
    smoothed2   = gaussian_filter1d(composite2, np.int32(0.2 * autodet.cfg.sampling_rate))
    base2 = np.hstack( (baseline(composite2[:window], int(window/20)), baseline(composite2[window:], window)) )
    comp_baseline2 = composite2 - base2
    detection_trace = smoothed2
    Q2 = detection_trace.max() / np.sqrt(np.var(detection_trace)) # quality factor of the relocation
    print("Reloc with 20 stations: Q={:.2f} (NR max={:.2f}) / with 12 stations: Q={:.2f} (NR max={:.2f})".format(Q, smoothed.max(), Q2, smoothed2.max()))
    T0 = detection_trace[buffer_length:].argmax() + buffer_length
    mv_min2 = moveoutsP[where2[T0],:].min()
    if Q2 > Q:
        Q = Q2
        composite = composite2
        smoothed = smoothed2
        base     = base2
        comp_baseline = comp_baseline2
        where = where2
        mv_min = mv_min2
        I = I2
    #detection_trace = comp_baseline
    detection_trace = smoothed
    #==================================================
    T0 = detection_trace[buffer_length:].argmax() + buffer_length
    source_index = where[T0]
    mvP = moveoutsP[source_index,:]
    mvS = moveoutsS[source_index,:]
    t_min = MV.p_relative[source_index,I].min() #useful information to determine absolute origin times
    I   = I2 # take only the 12 best stations to create the template
    #==================================================
    n_samples = composite.size
    #---------------------
    rho, mean_delta_r = get_delta_r(composite, where)
    #---------------------
    if plot:
        plt.figure('NR_tp{:d}'.format(tid), figsize=figsize)
        plt.plot(composite, label='CNR')
        plt.plot(smoothed, label='Smoothed CNR')
        plt.plot(comp_baseline, label='CNR without baseline')
        plt.plot(base, ls='--', label='Baseline approximation')
        plt.axvline(T0, color='k')
        plt.legend(loc='upper right', fancybox=True)
        plt.figure('reloc_tp{:d}'.format(tid), figsize=figsize)
        time = np.arange(0., svd[0].data.size/autodet.cfg.sampling_rate, 1./autodet.cfg.sampling_rate)
        for s in range(I.size):
            for c in range(nc):
                plt.subplot(I.size,nc,s*nc+c+1)
                svd_data = svd.select(station=svd.stations[I[s]])[c].data
                plt.plot(time, svd_data/svd_data.max()*K[I[s],c,:].max(), lw=0.5, label='{}.{}\n{:.2f}'.format(svd.stations[I[s]], net.components[c], SNR[I[s]]))
                plt.plot(time, K[I[s],c,:])
                plt.axvline((T0 + mvP[s] - mv_min)/autodet.cfg.sampling_rate, lw=1, color='k')
                plt.axvline((T0 + mvS[s] - mv_min)/autodet.cfg.sampling_rate, lw=1, color='r')
                if s != I.size-1:
                    plt.xticks([])
                else:
                    plt.xlabel('Time (s)')
                plt.xlim(time.min(), time.max())
                plt.legend(loc='upper right', fancybox=True, framealpha=0.7, handlelength=0.1)
        plt.subplots_adjust(
                top=0.95,
                bottom=0.05,
                left=0.02,
                right=0.98,
                hspace=0.0,
                wspace=0.1)
        plot_NR_3D(composite, where)
        plt.show()
    #-----------------------------------------------------------------------
    #                 attach SNR
    n_samples = np.int32(autodet.cfg.template_len * autodet.cfg.sampling_rate)
    n_stations = MV.s_relative.shape[-1]
    SNR_ = np.zeros(n_stations, dtype=np.float32)
    for s in range(n_stations):
        for c in range(nc):
            trace_sc = svd.select(station=net.stations[s])[c].data
            var = np.var(trace_sc)
            if var != 0.:
                id1_P = T0 + MV.p_relative_samp[source_index,s] - mv_min - n_samples//2
                id1_S = T0 + MV.s_relative_samp[source_index,s] - mv_min - n_samples//2
                if np.isnan(np.var(trace_sc[id1_S:id1_S+n_samples])):
                    continue
                if id1_S+n_samples > trace_sc.size:
                    continue
                snr = 0.
                snr += np.var(trace_sc[id1_P:id1_P+n_samples])/var
                snr += np.var(trace_sc[id1_S:id1_S+n_samples])/var
                SNR_[s] += snr
    #-----------------------------------------------------------------------
    metadata = {}
    metadata.update({'latitude'                 :     np.float32( MV.latitude[source_index])})
    metadata.update({'longitude'                :     np.float32(MV.longitude[source_index])})
    metadata.update({'depth'                    :     np.float32(    MV.depth[source_index])})
    metadata.update({'source_idx'               :     np.int32(source_index)})
    metadata.update({'template_idx'             :     np.int32(tid)})
    metadata.update({'Q_loc'                    :     np.float32(Q)})
    metadata.update({'loc_uncertainty'          :     np.float32(mean_delta_r)})
    metadata.update({'peak_NR'                  :     np.float32(smoothed.max())})
    metadata.update({'p_moveouts'               :     np.float32(mvP-mv_min) / autodet.cfg.sampling_rate})
    metadata.update({'s_moveouts'               :     np.float32(mvS-mv_min) / autodet.cfg.sampling_rate})
    metadata.update({'duration'                 :     np.int32(autodet.cfg.template_len*autodet.cfg.sampling_rate)})
    metadata.update({'sampling_rate'            :     np.float32(autodet.cfg.sampling_rate)})
    metadata.update({'stations'                 :     np.asarray(net.stations)[I]})
    metadata.update({'channels'                 :     np.asarray(['HHN', 'HHE', 'HHZ'])})
    metadata.update({'reference_absolute_time'  :     np.float32(t_min)})
    metadata.update({'travel_times'             :     np.hstack((MV.p_relative[source_index,:].reshape(-1,1), MV.s_relative[source_index,:].reshape(-1,1)))})
    metadata.update({'SNR'                      :     SNR})
    T = autodet.db_h5py.initialize_template(metadata)
    waveforms = np.zeros((I.size, nc, metadata['duration']), dtype=np.float32)
    time_before_S = np.int32(metadata['duration']/2)
    time_before_P = np.int32(1. * autodet.cfg.sampling_rate)
    size_stack = svd[0].data.size
    for s in range(I.size):
        for c in range(nc):
            data = svd.select(station=svd.stations[I[s]])[c].data
            MAX = np.abs(data).max()
            if MAX != 0.:
                data /= MAX
            if c < 2:
                id1 = T0 + mvS[s] - mv_min - time_before_S
            else:
                # take only 1sec before the P-wave arrival
                id1 = T0 + mvP[s] - mv_min - time_before_P
            id2 = id1 + metadata['duration']
            if id1 > size_stack:
                continue
            elif id2 > size_stack:
                DN = id2 - size_stack
                waveforms[s,c,:] = np.hstack( (data[id1:], np.zeros(DN, dtype=np.float32)) )
            elif id1 < 0:
                DN = 0 - id1
                waveforms[s,c,:] = np.hstack( (np.zeros(DN, dtype=np.float32), data[:id2]) )
            else:
                waveforms[s,c,:] = data[id1:id2]
            T.select(station=svd.stations[I[s]])[c].data = waveforms[s,c,:]
    T.waveforms = waveforms
    #=========================================================
    #=========================================================
    T.metadata['p_moveouts'] += (time_before_S - time_before_P)/autodet.cfg.sampling_rate # !!!!!!!!!! because the windows are not centered around the P- and S-wave in the same way !!!!!!!!!!!!!
    #=========================================================
    #=========================================================
    T.metadata['stations'] = T.metadata['stations'].astype('S')
    T.metadata['channels'] = T.metadata['channels'].astype('S')
    return T, composite, where

net = autodet.dataset.Network('network.in')
net.read()

filename = 'subgrid'

MV = autodet.moveouts.MV_object(filename, net, \
                                relative=False, \
                                remove_airquakes=True)

test_points = np.arange(MV.n_sources, dtype=np.int32)
test_points = test_points[MV.idx_EQ] # remove the airquakes

tp_to_check = []

#========================================================
#=================== LOOP ON THE TEMPLATES ==============
for tid in tids:
    #if os.path.isfile(autodet.cfg.dbpath+db_path_T_to_write+'template{:d}meta.h5'.format(tid)):
    #    continue
    print("===================================")
    print("===== Relocating template {:d} ... =====".format(tid))
    
    svd = persolib.SVDWF_multiplets(tid, db_path=autodet.cfg.dbpath, db_path_T=db_path_T, db_path_M=db_path_M, best=True, normRMS=True)
    if svd is None:
        print("Problem with template {:d}, add to the check list !".format(tid))
        tp_to_check.append(tid)
        continue
    nm = svd.data.shape[0]
    ns = svd.data.shape[1]
    nc = svd.data.shape[2]

    K = autodet.clib.kurtosis(np.mean(svd.data, axis=0), W*autodet.cfg.sampling_rate)
    for s in range(ns):
        for c in range(nc):
            K[s,c,:] = pl.F2(K[s,c,:])
            K[s,c,:] = pl.F3(K[s,c,:])
            K[s,c,:] = pl.F4(K[s,c,:])
            K[s,c,:] = pl.F4prim(K[s,c,:])
            mad = np.median(np.abs(K[s,c,:] - np.median(K[s,c,:])))
            if mad != 0.:
                K[s,c,:] /= mad

    #---------------  ----------------
    for s in range(ns):
        for c in range(nc):
            K[s,c,:] = np.abs(gaussian_filter1d(K[s,c,:], gaussian_width))
            M = K[s,c,:].max()
            if M != 0.:
                K[s,c,:] /= M

    #----------------------------------------
    SNR = np.zeros((ns,nc), dtype=np.float32)
    for s in range(ns):
        for c in range(nc):
            data = svd.select(station=svd.stations[s], channel=net.components[c])[0].data
            mad  = np.round(np.median(np.abs(data - np.median(data))), decimals=5)
            if mad != 0 :
                SNR[s,c] = np.percentile(np.abs(data), 95.) / mad
    SNR[np.isnan(SNR)] = 0.
    SNR = np.median(SNR, axis=-1)
    SNR[np.isnan(SNR)] = 0.
    I = np.argsort(SNR)[::-1][:20]

    relocation_traces = np.array(K, copy=True)
    for s in range(ns):
        relocation_traces[s,:,:] *= SNR[s] # scale with SNR so that best stations are heavier in the network response
    
    T, composite, where = reloc(tid, I, relocation_traces, svd, test_points, plot=True)
    print("New location: {:.2f}/{:.2f}/{:.2f}km".format(T.metadata['latitude'],\
                                                        T.metadata['longitude'],\
                                                        T.metadata['depth']))
    autodet.db_h5py.write_template('template{:d}'.format(tid), T, T.waveforms, db_path=autodet.cfg.dbpath+db_path_T_to_write)

    #-------------- FREE MEMORY -------------------
    del svd

with open(autodet.cfg.dbpath + db_path_T_to_write + 'problem_relocalization_part{:d}.txt'.format(idx_part), 'w') as f:
    for i in range(len(tp_to_check)):
        f.write('template{:d}\n'.format(tp_to_check[i]))
