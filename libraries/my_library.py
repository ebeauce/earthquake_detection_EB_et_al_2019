# -*- coding: utf-8 -*-
import numpy as np
import matplotlib.pyplot as plt
import sys
sys.path.append('/nobackup1/ebeauce/automatic_detection/')
import automatic_detection as autodet
import h5py as h5
from obspy import UTCDateTime as udt
from os.path import isfile
from scipy.signal import hilbert, tukey
from scipy.ndimage.filters import gaussian_filter1d
from scipy.interpolate import interp1d
import scipy.linalg as scilin
import scipy.signal as scisig

# debugger
from IPython.core.debugger import Tracer
debug_here = Tracer()

def SNR(st, stations, components, mv=None, T=None):
    """
    SNR(st, stations, components, mv=None, T=None) \n
    if mv is given, the origin time T has also to be given
    """
    import operator
    ns = len(stations)
    nc = len(components)
    #==============================
    # == SORT BY SNR ==
    SNR_c = np.zeros((ns,nc), dtype=np.float32)
    SNR = np.zeros(ns, dtype=np.float32)
    SNR_dic = {}
    for s in range(ns):
        for c in range(nc):
            if mv is None:
                data = st.select(station=stations[s])[c].data
            else:
                id1 = max(0, T + mv[s] - np.int32(autodet.cfg.template_len/2 * autodet.cfg.sampling_rate))
                id2 = min(st.select(station=stations[s])[c].data.size, T + mv[s] + np.int32(autodet.cfg.template_len/2 * autodet.cfg.sampling_rate))
                if id2-id1 <= 0:
                    data = np.float32([0.])
                else:
                    data = st.select(station=stations[s])[c].data[id1:id2]
            if np.var(data) != 0.:
                SNR_c[s,c] = np.power(data, 2).max()/np.var(data)
            else:
                pass
        SNR[s] = np.mean(SNR_c[s,c])
        SNR_dic.update({stations[s]:SNR[s]})
    SNR_sorted = sorted(SNR_dic.items(), key=operator.itemgetter(1))
    SNR_sorted.reverse()
    return SNR_sorted

def get_all_waveforms(tid, db_path_M='matched_filter_1/', db_path=autodet.cfg.dbpath):
    cat = autodet.db_h5py.read_catalog_multiplets('multiplets{:d}'.format(tid), db_path_M=db_path_M)
    n_detections = cat['origin_times'].size
    n_stations   = cat['stations'].size
    n_components = cat['components'].size
    #-----------------------------------------------------
    waveforms    = np.zeros((n_detections, n_stations, n_components, np.int32(autodet.cfg.multiplet_len * autodet.cfg.sampling_rate)), dtype=np.float32)
    #-----------------------------------------------------
    n = 0
    filename0 = ''
    for i, filename in enumerate(cat['filenames'].astype('U')):
        if filename != filename0:
            try:
                f.close()
            except:
                pass
            f = h5.File(db_path + db_path_M + filename + 'wav.h5', mode='r')
        waveforms[i,:,:,:] = f['waveforms'][cat['indices'][i],:,:,:]
        filename0 = str(filename)
    return waveforms
    

def spectral_filtering_detections(tid, db_path_T='template_db_1/', db_path_M='matched_filter_1/', db_path=autodet.cfg.dbpath, SNR_thres=5., WAVEFORMS=None, normRMS=True, best=True):
    from subprocess import Popen, PIPE
    from obspy import Stream, Trace
    #-----------------------------------------------------------------------------------------------
    T = autodet.db_h5py.read_template('template{:d}'.format(tid), db_path=db_path+db_path_T)
    #-----------------------------------------------------------------------------------------------
    print("Looking for {}{:d}_*".format(db_path + db_path_M + '*multiplets', tid))
    files_list = Popen('ls '+db_path+db_path_M+'*multiplets{:d}_*'.format(tid), stdout=PIPE, shell=True).stdout
    line  = files_list.readline()[:-1]
    files = []
    while len(line) != 0:
        files.append(line.decode('utf-8'))
        line = files_list.readline()[:-1]
    i = 0
    Nsamp = 0
    ns = 0
    S = Stream()
    #------------- retrieve metadata ---------------
    while True:
        try:
            wav = files[i][-len('wav.h5'):] == 'wav.h5'
            if wav:
                if Nsamp == 0:
                    with h5.File(files[i], mode='r') as fwav0:
                        Nsamp = fwav0['waveforms'][:,:,:,:].shape[-1]
                i += 1
            else:
                with h5.File(files[i], mode='r') as fm0:
                    if len(fm0['origin_times']) == 0:
                        i += 1
                        continue
                    else:
                        i += 1
                        nc = len(fm0['components'][()])
                        ns = len(fm0['stations'][()])
                        S.stations   = fm0['stations'][()].astype('U').tolist()
                        S.components = fm0['components'][()].astype('U').tolist()
                        S.latitude   = T.metadata['latitude']
                        S.longitude  = T.metadata['longitude']
                        S.depth      = T.metadata['depth']
                        S.template_ID = tid
            if ns != 0 and Nsamp != 0:
                break
        except IndexError:
            print("None multiplet for template {:d} !! Return None".format(tid))
            return None
    #----------------------------------------------
    if WAVEFORMS is None:
        CC = np.zeros(0, dtype=np.float32)
        if best:
            for file in files:
                if file[-len('meta.h5'):] != 'meta.h5':
                    continue
                with h5.File(file, mode='r') as fm:
                    if len(fm['correlation_coefficients']) == 0:
                        continue
                    else:
                        CC = np.hstack((CC, fm['correlation_coefficients'][:]))
            CC = np.sort(CC)
            #CC_thres = np.sort(CC)[-min(5, len(CC))]
            if len(CC) > 300:
                CC_thres = CC[-101] 
            elif len(CC) > 70:
                CC_thres = CC[int(7./10.*len(CC))] # the best 30%
            elif len(CC) > 30:
                CC_thres = np.median(CC) # the best 50%
            elif len(CC) > 10:
                CC_thres = np.percentile(CC, 33.) # the best 66% detections 
            else:
                CC_thres = 0.
        Nstack = np.zeros((ns, nc), dtype=np.float32)
        WAVEFORMS  = np.zeros((0,ns,nc,Nsamp), dtype=np.float32)
        Nmulti = 0
        for file in files:
            if file[-len('wav.h5'):] != 'wav.h5':
                continue
            with h5.File(file, mode='r') as fw:
                if len(fw['waveforms']) == 0:
                    continue
                else:
                    if best:
                        with h5.File(file[:-len('wav.h5')]+'meta.h5', mode='r') as fm:
                            selection = np.where(fm['correlation_coefficients'][:] > CC_thres)[0]
                            if selection.size == 0:
                                continue
                            waves = np.zeros((selection.size, ns, nc, Nsamp), dtype=np.float32)
                            waves[:,:,:,:] = fw['waveforms'][selection,:,:,:]
                    else:
                        waves = fw['waveforms'][:,:,:,:]
                    Nmulti += waves.shape[0]
                    for m in range(waves.shape[0]):
                        for s in range(ns):
                            for c in range(nc):
                                if normRMS:
                                    norm = np.sqrt(np.var(waves[m,s,c,:]))
                                else:
                                    norm =1.
                                if norm != 0.:
                                    waves[m,s,c,:] /= norm
                    WAVEFORMS = np.vstack((WAVEFORMS, waves))
    elif normRMS:
        for m in range(WAVEFORMS.shape[0]):
            for s in range(ns):
                for c in range(nc):
                    norm = np.sqrt(np.var(WAVEFORMS[m,s,c,:]))
                    if norm != 0.:
                        WAVEFORMS[m,s,c,:] /= norm
    else:
        pass
    filtered_waveforms = np.zeros((ns, nc, Nsamp), dtype=np.float32)
    for s in range(ns):
        for c in range(nc):
            filtered_waveforms[s, c, :] = np.sum(spectral_filtering(WAVEFORMS[:, s, c, :], SNR_thres=SNR_thres), axis=0)
    return filtered_waveforms

def SVDWF_multiplets_test(template_id, db_path=autodet.cfg.dbpath, db_path_M='matched_filter_2/', db_path_T='template_db_2/', WAVEFORMS=None, normRMS=True, Nsing_values=5, max_freq=autodet.cfg.max_freq, attach_raw_data=False):
    from subprocess import Popen, PIPE
    from obspy import Stream, Trace
    from scipy.linalg import svd
    from scipy.signal import wiener
    #-----------------------------------------------------------------------------------------------
    T = autodet.db_h5py.read_template('template{:d}'.format(template_id), db_path=db_path+db_path_T)
    #-----------------------------------------------------------------------------------------------
    print("Looking for {}{:d}_*".format(db_path + db_path_M + '*multiplets', template_id))
    files_list = Popen('ls '+db_path+db_path_M+'*multiplets{:d}_*'.format(template_id), stdout=PIPE, shell=True).stdout
    line  = files_list.readline()[:-1]
    files = []
    while len(line) != 0:
        files.append(line.decode('utf-8'))
        line = files_list.readline()[:-1]
    i = 0
    Nsamp = 0
    ns = 0
    S = Stream()
    #------------- retrieve metadata ---------------
    while True:
        try:
            wav = files[i][-len('wav.h5'):] == 'wav.h5'
            if wav:
                if Nsamp == 0:
                    with h5.File(files[i], mode='r') as fwav0:
                        Nsamp = fwav0['waveforms'][:,:,:,:].shape[-1]
                i += 1
            else:
                with h5.File(files[i], mode='r') as fm0:
                    if len(fm0['origin_times']) == 0:
                        i += 1
                        continue
                    else:
                        i += 1
                        nc = len(fm0['components'][()])
                        ns = len(fm0['stations'][()])
                        S.stations   = fm0['stations'][()].astype('U').tolist()
                        S.components = fm0['components'][()].astype('U').tolist()
                        S.latitude   = T.metadata['latitude']
                        S.longitude  = T.metadata['longitude']
                        S.depth      = T.metadata['depth']
                        S.template_id = template_id
            if ns != 0 and Nsamp != 0:
                break
        except IndexError:
            print("None multiplet for template {:d} !! Return None".format(template_id))
            return None
    #----------------------------------------------
    catalog = autodet.db_h5py.read_catalog_multiplets('multiplets{:d}'.format(template_id), db_path_M=db_path_M, db_path=db_path)
    CC      = catalog['correlation_coefficients']
    best_detection_indexes = np.argsort(CC)[::-1]
    if CC.size > 300:
        best_detection_indexes = best_detection_indexes[:100]                     # the best 100 detections
    elif CC.size > 100:
        best_detection_indexes = best_detection_indexes[:int(30./100. * CC.size)] # the best 30%
    elif CC.size > 50:
        best_detection_indexes = best_detection_indexes[:int(50./100. * CC.size)] # the best 50%
    elif CC>size > 10:
        best_detection_indexes = best_detection_indexes[:int(66./100. * CC.size)] # the best 66%
    else:
        pass # keep all detections
    # reorder by chronological order
    best_detection_indexes = best_detection_indexes[np.argsort(catalog['origin_times'][best_detection_indexes])]
    # get the waveforms
    n_events  = best_detection_indexes.size
    WAVEFORMS = np.zeros((n_events, ns, nc, Nsamp), dtype=np.float32)
    filename0 = db_path + db_path_M + catalog['filenames'][best_detection_indexes[0]].decode('utf-8')
    f         = h5.File(filename0 + 'wav.h5', mode='r')
    for n in range(n_events):
        filename = db_path + db_path_M + catalog['filenames'][best_detection_indexes[n]].decode('utf-8')
        if filename == filename0:
            pass
        else:
            f.close()
            f = h5.File(filename + 'wav.h5', mode='r')
        WAVEFORMS[n, :, :, :] = f['waveforms'][catalog['indices'][best_detection_indexes[n]], :, :, :]
        # normalization
        for s in range(ns):
            for c in range(nc):
                if normRMS:
                    norm = np.std(WAVEFORMS[n, s, c, :])
                else:
                    norm = np.abs(WAVEFORMS[n, s, c, :]).max()
                if norm != 0.:
                    WAVEFORMS[n, s, c, :] /= norm
    filtered_data = np.zeros((n_events, ns, nc, Nsamp), dtype=np.float32)
    for s in range(ns):
        for c in range(nc):
            filtered_data[:,s,c,:] = SVDWF(WAVEFORMS[:,s,c,:], Nsing_values, max_freq=max_freq)
            #filtered_data[:,s,c,:] = spectral_filtering(WAVEFORMS[:,s,c,:], SNR_thres=5., max_freq=max_freq)
            mean = np.mean(filtered_data[:,s,c,:], axis=0)
            mean /= np.abs(mean).max()
            S += Trace(data=mean)
            S[-1].stats.station = S.stations[s]
            S[-1].stats.channel = S.components[c]
            S[-1].stats.sampling_rate = autodet.cfg.sampling_rate
    S.data = filtered_data
    if attach_raw_data:
        S.raw_data = WAVEFORMS
    S.Nmulti = best_detection_indexes.size
    return S

def SVDWF_multiplets(template_ID, db_path=autodet.cfg.dbpath, db_path_M='matched_filter_1/', db_path_T='template_db_1/', WAVEFORMS=None, best=False, normRMS=True, Nsing_values=5, max_freq=autodet.cfg.max_freq, attach_raw_data=False):
    from subprocess import Popen, PIPE
    from obspy import Stream, Trace
    from scipy.linalg import svd
    from scipy.signal import wiener

    #-----------------------------------------------------------------------------------------------
    T = autodet.db_h5py.read_template('template{:d}'.format(template_ID), db_path=db_path+db_path_T)
    #-----------------------------------------------------------------------------------------------
    print("Looking for {}{:d}_*".format(db_path + db_path_M + '*multiplets', template_ID))
    files_list = Popen('ls '+db_path+db_path_M+'*multiplets{:d}_*'.format(template_ID), stdout=PIPE, shell=True).stdout
    line  = files_list.readline()[:-1]
    files = []
    while len(line) != 0:
        files.append(line.decode('utf-8'))
        line = files_list.readline()[:-1]
    i = 0
    Nsamp = 0
    ns = 0
    S = Stream()
    #------------- retrieve metadata ---------------
    while True:
        try:
            wav = files[i][-len('wav.h5'):] == 'wav.h5'
            if wav:
                if Nsamp == 0:
                    with h5.File(files[i], mode='r') as fwav0:
                        Nsamp = fwav0['waveforms'][:,:,:,:].shape[-1]
                i += 1
            else:
                with h5.File(files[i], mode='r') as fm0:
                    if len(fm0['origin_times']) == 0:
                        i += 1
                        continue
                    else:
                        i += 1
                        nc = len(fm0['components'][()])
                        ns = len(fm0['stations'][()])
                        S.stations   = fm0['stations'][()].astype('U').tolist()
                        S.components = fm0['components'][()].astype('U').tolist()
                        S.latitude   = T.metadata['latitude']
                        S.longitude  = T.metadata['longitude']
                        S.depth      = T.metadata['depth']
                        S.template_ID = template_ID
            if ns != 0 and Nsamp != 0:
                break
        except IndexError:
            print("None multiplet for template {:d} !! Return None".format(template_ID))
            return None
    #----------------------------------------------
    if WAVEFORMS is None:
        CC = np.zeros(0, dtype=np.float32)
        if best:
            for file in files:
                if file[-len('meta.h5'):] != 'meta.h5':
                    continue
                with h5.File(file, mode='r') as fm:
                    if len(fm['correlation_coefficients']) == 0:
                        continue
                    else:
                        CC = np.hstack((CC, fm['correlation_coefficients'][:]))
            CC = np.sort(CC)
            #CC_thres = np.sort(CC)[-min(5, len(CC))]
            if len(CC) > 300:
                CC_thres = CC[-101] 
            elif len(CC) > 70:
                CC_thres = CC[int(7./10.*len(CC))] # the best 30%
            elif len(CC) > 30:
                CC_thres = np.median(CC) # the best 50%
            elif len(CC) > 10:
                CC_thres = np.percentile(CC, 33.) # the best 66% detections 
            else:
                CC_thres = 0.
        Nstack = np.zeros((ns, nc), dtype=np.float32)
        WAVEFORMS  = np.zeros((0,ns,nc,Nsamp), dtype=np.float32)
        Nmulti = 0
        for file in files:
            if file[-len('wav.h5'):] != 'wav.h5':
                continue
            with h5.File(file, mode='r') as fw:
                if len(fw['waveforms']) == 0:
                    continue
                else:
                    if best:
                        with h5.File(file[:-len('wav.h5')]+'meta.h5', mode='r') as fm:
                            selection = np.where(fm['correlation_coefficients'][:] > CC_thres)[0]
                            if selection.size == 0:
                                continue
                            waves = np.zeros((selection.size, ns, nc, Nsamp), dtype=np.float32)
                            waves[:,:,:,:] = fw['waveforms'][selection,:,:,:]
                    else:
                        waves = fw['waveforms'][:,:,:,:]
                    Nmulti += waves.shape[0]
                    for m in range(waves.shape[0]):
                        for s in range(ns):
                            for c in range(nc):
                                if normRMS:
                                    norm = np.sqrt(np.var(waves[m,s,c,:]))
                                else:
                                    norm =1.
                                if norm != 0.:
                                    waves[m,s,c,:] /= norm
                    WAVEFORMS = np.vstack((WAVEFORMS, waves))
    elif normRMS:
        for m in range(WAVEFORMS.shape[0]):
            for s in range(ns):
                for c in range(nc):
                    norm = np.sqrt(np.var(WAVEFORMS[m,s,c,:]))
                    if norm != 0.:
                        WAVEFORMS[m,s,c,:] /= norm
    else:
        pass
    filtered_data = np.zeros((WAVEFORMS.shape[0], ns, nc, Nsamp), dtype=np.float32)
    for s in range(ns):
        for c in range(nc):
            filtered_data[:,s,c,:] = SVDWF(WAVEFORMS[:,s,c,:], Nsing_values, max_freq=max_freq)
            #filtered_data[:,s,c,:] = spectral_filtering(WAVEFORMS[:,s,c,:], SNR_thres=5., max_freq=max_freq)
            mean = np.mean(filtered_data[:,s,c,:], axis=0)
            mean /= np.abs(mean).max()
            S += Trace(data=mean)
            S[-1].stats.station = S.stations[s]
            S[-1].stats.channel = S.components[c]
            S[-1].stats.sampling_rate = autodet.cfg.sampling_rate
    S.data = filtered_data
    if attach_raw_data:
        S.raw_data = WAVEFORMS
    S.Nmulti = Nmulti
    return S

def SVDWF(matrix, N_singular_values, max_freq=autodet.cfg.max_freq):
    """
    SVDWF(matrix, N_singular_values, max_freq=autodet.cfg.max_freq)\n
    Implementation of the Singular Value Decomposition Wiener Filter (SVDWF) described
    in Moreau et al 2017.
    """
    U, S, Vt = scilin.svd(matrix, full_matrices=0)
    filtered_data = np.zeros((U.shape[0], Vt.shape[1]), dtype=np.float32)
    for n in range(min(U.shape[0], N_singular_values)):
        s_n = np.zeros(S.size, dtype=np.float32)
        s_n[n] = S[n]
        projection_n = np.dot(U, np.dot(np.diag(s_n), Vt))
        # the following application of Wiener filtering is questionable: because each projection in this loop is a projection
        # onto a vector space with one dimension, all the waveforms are colinear: they just differ by an amplitude factor (but same shape).
        filtered_projection = scisig.wiener(projection_n, mysize=[max(2, int(U.shape[0]/10)), int(autodet.cfg.sampling_rate/max_freq)])
        if np.isnan(filtered_projection.max()):
            continue
        filtered_data += filtered_projection
    filtered_data = scisig.wiener(filtered_data, mysize=[max(2, int(U.shape[0]/10)), int(autodet.cfg.sampling_rate/max_freq)])
    return filtered_data

def spectral_filtering(M, SNR_thres=5., max_freq=autodet.cfg.max_freq):
    """
    spectral_filtering(M, SNR_thres=5., max_freq=autodet.cfg.max_freq)\n
    Less fancy version of SVDWF. Try an adaptive threshold based on the SNR.
    """
    U, S, Vt = np.linalg.svd(M, full_matrices=False)
    #-----------------------------
    SNR = np.zeros(Vt.shape[0], dtype=np.float32)
    for p in range(Vt.shape[0]):
        mad = autodet.common.mad(Vt[p,:])
        if mad != 0.:
            SNR[p] = np.percentile(Vt[p,:], 95.) / mad
    SNR_thres = max(SNR_thres, SNR.max()/10.)
    if np.sum(SNR > SNR_thres) == 0:
        SNR_thres = 0.99*SNR.max()
    S_filtered = np.array(S, copy=True)
    S_filtered[SNR < SNR_thres] = 0.
    filtered_data = np.dot(np.dot(U, np.diag(S_filtered)), Vt)
    # in the original paper: apply a Wiener filter to each projection onto 
    # a single singular vector ===> does not make sense !!!!!
    # the projected matrix has rows that are perfectly coherent !
    filtered_data = scisig.wiener(filtered_data, mysize=[max(2, int(U.shape[0]/10)), int(autodet.cfg.sampling_rate/max_freq)])
    return filtered_data

def plot_template(idx, db_path_T='template_db_2/', db_path=autodet.cfg.dbpath, CC_comp=False, mv_view=True, show=True):
    font = {'family' : 'serif', 'weight' : 'normal', 'size' : 14}
    plt.rc('font', **font)
    template = autodet.db_h5py.read_template('template{:d}'.format(idx), db_path=db_path+db_path_T)
    if 'loc_uncertainty' not in template.metadata.keys():
        template.metadata['loc_uncertainty'] = 10000.
    sta = list(template.metadata['stations'])
    sta.sort()
    ns = len(sta)
    nc = template.metadata['channels'].size
    if CC_comp:
        CC = np.zeros(ns, dtype=np.float32)
        from scipy.signal import hilbert
        for s in range(ns):
            H = []
            num = np.ones(template.select(station=sta[s])[0].data.size, dtype=np.float32)
            den = 1.
            for c in range(nc):
                H.append(np.abs(hilbert(template.select(station=sta[s])[c].data)))
                if np.var(H[-1]) == 0.:
                    H[-1] = np.ones(len(H[-1]), dtype=np.float32)
                num *= H[-1]
                den *= np.power(H[-1], 3).sum()
            num = num.sum()
            den = np.power(den, 1./3.)
            if den != 0.:
                CC[s] = num/den
    plt.figure('template_{:d}_from_{}'.format(idx, db_path+db_path_T), figsize=(20,12))
    if mv_view:
        data_availability = np.zeros(ns, dtype=np.bool)
        for s in range(ns):
            sig = 0.
            for tr in template.select(station=template.metadata['stations'][s]):
                sig += np.var(tr.data)
            if np.isnan(sig):
                data_availability[s] = False
            else:
                data_availability[s] = True
        MVs = np.int32(np.float32([template.metadata['s_moveouts'], template.metadata['s_moveouts'], template.metadata['p_moveouts']]) * \
                                  template.metadata['sampling_rate'])
        time = np.arange(template.traces[0].data.size + MVs[0,data_availability].max())
    else:
        time = np.arange(template.traces[0].data.size)
    for s in range(ns):
        for c in range(nc):
            ss = np.where(template.metadata['stations'] == sta[s])[0][0]
            plt.subplot(ns,nc,s*nc+c+1)
            lab = '{}.{}'.format(template.select(station=sta[s])[c].stats['station'],template.select(station=sta[s])[c].stats['channel'])
            if CC_comp:
                lab += ' {:.2f}'.format(CC[s])
            if mv_view:
                id1 = MVs[c,ss]
                id2 = id1 + template.traces[0].data.size
                if data_availability[ss]:
                    plt.plot(time[id1:id2], template.select(station=sta[s])[c].data, label=lab)
                    if c < 2:
                        plt.axvline(int((id1+id2)/2), lw=2, ls='--', color='k')
                    else:
                        plt.axvline(int(id1 + 1. * template.metadata['sampling_rate']), lw=2, ls='--', color='k')
                else:
                    plt.plot(time, np.zeros(time.size), label=lab)
            else:
                plt.plot(time, template.select(station=sta[s])[c].data, label=lab)
            #plt.axvline(time[time.size/2], color='k', ls='--')
            plt.xlim((time[0], time[-1]))
            plt.yticks([])
            plt.xticks([])
            plt.legend(loc='upper left', frameon=False, handlelength=0.1, borderpad=0.)
            if s == ns-1:
                plt.xlabel('Time (s)')
                xpos = np.arange(0, time.size, np.int32(10.*autodet.cfg.sampling_rate))
                xticks = [str(float(X)/autodet.cfg.sampling_rate) for X in xpos]
                plt.xticks(xpos, xticks)
    plt.subplots_adjust(bottom = 0.06, top = 0.94, hspace = 0.04, wspace = 0.12)
    plt.suptitle('Template {:d}, location: lat {:.2f}, long {:.2f}, depth {:.2f}km ($\Delta r=${:.2f}km)'\
                   .format(template.metadata['template_idx'], \
                           template.metadata['latitude'], \
                           template.metadata['longitude'], \
                           template.metadata['depth'], \
                           template.metadata['loc_uncertainty']), fontsize=24)
    if show:
        plt.show()

def plot_match(filename, index, db_path_T='template_db_1/', db_path_M='matched_filter_1/', db_path=autodet.cfg.dbpath, show=True):
    """
    plot_match(filename, index, db_path_T='template_db_1/', db_path_M='matched_filter_1/', db_path=autodet.cfg.dbpath, show=True)\n
    Function that plots the template waveforms on top on the continuous waveforms, where the detection was triggered. This is useful to
    investigate what kind of events are detected with a given template.
    """
    font = {'family' : 'serif', 'weight' : 'normal', 'size' : 20}
    plt.rc('font', **font)
    plt.rcParams.update({'ytick.labelsize'  :  14})
    plt.rcParams['pdf.fonttype'] = 42 #TrueType
    M, T = autodet.db_h5py.read_multiplet(filename, index, return_tp=True, db_path=db_path, db_path_T=db_path_T, db_path_M=db_path_M)
    T.metadata['p_moveouts'] += 3. # to remove later
    st_sorted = np.asarray(T.metadata['stations'])
    st_sorted = np.sort(st_sorted)
    I_s = np.argsort(T.metadata['stations'])
    ns = len(T.metadata['stations'])
    nc = len(M.components)
    plt.figure('multiplet{:d}_{}_TP{:d}'.format(index, M[0].stats.starttime.strftime('%Y-%m-%d'), M.template_ID))
    plt.suptitle('Template {:d} ({:.2f}/{:.2f}/{:.2f}km): Detection on {}'.format(M.template_ID,\
                                                                                  M.latitude,\
                                                                                  M.longitude,\
                                                                                  M.depth,\
                                                                                  M[0].stats.starttime.strftime('%Y,%m,%d -- %H:%M:%S')))
    try:
        mv = T.s_moveout
        mv = np.repeat(mv, nc).reshape(ns, nc)
    except:
        mv = np.hstack((T.metadata['s_moveouts'].reshape(-1,1),
                        T.metadata['s_moveouts'].reshape(-1,1),
                        T.metadata['p_moveouts'].reshape(-1,1)))
    if mv.dtype == np.dtype(np.int32):
        mv = np.float32(mv)/T.metadata['sampling_rate']
    t_min = 10. - 0.5
    t_max = t_min + autodet.cfg.template_len + mv.max() + 0.5
    idx_min = np.int32(t_min * T.metadata['sampling_rate'])
    idx_max = min(M.traces[0].data.size, np.int32(t_max * T.metadata['sampling_rate']))
    time  = np.linspace(t_min, t_max, idx_max-idx_min)
    time -= time.min()
    #-----------------------------------------
    mv = np.int32(mv * T.metadata['sampling_rate'])
    for s in range(ns):
        for c in range(nc):
            plt.subplot(ns, nc, s*nc+c+1)
            plt.plot(time, M.select(station=st_sorted[s])[c].data[idx_min:idx_max], color='C0', label='{}.{}'.format(st_sorted[s], M.components[c]), lw=0.75)
            idx1 = np.int32(min(10., autodet.cfg.multiplet_len/4.)*autodet.cfg.sampling_rate ) + mv[I_s[s],c]
            idx2 = idx1 + T[0].data.size
            if idx2 > (time.size + idx_min):
                idx2 = time.size + idx_min
            #try:
            #    Max = M.select(station=st_sorted[s])[c].data[idx1:idx2].max()
            #    if Max == 0.:
            #        Max = 1.
            #    data_toplot = T.select(station=st_sorted[s])[c].data/T.select(station=st_sorted[s])[c].data.max() * Max
            #    plt.plot(time[idx1-idx_min:idx2-idx_min], data_toplot, color='C2', lw=0.5)
            #except IndexError:
            #    # more components in multiplets than in template
            #    pass
            #except ValueError:
            #    # empty slice
            #    pass
            Max = M.select(station=st_sorted[s])[c].data[idx1:idx2].max()
            if Max == 0.:
                Max = 1.
            data_template = T.select(station=st_sorted[s])[c].data[:idx2-idx1]
            data_toplot   = data_template/data_template.max() * Max
            plt.plot(time[idx1-idx_min:idx2-idx_min], data_toplot, color='C3', lw=0.8)
            plt.xlim(time.min(), time.max())
            if s == ns-1: 
                plt.xlabel('Time (s)')
            else:
                plt.xticks([])
            if c < 2:
                plt.legend(loc='upper left', framealpha=1.0, handlelength=0.1, borderpad=0.2)
            else:
                plt.legend(loc='upper right', framealpha=1.0, handlelength=0.1, borderpad=0.2)
    plt.subplots_adjust(top=0.95,
                        bottom=0.04,
                        left=0.07,
                        right=0.95,
                        hspace=0.2,
                        wspace=0.2)
    if show:
        plt.show()

