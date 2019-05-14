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

#def reloc_manual(tp_id, net, MV, version, STEP, repicking=False, normRMS=True, mode=2):
#    import operator
#    db_path_M = 'MULTIPLETS_%s_STEP%i_' %(version, STEP)
#    db_path_T = 'TEMPLATES_%s_STEP%i/' %(version, STEP-1)
#    font = {'family' : 'serif', 'weight' : 'normal', 'size' : 14}
#    plt.rc('font', **font)
#    M = stack_multiplets(tp_id, db_path=autodet.cfg.dbpath+db_path_M+'RMS/', best=True, normRMS=normRMS)
#    M=M.filter('bandpass', freqmin=2., freqmax=12., zerophase=True)
#    T = autodet.db_h5py.read_template('template%i' %M.template_ID, db_path=autodet.cfg.dbpath+db_path_T)
#    print T.source_idx
#    ns = net.n_stations()
#    nc = net.n_components()
#    #==============================
#    # == SORT BY SNR ==
#    SNR_sorted = SNR(M, net.stations, net.components)
#    #==============================
#    if repicking:
#        if mode == 1:
#            Ns = 12
#            SNR_criterion = np.sort(SNR)[-Ns]
#            M.SNR_thrs = SNR_criterion
#            Ns = np.where(SNR >= SNR_criterion)[0].size
#            stations = np.asarray(net.stations)[np.where(SNR >= SNR_criterion)[0]].tolist()
#            idx = net.stations_idx(stations)
#        elif mode == 2: 
#            ns = net.n_stations()/4
#            nc = net.n_components()
#            stations = []
#            for s in range(ns):
#                stations.append(SNR_sorted[s][0])
#            idx = np.int32(net.stations_idx(stations))
#            global selection
#            selection = []
#            fig = plt.figure('stacked_multiplets_from_TP%i' %M.template_ID, figsize=(25, 12))
#            t = np.arange(M[0].stats.npts)
#            plt.suptitle('Earthquakes from the source at %.2f/%.2f/%.2fkm, template %i (%i multiplets)' %(M.latitude,\
#                                                                                                          M.longitude,\
#                                                                                                          M.depth,\
#                                                                                                          T.template_ID,\
#                                                                                                          M.Nmulti), fontsize=24)
#            for s in range(ns):
#                for c in range(nc):
#                    plt.subplot(ns, nc, s*nc+c+1)
#                    plt.plot(t[::2], M.select(station=stations[s])[c].data[::2], color='b', label='%s.%s' %(stations[s], M.components[c]))
#                    plt.legend(loc='upper left', frameon=False, handlelength=0.1, borderpad=0.)
#            plt.subplots_adjust(bottom = 0.06, top = 0.94, hspace = 0.04, wspace = 0.12)
#            cid = fig.canvas.mpl_connect('button_press_event', select_stations)
#            plt.show()
#            stations = np.asarray(net.stations)[idx[np.int32(selection)]].tolist()
#        else:
#            stations = T.stations
#        with open(autodet.cfg.picks_path+'%s/STEP%i/stations_tp%i.dat' %(version, STEP, tp_id), 'w') as f:
#            np.savetxt(f, np.asarray(stations), fmt="%s")
#    else:
#        with open(autodet.cfg.picks_path+'%s/STEP%i/stations_tp%i.dat' %(version, STEP, tp_id), 'r') as f:
#            stations = np.loadtxt(f, dtype='|S4').tolist()
#    print "Relocation using: ", stations
#    ns = len(stations)
#    nc = len(M.components)
#    print nc,ns
#    import os
#    if not os.path.isfile(autodet.cfg.picks_path+'%s/STEP%i/picksP_tp%i.dat' %(version, STEP, tp_id)) or repicking:
#        global picksP, orderP, picksS, orderS
#        picksP = []
#        picksS = []
#        PH = [[],[]]
#        phases = ['P', 'S']
#        clrs = ['b', 'k']
#        t = np.arange(M[0].stats.npts)
#        for ss in range(2*ns):
#            global picks
#            picks = []
#            s = ss % ns
#            ph = ss / ns
#            fig = plt.figure('stacked_multiplets_from_TP%i' %M.template_ID, figsize=(25, 12))
#            plt.suptitle('Earthquakes from the source at %.2f/%.2f/%.2fkm, template %i (%i multiplets)' %(M.latitude,\
#                                                                                                         M.longitude,\
#                                                                                                         M.depth,\
#                                                                                                         T.template_ID,\
#                                                                                                         M.Nmulti), fontsize=24)
#            for c in range(nc):
#                plt.subplot(nc, 1, 1+c)
#                plt.title("Picking phase %s" %phases[ph])
#                plt.plot(t, M.select(station=stations[s])[c].data, color=clrs[ph], label='%s.%s' %(stations[s], M.components[c]))
#                plt.legend(loc='upper left', frameon=False, handlelength=0.1, borderpad=0.)
#            plt.subplots_adjust(bottom = 0.10, top = 0.90, left=0.02, right=0.98)
#            cid = fig.canvas.mpl_connect('button_press_event', interactive_picking)
#            plt.show()
#            PH[ph].append(picks)
#        picksP = np.int32(PH[0])
#        picksS = np.int32(PH[1])
#        print picksP
#        print picksS
#        with open(autodet.cfg.picks_path+'%s/STEP%i/picksP_tp%i.dat' %(version, STEP, tp_id), 'wb') as f:
#            picksP.tofile(f)
#        with open(autodet.cfg.picks_path+'%s/STEP%i/picksS_tp%i.dat' %(version, STEP, tp_id), 'wb') as f:
#            picksS.tofile(f)
#        dP = np.abs(picksP[:,0] - picksP[:,1])
#        P_excl = dP > np.int32(3.*autodet.cfg.sampling_rate)
#        t0 = picksP[~P_excl,0].min()
#        picksS -= t0 # S - P
#        picksP -= t0
#    else:
#        with open(autodet.cfg.picks_path+'%s/STEP%i/picksP_tp%i.dat' %(version, STEP, tp_id), 'rb') as f:
#            picksP = np.fromfile(f, dtype=np.int32).reshape((-1,2))
#        with open(autodet.cfg.picks_path+'%s/STEP%i/picksS_tp%i.dat' %(version, STEP, tp_id), 'rb') as f:
#            picksS = np.fromfile(f, dtype=np.int32).reshape((-1,2)) 
#        dP = np.abs(picksP[:,0] - picksP[:,1])
#        P_excl = dP > np.int32(3.*autodet.cfg.sampling_rate)
#        t0 = picksP[~P_excl,0].min()
#        picksS -= t0
#        picksP -= t0
#    idx = net.stations_idx(stations)
#    mvP = MV.p_relative_samp[:,:,:,idx]
#    mvS = MV.s_relative_p_samp[:,:,:,idx]
#    # ====== WEIGHTS ======
#    Wp = 2 * np.ones(picksP.shape[0], dtype=np.int32)
#    sigP = np.sqrt(np.var(dP[~P_excl]))
#    Wp[np.where(dP > 5.*sigP)] = 1. # bad pick
#    Wp[np.where(dP < 0.5*sigP)] = 3. # good pick
#    Wp[P_excl] = 0
#    Ws = 2 * np.ones(picksS.shape[0], dtype=np.int32) 
#    dS = np.abs(picksS[:,0] - picksS[:,1])
#    S_excl = dS > np.int32(3.*autodet.cfg.sampling_rate)
#    sigS = np.sqrt(np.var(dS[~S_excl]))
#    Ws[np.where(dS > 5.*sigS)] = 1. # bad pick
#    Ws[np.where(dS < 0.5*sigS)] = 3. # good pick
#    Ws[S_excl] = 0
#    A = max(1., float(dP[~P_excl].sum()) / float(dS[~S_excl].sum()))
#    print float(dP[~P_excl].sum()) / float(dS[~S_excl].sum())
#    Ws = np.int32( Ws * A)
#    print Wp, Ws
#    station_ref = np.where(picksP == picksP[~P_excl].min())[0][0]
#    print station_ref
#    RMS, bestIdx = autodet.clib.manual_reloc(picksP[:,0], picksS[:,0], mvP, mvS, len(idx), station_ref, Wp, Ws, idx_EQ=MV.idx_EQ)
#    #RMS, bestIdx = autodet.clib.manual_reloc(picksP[:,0], picksS[:,0], mvP, mvS, len(idx), idx_EQ=MV.idx_EQ)
#    RMS = RMS.reshape((mvP.shape[0], mvP.shape[1], mvP.shape[2]))
#    with open(autodet.cfg.picks_path+'%s/STEP%i/RMS_tp%i.dat' %(version, STEP, tp_id), 'wb') as f:
#        RMS.tofile(f)
#    print bestIdx
#    i,j,k = MV.idx2sub(bestIdx)
#    print i,j,k
#    print "New location: %.2f/%.2f/%.2fkm" %(MV.latitude[i,j,k], MV.longitude[i,j,k], MV.depth[i,j,k])
#    mvS = MV.s_relative_p_samp[i,j,k,:] - MV.p_relative_samp[i,j,k,idx].min()
#    if len(stations) < 12:
#        SNR_sorted = SNR(M, net.stations, net.components, mv=mvS, T=t0)
#        to_add = 12 - len(stations)
#        stations_to_add = []
#        n = 0
#        s = 0
#        while n < to_add:
#            if SNR_sorted[s][0] not in stations:
#                stations_to_add.append(SNR_sorted[s][0])
#                n += 1
#            s += 1
#        stations.extend(stations_to_add)
#    elif len(stations) > 12:
#        SNR_sorted = SNR(M, net.stations, net.components, mv=mvS, T=t0)
#        SNR_sorted.reverse()
#        to_rm = len(stations) - 12
#        stations_to_rm = []
#        n = 0
#        s = 0
#        while n < to_rm:
#            if SNR_sorted[s][0] in stations:
#                stations.remove(SNR_sorted[s][0])
#                n += 1
#            s += 1
#    print "Stations finally returned: ", stations
#    t0 -= mvP[i,j,k,:].min()
#    idx = net.stations_idx(stations)
#    t0 += MV.p_relative_samp[i,j,k,idx].min()
#    return (i,j,k), MV.p_relative_samp[i,j,k,idx] - MV.p_relative_samp[i,j,k,idx].min(), MV.s_relative_p_samp[i,j,k,idx] - MV.p_relative_samp[i,j,k,idx].min(), RMS[:,:,k], stations, t0
#    #return (i,j,k), mvP[i,j,k,:] - mvP[i,j,k,:].min(), mvS[i,j,k,:] - mvP[i,j,k,:].min(), RMS[:,:,k], stations

#def redundant(tp1, tp2, db_path_M='MULTIPLETS_V1_STEP4_', db_path_T='TEMPLATES_V1_STEP3/', db_path_S='STACKS_V1_STEP1_', type_thrs='RMS', \
#              R_thres=0.7, CC_thres=0.7, DT=5., fband=None, temporal_catalog=None):
#    """
#    redundant(tp1, tp2, db_path_M='MULTIPLETS_V1_Z4_', db_path_T='TEMPLATES_V1_Z3/', type='MAD', R_thres=0.5, CC_thres=0.7, DT=5., fband=None): \n
#    Determines whether or not the templates tp1 and tp2 are redundant. It is based on the following criterions: \n
#    - The number of detections closer than DT sec must be larger than R_thres * length_shortest_family.
#    - The coherency between their waveforms' envelopes must be larger than CC_thres on average.
#    """
#    from scipy.signal import hilbert
#    if temporal_catalog is not None:
#        OT1, C1, OT2, C2 = temporal_catalog
#    else:
#        OT1, C1 = temporal_distribution(tp1, 3600.*24., db_path_M=db_path_M, type_thrs=type_thrs)
#        OT2, C2 = temporal_distribution(tp2, 3600.*24., db_path_M=db_path_M, type_thrs=type_thrs)
#    imax = 0
#    if len(OT1) > len(OT2):
#        t = OT2
#        T = OT1
#        imax = 0
#    else:
#        t = OT1
#        T = OT2
#        imax = 1
#    # T = list of the origin times of the biggest family / t = list of the origin times of the shortest family
#    jj = 0
#    Nred = 0 # number of redundant detections
#    for i in range(len(t)):
#        for j in range(jj, len(T)):
#            if abs(t[i] - T[j]) < DT:
#                Nred += 1
#                jj = j
#    T1 = autodet.db_h5py.read_template('template%i' %tp1, db_path=autodet.cfg.dbpath+db_path_T)
#    T2 = autodet.db_h5py.read_template('template%i' %tp2, db_path=autodet.cfg.dbpath+db_path_T)
#    st_com = list(set(T1.stations).intersection(T2.stations))
#    if float(Nred)/len(t) < R_thres or len(st_com) < 8:
#        #print "Templates %i and %i are not redundant (shared detections: %.2f, %i stations in common)" %(tp1, tp2, float(Nred)/len(t), len(st_com))
#        return 0, 0
#    M1 = autodet.db_h5py.read_stack_multiplets('stack%i' %tp1, db_path_S=db_path_S+type_thrs+'/', db_path=autodet.cfg.stacks_path+autodet.cfg.substudy+'/')
#    M2 = autodet.db_h5py.read_stack_multiplets('stack%i' %tp2, db_path_S=db_path_S+type_thrs+'/', db_path=autodet.cfg.stacks_path+autodet.cfg.substudy+'/')
#    if fband is not None:
#        M1.filter('bandpass', freqmin=fband[0], freqmax=fband[1], zerophase=True)
#        M2.filter('bandpass', freqmin=fband[0], freqmax=fband[1], zerophase=True)
#    CC_moy = 0.
#    for st in st_com:
#        print "================= STATION %s ==================" %st
#        for c in range(len(T1.components)):
#            data1 = np.abs(hilbert(M1.select(station=st, channel=M1.components[c])[0].data))
#            data2 = np.abs(hilbert(M2.select(station=st, channel=M1.components[c])[0].data))
#            print np.sum(data1*data2)/np.sqrt(np.power(data1, 2).sum() * np.power(data2, 2).sum())
#            CC_moy += np.sum(data1*data2)/np.sqrt(np.power(data1, 2).sum() * np.power(data2, 2).sum())
#    CC_moy /= float(len(st_com) * len(T1.components))
#    print "The templates %i and %i share %.2f perc of their detections, and their mean CC is %.2f" %(tp1, tp2, float(Nred)/len(t), CC_moy)
#    if CC_moy < CC_thres: 
#        #print "Templates %i and %i are not redundant (mean CC = %.2f)" %(tp1, tp2, CC_moy)
#        return 0, 0
#    import copy
#    M = [M1, M2]
#    SNRi = np.zeros((len(M1.stations), len(M1.components)), dtype=np.float32)
#    SNRf = np.zeros((len(M1.stations), len(M1.components)), dtype=np.float32)
#    M_stack = copy.deepcopy(M[imax])
#    for s in range(len(M_stack.stations)):
#        for c in range(len(M_stack.components)):
#            data = M_stack.select(station=M_stack.stations[s], channel=M_stack.components[c])[0].data
#            var = np.var(data)
#            if var != 0.:
#                SNRi[s, c] += np.square(data).max()/var
#            M_stack.select(station=M_stack.stations[s], channel=M_stack.components[c])[0].data = \
#            (M1.select(station=M_stack.stations[s], channel=M_stack.components[c])[0].data + \
#            M2.select(station=M_stack.stations[s], channel=M_stack.components[c])[0].data) / 2.
#            var = np.var(M_stack.select(station=M_stack.stations[s], channel=M_stack.components[c])[0].data)
#            if var != 0.:
#                SNRf[s, c] += np.square(M_stack.select(station=M_stack.stations[s], channel=M_stack.components[c])[0].data).max()/var
#    M[imax].SNR = SNRi
#    M_stack.SNR = SNRf
#    # HEADER
#    M_stack.template_ID1 = tp1
#    M_stack.template_ID2 = tp2
#    M_stack.latitude1, M_stack.longitude1, M_stack.depth1 = (M1.latitude, M1.longitude, M1.depth)
#    M_stack.latitude2, M_stack.longitude2, M_stack.depth2 = (M2.latitude, M2.longitude, M2.depth)
#    return M_stack, M[imax]
#
#
#def stack_multiplets(template_ID, db_path=autodet.cfg.dbpath, best=False, \
#                     Nth_root=False, normRMS=False, \
#                     return_waveforms=False):
#    from subprocess import Popen, PIPE
#    from obspy import Stream, Trace
#    print "Looking for %s%i_*" %(db_path+'*multiplets', template_ID)
#    files_list = Popen('ls '+db_path+'*multiplets%i_*' %template_ID, stdout=PIPE, shell=True)
#    files = []
#    while True:
#        line = files_list.stdout.readline()[:-1]
#        if line == '':
#            break
#        else:
#            files.append(str(line))
#    S = Stream()
#    i = 0
#    while True:
#        try:
#            if files[i][-len('meta.h5'):] != 'meta.h5':
#                i += 1
#            else:
#                with h5.File(files[i], mode='r') as fm0:
#                    if len(fm0['amplitudes']) == 0:
#                        i += 1
#                        continue
#                    else:
#                        nc = len(fm0['components'][0,:])
#                        ns = len(fm0['stations'][0,:])
#                        S.stations = fm0['stations'][0,:].tolist()
#                        S.components = fm0['components'][0,:].tolist()
#                        S.latitude = fm0['latitude'][()]
#                        S.longitude = fm0['longitude'][()]
#                        S.depth = fm0['depth'][()]
#                        S.template_ID = fm0['template_ID'][()]
#                        #S.template_ID = fm0['template_idx'][()]
#                        Nsamp = np.int32(autodet.cfg.multiplet_len * autodet.cfg.sampling_rate)
#                        break
#        except IndexError:
#            print "None multiplet for template %i !! Return None" %template_ID
#            return None
#    print S.latitude, S.longitude, S.depth, S.template_ID
#    stack = np.zeros((ns, nc, Nsamp), dtype=np.float32)
#    print 'Shape of the stack: ', stack.shape
#    CC = []
#    if best:
#        for file in files:
#            if file[-len('meta.h5'):] != 'meta.h5':
#                continue
#            with h5.File(file, mode='r') as fm:
#                if len(fm['coherencies']) == 0:
#                    continue
#                else:
#                    CC.extend(fm['coherencies'][:].tolist())
#        CC.sort()
#        #CC_thres = np.sort(CC)[-min(5, len(CC))]
#        if len(CC) > 70:
#            CC_thres = CC[int(7./10.*len(CC))] # the best 30%
#        elif len(CC) > 30:
#            CC_thres = np.median(CC) # the best 50%
#        elif len(CC) > 10:
#            CC_thres = np.percentile(CC, 33.) # the best 66% detections 
#        else:
#            CC_thres = 0.
#    Nstack = np.zeros((ns, nc), dtype=np.float32)
#    Nmulti = 0
#    WAVEFORMS = np.zeros((0,ns,nc,Nsamp), dtype=np.float32)
#    for file in files:
#        if file[-len('wav.h5'):] != 'wav.h5':
#            continue
#        with h5.File(file, mode='r') as fw:
#            if len(fw['waveforms']) == 0:
#                continue
#            else:
#                if best:
#                    with h5.File(file[:-len('wav.h5')]+'meta.h5', mode='r') as fm:
#                        selection = np.where(fm['coherencies'][:] > CC_thres)[0]
#                        if selection.size == 0:
#                            continue
#                        waves = np.zeros((selection.size, ns, nc, Nsamp), dtype=np.float32)
#                        waves[:,:,:,:] = fw['waveforms'][selection,:,:,:]
#                else:
#                    waves = fw['waveforms'][:,:,:,:]
#                if return_waveforms:
#                    WAVEFORMS = np.vstack( (WAVEFORMS, waves) )
#                Nmulti += waves.shape[0]
#                for m in range(waves.shape[0]):
#                    for s in range(ns):
#                        for c in range(nc):
#                            if normRMS:
#                                norm = np.sqrt(np.var(waves[m,s,c,:]))
#                            else:
#                                norm = np.abs(waves[m,s,c,:]).max()
#                            if norm != 0.:
#                                waves[m,s,c,:] /= norm
#                                if Nth_root:
#                                    stack[s,c,:] += np.sqrt(np.abs(waves[m,s,c,:]))*np.sign(waves[m,s,c,:])
#                                else:
#                                    stack[s,c,:] += waves[m,s,c,:]
#                                Nstack[s,c] += 1.
#    if Nth_root:
#        stack = np.power(stack, 2)*np.sign(stack)
#    # make the stream object
#    for s in range(ns):
#        for c in range(nc):
#            if Nstack[s,c] != 0.:
#                S += Trace(data = stack[s,c,:]/Nstack[s,c])
#            else:
#                S += Trace(data = stack[s,c,:])
#            S[-1].stats['sampling_rate'] = autodet.cfg.sampling_rate
#            S[-1].stats['station'] = S.stations[s]
#            S[-1].stats['channel'] = S.components[c]
#    S.Nmulti = Nmulti
#    if return_waveforms:
#        S.data = WAVEFORMS
#    return S

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
    

def spectral_filtering(M, SNR_thres=5., max_freq=autodet.cfg.max_freq):
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
    # a single singular vector ===> does nit make sense !!!!!
    # the projected matrix has rows that are perfectly coherent !
    filtered_data = scisig.wiener(filtered_data, mysize=[max(2, int(U.shape[0]/10)), int(autodet.cfg.sampling_rate/max_freq)])
    return filtered_data

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
    U, S, Vt = scilin.svd(matrix, full_matrices=0)
    filtered_data = np.zeros((U.shape[0], Vt.shape[1]), dtype=np.float32)
    for n in range(min(U.shape[0], N_singular_values)):
        s_n = np.zeros(S.size, dtype=np.float32)
        s_n[n] = S[n]
        projection_n = np.dot(U, np.dot(np.diag(s_n), Vt))
        filtered_projection = scisig.wiener(projection_n, mysize=[max(2, int(U.shape[0]/10)), int(autodet.cfg.sampling_rate/max_freq)])
        if np.isnan(filtered_projection.max()):
            continue
        filtered_data += filtered_projection
    filtered_data = scisig.wiener(filtered_data, mysize=[max(2, int(U.shape[0]/10)), int(autodet.cfg.sampling_rate/max_freq)])
    return filtered_data


#def SVD_multiplets(template_ID, db_path=autodet.cfg.dbpath, WAVEFORMS=None, best=False, normRMS=True, Nsv=1, return_waveforms=False, wiener_filter=False):
#    from subprocess import Popen, PIPE
#    from obspy import Stream, Trace
#    from scipy.linalg import svd
#    from scipy.signal import wiener
#    print "Looking for %s%i_*" %(db_path+'*multiplets', template_ID)
#    files_list = Popen('ls '+db_path+'*multiplets%i_*' %template_ID, stdout=PIPE, shell=True)
#    files = []
#    while True:
#        line = files_list.stdout.readline()[:-1]
#        if line == '':
#            break
#        else:
#            files.append(str(line))
#    i = 0
#    Nsamp = 0
#    ns = 0
#    S = Stream()
#    #------------- retrieve metadata ---------------
#    while True:
#        try:
#            wav = files[i][-len('wav.h5'):] == 'wav.h5'
#            if wav:
#                if Nsamp == 0:
#                    with h5.File(files[i], mode='r') as fwav0:
#                        Nsamp = fwav0['waveforms'][:,:,:,:].shape[-1]
#                i += 1
#            else:
#                with h5.File(files[i], mode='r') as fm0:
#                    if len(fm0['amplitudes']) == 0:
#                        i += 1
#                        continue
#                    else:
#                        i += 1
#                        nc = len(fm0['components'][0,:])
#                        ns = len(fm0['stations'][0,:])
#                        S.stations = fm0['stations'][0,:].tolist()
#                        S.components = fm0['components'][0,:].tolist()
#                        S.latitude = fm0['latitude'][()]
#                        S.longitude = fm0['longitude'][()]
#                        S.depth = fm0['depth'][()]
#                        S.template_ID = fm0['template_idx'][()]
#
#            if ns != 0 and Nsamp != 0:
#                break
#        except IndexError:
#            print "None multiplet for template %i !! Return None" %template_ID
#            return None
#    #----------------------------------------------
#    if WAVEFORMS is None:
#        CC = np.zeros(0, dtype=np.float32)
#        if best:
#            for file in files:
#                if file[-len('meta.h5'):] != 'meta.h5':
#                    continue
#                with h5.File(file, mode='r') as fm:
#                    if len(fm['coherencies']) == 0:
#                        continue
#                    else:
#                        CC = np.hstack((CC, fm['coherencies'][:]))
#            CC = np.sort(CC)
#            #CC_thres = np.sort(CC)[-min(5, len(CC))]
#            if len(CC) > 300:
#                CC_thres = CC[-101] 
#            elif len(CC) > 70:
#                CC_thres = CC[int(7./10.*len(CC))] # the best 30%
#            elif len(CC) > 30:
#                CC_thres = np.median(CC) # the best 50%
#            elif len(CC) > 10:
#                CC_thres = np.percentile(CC, 33.) # the best 66% detections 
#            else:
#                CC_thres = 0.
#        Nstack = np.zeros((ns, nc), dtype=np.float32)
#        WAVEFORMS  = np.zeros((0,ns,nc,Nsamp), dtype=np.float32)
#        Nmulti = 0
#        for file in files:
#            if file[-len('wav.h5'):] != 'wav.h5':
#                continue
#            with h5.File(file, mode='r') as fw:
#                if len(fw['waveforms']) == 0:
#                    continue
#                else:
#                    if best:
#                        with h5.File(file[:-len('wav.h5')]+'meta.h5', mode='r') as fm:
#                            selection = np.where(fm['coherencies'][:] > CC_thres)[0]
#                            if selection.size == 0:
#                                continue
#                            waves = np.zeros((selection.size, ns, nc, Nsamp), dtype=np.float32)
#                            waves[:,:,:,:] = fw['waveforms'][selection,:,:,:]
#                    else:
#                        waves = fw['waveforms'][:,:,:,:]
#                    Nmulti += waves.shape[0]
#                    for m in range(waves.shape[0]):
#                        for s in range(ns):
#                            for c in range(nc):
#                                if normRMS:
#                                    norm = np.sqrt(np.var(waves[m,s,c,:]))
#                                else:
#                                    norm =1.
#                                if norm != 0.:
#                                    waves[m,s,c,:] /= norm
#                    WAVEFORMS = np.vstack((WAVEFORMS, waves))
#    elif normRMS:
#        for m in range(WAVEFORMS.shape[0]):
#            for s in range(ns):
#                for c in range(nc):
#                    norm = np.sqrt(np.var(WAVEFORMS[m,s,c,:]))
#                    if norm != 0.:
#                        WAVEFORMS[m,s,c,:] /= norm
#    else:
#        pass
#    if return_waveforms:
#        S.data = np.array(WAVEFORMS, copy=True)
#    Principal_Components = np.zeros((Nsv, ns, nc, Nsamp), dtype=np.float32)
#    Largest_SV = np.zeros((ns, nc), dtype=np.float32)
#    Coeff_First_SV = np.zeros((ns,nc,Nmulti), dtype=np.float32)
#    R_SV = np.zeros((ns, nc), dtype=np.float32)
#    for s in range(ns):
#        for c in range(nc):
#            #U, sig, V = np.linalg.svd(WAVEFORMS[:,s,c,:])
#            U, sig, V = svd(WAVEFORMS[:,s,c,:], full_matrices=0, overwrite_a=False)
#            Principal_Components[:,s,c,:] = V[:Nsv,:]
#            Largest_SV[s,c] = sig[0]
#            Coeff_First_SV[s,c,:] = U[:,0]
#            R_SV[s,c] = sig[0] / sig.sum()
#            if wiener_filter:
#                var = np.sqrt(np.var(V, axis=-1))
#                var[var == 0.] = 1.
#                for i in range(V.shape[0]):
#                    V[i,:] /= var[i]
#                y = wiener(V, 50)
#                #S += Trace(data=np.sum(np.dot(np.diag(sig), y), axis=0))
#                S += Trace(data=np.sum(np.dot(U, np.dot(np.diag(sig), y)), axis=0))
#            else:
#                S += Trace(data=V[0,:])
#            S[-1].stats.station = S.stations[s]
#            S[-1].stats.channel = S.components[c]
#            S[-1].stats.sampling_rate = autodet.cfg.sampling_rate
#    S.singular_vectors = Principal_Components
#    S.SV = Largest_SV
#    S.R_SV = R_SV
#    S.coeff_SV = Coeff_First_SV
#    S.Nmulti = Nmulti
#    del WAVEFORMS
#    return S
#
#def mangitude_svd(template_ID, db_path=autodet.cfg.dbpath, best=False, norm=True, Nsv=1):
#    from subprocess import Popen, PIPE
#    from obspy import Stream, Trace
#    from scipy.linalg import svd
#    print "Looking for %s%i_*" %(db_path+'*multiplets', template_ID)
#    files_list = Popen('ls '+db_path+'*multiplets%i_*' %template_ID, stdout=PIPE, shell=True)
#    files = []
#    while True:
#        line = files_list.stdout.readline()[:-1]
#        if line == '':
#            break
#        else:
#            files.append(str(line))
#    WAVEFORMS  = np.zeros((0,ns,nc,Nsamp), dtype=np.float32)
#    Nmulti = 0
#    for file in files:
#        if file[-len('wav.h5'):] != 'wav.h5':
#            continue
#        with h5.File(file, mode='r') as fw:
#            if len(fw['waveforms']) == 0:
#                continue
#            else:
#                waves = fw['waveforms'][:,:,:,:]
#                Nmulti += waves.shape[0]
#                WAVEFORMS = np.vstack((WAVEFORMS, waves))
#    Principal_Components = np.zeros((Nsv, ns, nc, Nsamp), dtype=np.float32)
#    Largest_SV = np.zeros((ns, nc), dtype=np.float32)
#    R_SV = np.zeros((ns, nc), dtype=np.float32)
#    relAmp = np.zeros((Nmulti, ns, nc), dtype=np.float32)
#    for s in range(ns):
#        for c in range(nc):
#            U, sig, V = np.linalg.svd(WAVEFORMS[:,s,c,:], full_matrices=0)
#            relAmp[:,s,c] = U[0,:]
#    return relAmp
#
#def mag_multiplets(M_idx, db_path_M='MULTIPLETS_V1_Z4_', type_thrs='MAD'):
#    from subprocess import Popen, PIPE
#    from obspy import UTCDateTime as udt
#    print "Looking for %s%i_*meta.h5" %(autodet.cfg.dbpath+db_path_M+type_thrs+'/*multiplets', M_idx)
#    files_list = Popen('ls '+autodet.cfg.dbpath+db_path_M+type_thrs+'/*multiplets%i_*meta.h5' %M_idx, stdout=PIPE, shell=True)
#    files = []
#    while True:
#        line = files_list.stdout.readline()[:-1]
#        if line == '':
#            break
#        else:
#            files.append(str(line))
#    Med_ampl = []
#    OT = []
#    for file in files:
#        with h5.File(file, mode='r') as fm:
#            Med_ampl.extend(fm['amplitudes'][:].tolist())
#            OT.extend(fm['origin_times'][:].tolist())
#    return np.asarray(OT), np.asarray(Med_ampl)
#
#def burstiness(tp_ID, db_path_M='MULTIPLETS_V1_Z4_', type_thrs='RMS', db_path=autodet.cfg.dbpath):
#    OT, C = temporal_distribution(tp_ID, 3600.*24., db_path_M=db_path_M, type_thrs=type_thrs, db_path=db_path)
#    b = 0.
#    for i in range(1, len(OT)):
#        if OT[i] - OT[i-1] < 60.*10.: # if inf to 10 minutes
#            b += 1.
#    b = b/len(OT)*100. # b is the burstiness in percentage
#    return b, len(OT)
#
#def false_buffer(tp_ID, version, STEP, type_thrs='RMS', db_path=autodet.cfg.dbpath):
#    import datetime as dt
#    OT, C = temporal_distribution(tp_ID, 3600.*24., version=version, STEP=STEP, type_thrs=type_thrs, db_path=db_path)
#    f = 0.
#    if len(OT) == 0:
#        print "NO DETECTIONS FOR TEMPLATE %i ! To remove !" %tp_ID
#        return f, 0
#    for i in range(len(OT)):
#        T1 = udt(udt(OT[i]).strftime('%Y,%m,%d'))
#        T2 = T1 + dt.timedelta(days=1)
#        if OT[i] - T1.timestamp < autodet.cfg.data_buffer:
#            f += 1.
#        elif T2.timestamp - OT[i] < autodet.cfg.data_buffer:
#            f += 1.
#        else:
#            pass
#    f = f/len(OT)*100. # f is the number of false detections in percentage
#    return f, len(OT)
#
#def temporal_distribution(tid, dt, version='V1', STEP=7, db_path_M=None, \
#                          type_thrs='MAD', corr=False, \
#                          study=autodet.cfg.substudy, db_path=autodet.cfg.dbpath):
#    """
#    temporal_distribution(M_idx, dt, db_path_M='MULTIPLETS_Z2_', type_thrs='MAD') \n
#    M_idx = template's index of the multiplets, dt = temporal resolution
#    wanted for the event count.
#    """
#    from subprocess import Popen, PIPE
#    if db_path_M is None:
#        db_path_M = 'MULTIPLETS_%s_STEP%i_' %(version, STEP)
#    if isfile(autodet.cfg.tpdistrib_path + '%s/%s/STEP%i/%s/' %(study, version, STEP, type_thrs) + 'C_dt%.0f_tp%i.dat' %(dt, tid)) and not corr:
#        print "Temporal distribution of template %i s detections loaded from existing file" %tid
#        with open(autodet.cfg.tpdistrib_path + '%s/%s/STEP%i/%s/' %(study, version, STEP, type_thrs) + 'C_dt%.0f_tp%i.dat' %(dt, tid), 'rb') as fC:
#            C = np.fromfile(fC, dtype=np.int32)
#        with open(autodet.cfg.tpdistrib_path + '%s/%s/STEP%i/%s/' %(study, version, STEP, type_thrs) + 'OT_tp%i.dat' %tid, 'rb') as fOT:
#            OT = np.fromfile(fOT, dtype=np.float64)
#        return OT, C
#    else:
#        print "Looking for %s%i_*meta.h5" %(db_path+db_path_M+type_thrs+'/*multiplets', tid)
#    files_list = Popen('ls '+db_path+db_path_M+type_thrs+'/*multiplets%i_*meta.h5' %tid, stdout=PIPE, shell=True)
#    files = []
#    while True:
#        line = files_list.stdout.readline()[:-1]
#        if line == '':
#            break
#        else:
#            files.append(str(line))
#    OT = []
#    if corr:
#        CC = []
#    for file in files:
#        with h5.File(file, mode='r') as fm:
#            OT.extend(fm['origin_times'][:].tolist())
#            if corr:
#                CC.extend(fm['coherencies'][:].tolist())
#    start_date = udt('2012,08,01')
#    end_date = udt('2013,08,02')
#    OT = np.asarray(OT)
#    idx_sort = np.argsort(OT)
#    OT = np.sort(OT)
#    #OT.sort()
#    if corr:
#        CC = np.asarray(CC)[idx_sort]
#    t = np.arange(start_date.timestamp, end_date.timestamp+dt, dt)
#    count = np.zeros(t.size, dtype=np.int32)
#    n = 1
#    for i in range(len(OT)):
#        while OT[i] > t[n]:
#            n += 1
#        count[n-1] += 1
#    # write the results for next time
#    with open(autodet.cfg.tpdistrib_path + '%s/%s/STEP%i/%s/' %(study, version, STEP, type_thrs) + 'C_dt%.0f_tp%i.dat' %(dt, tid), 'wb') as fC:
#        count.tofile(fC)
#    with open(autodet.cfg.tpdistrib_path + '%s/%s/STEP%i/%s/' %(study, version, STEP, type_thrs) + 'OT_tp%i.dat' %tid, 'wb') as fOT:
#        OT.tofile(fOT) # !!!!!!!! float 64 DOUBLE PRECISION !!!!!!!!!
#    if corr:
#        return OT, count, CC
#    else:
#        return OT, count
#
#def spatial_corr(pairs_list, tp_IDs, version, STEP, day, DT=60., W_days=10, type_thrs='MAD'):
#    db_path_M = 'MULTIPLETS_%s_Z%i_' %(version, STEP)
#    C = []
#    for tid in tp_IDs:
#        C.append(temporal_distribution(tid, DT, version=version, STEP=STEP, type_thrs=type_thrs)[1])
#    ntp = len(C)
#    start_day = udt('2012,08,01')
#    idx_corr = int((udt(day).timestamp - start_day.timestamp)//DT)
#    W_DT = int(W_days*3600.*24./DT)
#    CC = np.zeros(len(pairs_list), dtype=np.float32)
#    for i,p in enumerate(pairs_list):
#        source1 = C[p[0]][idx_corr-W_DT/2:idx_corr+W_DT/2]
#        source2 = C[p[1]][idx_corr-W_DT/2:idx_corr+W_DT/2]
#        den = np.sqrt( np.power(source1, 2).sum() * np.power(source2, 2).sum() )
#        if den != 0.:
#            CC[i] = np.sum(source1*source2) / den
#    CC_moy = np.zeros(ntp, dtype=np.float32)
#    p_array = np.asarray(pairs_list)
#    for t in range(CC_moy.size):
#        CC_moy[t] = CC[np.where(p_array == t)[0]].mean()
#    return CC, CC_moy
#
#def spatial_corr_period(pairs_list, tp_IDs, study, version, STEP, \
#                        day1='2012,08,01:12,00,00', day2='2013,08,01', \
#                        DT=60., W_days=10, \
#                        type_thrs='MAD', path_SC = autodet.cfg.scorr_path):
#    from os.path import isfile
#    start_day = udt(day1)
#    end_day = udt(day2)
#    CCname = 'CC%s_%s_%s_%s.dat' %(study, version, start_day.strftime('%Y%m%d'), end_day.strftime('%Y%m%d'))
#    CCmoyname = 'CCmoy%s_%s_%s_%s.dat' %(study, version, start_day.strftime('%Y%m%d'), end_day.strftime('%Y%m%d'))
#    T = [start_day]
#    while T[-1].timestamp < end_day.timestamp:
#        T.append(T[-1] + 3600.*24.)
#    print len(T)
#    if isfile(path_SC + CCname):
#        with open(path_SC + CCname, 'rb') as fCC:
#            CC = np.fromfile(fCC, dtype=np.float32)
#            CC = CC.reshape( (len(T), -1) )
#        with open(path_SC + CCmoyname, 'rb') as fCCmoy:
#            CC_moy = np.fromfile(fCCmoy, dtype=np.float32)
#            CC_moy = CC_moy.reshape( (len(T), -1) )
#        return CC, CC_moy
#    db_path_M = 'MULTIPLETS_%s_Z%i_' %(version, STEP)
#    p_array = np.asarray(pairs_list)
#    C = []
#    for tid in tp_IDs:
#        C.append(temporal_distribution(tid, DT, version=version, STEP=STEP, type_thrs=type_thrs)[1])
#    ntp = len(C)
#    W_DT = int(W_days*3600.*24./DT)
#    CC = np.zeros((len(T), len(pairs_list)), dtype=np.float32)
#    CC_moy = np.zeros((len(T), ntp), dtype=np.float32)
#    for d in range(len(T)):
#        day = T[d]
#        idx_corr = int((udt(day).timestamp - start_day.timestamp)//DT)
#        for i,p in enumerate(pairs_list):
#            source1 = C[p[0]][idx_corr-W_DT/2:idx_corr+W_DT/2]
#            source2 = C[p[1]][idx_corr-W_DT/2:idx_corr+W_DT/2]
#            den = np.sqrt( np.power(source1, 2).sum() * np.power(source2, 2).sum() )
#            if den != 0.:
#                CC[d,i] = np.sum(source1*source2) / den
#        for t in range(ntp):
#            CC_moy[d,t] = CC[d, np.where(p_array == t)[0]].mean()
#    with open(path_SC + CCname, 'wb') as fCC:
#        CC.tofile(fCC)
#    with open(path_SC + CCmoyname, 'wb') as fCCmoy:
#        CC_moy.tofile(fCCmoy)
#    return CC, CC_moy
#
#def plot_spatial_corr(templates_list, study, version, STEP, day, DT=60., W_days=10, type_thrs='MAD', db_path=autodet.cfg.dbpath):
#    """
#    plot_spatial_distrib(templates_list, version, STEP, day, DT=60., W_days=10, type_thrs='MAD') \n
#    """
#    db_path_T = 'TEMPLATES_%s_Z3/' %version
#    from mpl_toolkits.basemap import Basemap
#    lat = [43.5, 46.0]
#    lon = [4.5, 9.0]
#    map = Basemap(projection='lcc', \
#                  llcrnrlon=lon[0], llcrnrlat=lat[0], urcrnrlon=lon[1], urcrnrlat=lat[1],\
#                  lat_0=np.mean(lat), lon_0=np.mean(lon),\
#                  resolution='l')
#    X = []
#    Y = []
#    tp_IDs = []
#    with open(autodet.cfg.tpfiles_path + templates_list, 'r') as f:
#        line = f.readline()[:-1].strip()
#        while line != '':
#            tp_IDs.append(int(line[len('template'):]))
#            tp = autodet.db_h5.read_template(line, db_path=db_path+db_path_T)
#            x, y = map(tp.longitude, tp.latitude)
#            X.append(x)
#            Y.append(y)
#            line = f.readline()[:-1].strip()
#    X = np.asarray(X)
#    Y = np.asarray(Y)
#    pairs_list, tp_IDs = find_neighbors(templates_list, db_path=autodet.cfg.dbpath, db_path_T=db_path_T, return_tpIDs=True, plot=False)
#    CC, CC_moy = spatial_corr(pairs_list, tp_IDs, version, STEP, day, DT=DT, W_days=W_days, type_thrs=type_thrs)
#    parallels = np.arange(lat[0], lat[1], 0.5)
#    labels_p = np.zeros(parallels.size, dtype=np.bool)
#    labels_p[np.where(parallels % 0.5 == 0.)[0]] = 1
#    meridians = np.arange(lon[0], lon[1], 0.5)
#    labels_m = np.zeros(meridians.size, dtype=np.bool)
#    labels_m[np.where(meridians % 0.5 == 0.)[0]] = 1
#    map.drawparallels(parallels, labels=labels_p)
#    map.drawmeridians(meridians, labels=labels_m)
#    map.drawcountries()
#    #-----------------------------------------------------------
#    from matplotlib.colors import Normalize
#    from matplotlib.cm import ScalarMappable   
#    cm = plt.get_cmap('gnuplot_r')
#    cNorm = Normalize(vmin=CC.min(),vmax=CC.max()) # echelle de profondeur
#    scalarMap = ScalarMappable(norm=cNorm, cmap=cm)
#    for t in range(len(X)):
#        clr = scalarMap.to_rgba(CC_moy[t])
#        map.plot(X[t], Y[t], marker='o', ls='', markersize=5, color=clr)
#    for i,p in enumerate(pairs_list):
#        clr = scalarMap.to_rgba(CC[i])
#        map.plot(X[p], Y[p], ls='--', color=clr)
#    Xcb = np.asarray([ (i * (lon[1]-lon[0])/len(CC) + lon[0]) for i in range(len(CC))])
#    Ycb = np.asarray([ (i * (lat[1]-lat[0])/len(CC) + lat[0]) for i in range(len(CC))])
#    Xcb, Ycb = map(Xcb, Ycb)
#    CB = map.pcolor(Xcb, \
#                    Ycb, \
#                    CC, cmap=cm, tri=True, vmin=cNorm.vmin, vmax=cNorm.vmax)
#    cb = map.colorbar(label='Coherency coefficient', pad='5%', location='bottom')
#    CB.remove()
#    plt.show()
#
#def slope_clustering(T, w_days, dt, L_hours=1.):
#    """
#    slope_clustering(T, w_days, dt, L_hours=1.) \n
#    Calculates the spectrum's slope (ie the degree of clustering / coefficient of the power law) of a sliding window (width: w days) on T, with a lag of +/- L_houautodet.
#    dt is the temporal resolution of T, in seconds. In our study, T is the event count signal.
#    """
#    from repeat_search import clib
#    from scipy import stats
#    W = int(w_days * (24.*3600./dt))
#    L = int(L_hours*3600. / dt)
#    slopes = np.zeros(T.size - W, dtype=np.float32)
#    freq = np.fft.rfftfreq(W, dt)
#    period = np.power(freq[1:], -1)
#    # conversion to days
#    period /= 3600.*24.
#    idx_fit = np.where((period > 0.01) & (period < 1.))[0]
#    per_log = np.log10(period[idx_fit])
#    sm = np.ones(30, dtype=np.float32)
#    sm /= sm.size
#    for i in range(W/2+L, T.size-W/2-L):
#        if i%10000 == 0:
#            print "---------- %i / %i ----------" %(i-W/2-L, T.size-W-2*L)
#        FFT = np.abs(np.fft.rfft(T[i-W/2:i+W/2]))
#        FFT = np.convolve(FFT, sm, mode='same')[idx_fit]
#        s, intercept, r_value, p_value, std_err = stats.linregress(per_log, np.log10(FFT))
#        slopes[i-(W/2+L)] = s
#    return slopes
#
#def slope_clustering_ONEPERDAY(T, w_days, dt, L_hours=1., NDAYS=366):
#    """
#    slope_clustering(T, w_days, dt, L_hours=1.) \n
#    Calculates the spectrum's slope (ie the degree of clustering / coefficient of the power law) of a sliding window (width: w days) on T, with a lag of +/- L_houautodet.
#    dt is the temporal resolution of T, in seconds. In our study, T is the event count signal.
#    """
#    from repeat_search import clib
#    from scipy import stats
#    W = int(w_days * (24.*3600./dt))
#    L = int(L_hours*3600. / dt)
#    slopes = np.zeros(NDAYS, dtype=np.float32)
#    freq = np.fft.rfftfreq(W, dt)
#    period = np.power(freq[1:], -1)
#    # conversion to days
#    period /= 3600.*24.
#    idx_fit = np.where((period > 0.01) & (period < 1.))[0]
#    per_log = np.log10(period[idx_fit])
#    sm = np.ones(30, dtype=np.float32)
#    sm /= sm.size
#    for i in range(w_days/2, 366-w_days/2): 
#        #print "---------- %i / %i ----------" %(i, 366-w_days/2)
#        i_days = int((i+0.5)*(3600.*24./dt)) # +0.5 to center on the middle of each day
#        FFT = np.abs(np.fft.rfft(T[i_days-W/2:i_days+W/2]))
#        FFT = np.convolve(FFT, sm, mode='same')[idx_fit]
#        s, intercept, r_value, p_value, std_err = stats.linregress(per_log, np.log10(FFT))
#        slopes[i-(w_days/2)] = s
#    return slopes
#
#def A(T0, L, fft=False, Wliss = 20):
#    """
#    A(T0, L) \n
#    Calculates the autocorrelation on 1 given position: T0.
#    """
#    w = T0.size - 2*L
#    t0 = T0[L:-L]
#    den_t0 = np.power(t0,2).sum()
#    A = np.zeros(2*L, dtype=np.float32)
#    #-------------------------------------------
#    #-------- FIRST STEP ---------
#    t1 = T0[:w]
#    num = np.sum(t0*t1)
#    den_t1 = np.power(t1,2).sum()
#    den = np.sqrt(den_t0 * den_t1)
#    A[0] = num/den
#    for i in range(1,2*L):
#        den_t1 -= np.power(t1[0],2) #rightward shift: get rid of the left side sample
#        t1 = T0[i:i+w]
#        den_t1 += np.power(t1[-1],2) #add the new right side sample
#        den = np.sqrt(den_t0 * den_t1)
#        num = np.sum(t0*t1)
#        A[i] = num/den
#    if fft:
#        FFT = np.fft.rfft(T0[L:-L] - T0[L:-L].mean())
#        if Wliss != 0:
#            liss = np.ones(Wliss, dtype=np.float32)
#            liss /= liss.size
#            FFT = np.convolve(FFT, liss, mode='same')
#        return A, np.abs(FFT)
#    else:
#        return A
##==================================================================================
##                        PRINT FUNCTIONS
##==================================================================================
#
#def print_waveforms(wav, stations):
#    ns = wav.shape[0]
#    nc = wav.shape[1]
#    plt.figure()
#    for s in range(ns):
#        for c in range(nc):
#            plt.subplot(ns, nc, s*nc+c+1)
#            plt.plot(wav[s,c,:], label='%s' %stations[s])
#            plt.legend(loc='upper right')
#    plt.show()
#
#def print_family(family, station):
#    from scipy.signal import hilbert
#    ns = len(family[0].stations_family)
#    nc = len(family[0].components)
#    nd = len(family)
#    print nd, ns, nc
#    s_idx = family[0].stations_family.index(station)
#    time = np.arange(family[0].traces[0].data.size)
#    fig=plt.figure('station_%s' %station)
#    try:
#        dloc = family[0].d_best
#    except:
#        dloc = 0
#    plt.suptitle('Family of detections originating from %.2f/%.2f/%.2fkm, on station %s' %(family[dloc].latitude, \
#                                                                                           family[dloc].longitude, \
#                                                                                           family[dloc].depth, \
#                                                                                           station))
#    for d in range(nd):
#        for c in range(nc):
#            data_toplot = family[d].traces.select(station=station)[c].data
#            ax = fig.add_subplot(nd,nc,d*nc+c+1)
#            if d == 0:
#                ax.set_title('Component %s' %family[0].components[c])
#            if family[d].CC[s_idx] == 1.0:
#                clr = 'r'
#            else:
#                clr = 'b'
#            plt.plot(time, data_toplot, label='%s' %(family[d].start_time.strftime('%Y-%m-%d/%H:%M:%S')), color=clr)
#            xpos = np.arange(0, time.size, np.int32(1.*autodet.cfg.sampling_rate))
#            xticks = [str(float(X)/autodet.cfg.sampling_rate) for X in xpos]
#            plt.xticks(xpos, xticks)
#            plt.text(0.1, 0.4, 'CC=%.2f' %family[d].CC[s_idx], \
#                    fontweight='bold',\
#                    transform=ax.transAxes,\
#                    bbox={'alpha':0.5, 'facecolor':'white'})
#            ax.yaxis.set_ticks([data_toplot.min(), 0., data_toplot.max()])
#            if c == 0:
#                plt.legend(loc='upper left', fancybox=True, bbox_to_anchor=(-0.7, 1.2), handlelength=0.1)
#            if d == nd-1:
#                plt.xlabel('Time (s)')
#    plt.subplots_adjust(left=0.17, right=0.86)
#    plt.show()
#
#def print_template(idx, db_path_T='TEMPLATES_V1_STEP1/', db_path=autodet.cfg.dbpath, CC_comp=False, mv_view=True, show=True):
#    font = {'family' : 'serif', 'weight' : 'normal', 'size' : 14}
#    plt.rc('font', **font)
#    template = autodet.db_h5py.read_template('template%i' %idx, db_path=db_path+db_path_T)
#    sta = list(template.stations)
#    sta.sort()
#    ns = len(template.stations)
#    nc = len(template.traces)/ns
#    if CC_comp:
#        CC = np.zeros(ns, dtype=np.float32)
#        from scipy.signal import hilbert
#        for s in range(ns):
#            H = []
#            num = np.ones(template.select(station=sta[s])[0].data.size, dtype=np.float32)
#            den = 1.
#            for c in range(nc):
#                H.append(np.abs(hilbert(template.select(station=sta[s])[c].data)))
#                if np.var(H[-1]) == 0.:
#                    H[-1] = np.ones(len(H[-1]), dtype=np.float32)
#                num *= H[-1]
#                den *= np.power(H[-1], 3).sum()
#            num = num.sum()
#            den = np.power(den, 1./3.)
#            if den != 0.:
#                CC[s] = num/den
#    plt.figure('TEMPLATE_%i from %s' %(idx, db_path+db_path_T))
#    if mv_view:
#        time = np.arange(template.traces[0].data.size + np.int32(template.s_moveout.max()*autodet.cfg.sampling_rate))
#    else:
#        time = np.arange(template.traces[0].data.size)
#    for s in range(ns):
#        for c in range(nc):
#            plt.subplot(ns,nc,s*nc+c+1)
#            lab = '%s.%s' %(template.select(station=sta[s])[c].stats['station'],template.select(station=sta[s])[c].stats['channel'])
#            if CC_comp:
#                lab += ' %.2f' %CC[s]
#            if mv_view:
#                id1 = np.int32(template.s_moveout[template.stations.index(sta[s])] * autodet.cfg.sampling_rate)
#                id2 = id1 + template.traces[0].data.size
#                plt.plot(time[id1:id2], template.select(station=sta[s])[c].data, label=lab)
#            else:
#                plt.plot(time, template.select(station=sta[s])[c].data, label=lab)
#            #plt.axvline(time[time.size/2], color='k', ls='--')
#            plt.xlim((time[0], time[-1]))
#            plt.yticks([])
#            plt.xticks([])
#            plt.legend(loc='upper left', frameon=False, handlelength=0.1, borderpad=0.)
#            if s == ns-1:
#                plt.xlabel('Time (s)')
#                xpos = np.arange(0, time.size, np.int32(2.*autodet.cfg.sampling_rate))
#                xticks = [str(float(X)/autodet.cfg.sampling_rate) for X in xpos]
#                plt.xticks(xpos, xticks)
#    plt.subplots_adjust(bottom = 0.06, top = 0.94, hspace = 0.04, wspace = 0.12)
#    plt.suptitle('Template %i, location: lat %.2f, long %.2f, depth %.2fkm' %(template.template_ID, template.latitude, template.longitude, template.depth), fontsize=24)
#    if show:
#        plt.show()
#
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

#def print_tp_evol(tid, firstSTEP=0, lastSTEP=5, version='V1', db_path=autodet.cfg.dbpath):
#    T = []
#    for i in range(firstSTEP, lastSTEP+1):
#        db_path_T = 'TEMPLATES_%s_Z%i/' %(version, i)
#        T.append(autodet.db_h5py.read_template('template%i' %tid, db_path=db_path+db_path_T))
#    I = set(T[0].stations).intersection(T[1].stations)
#    for i in range(firstSTEP+1, lastSTEP):
#        I = I.intersection(T[i].stations)
#    print "Stations common to all the steps: ", I
#    station = str(input('Which one do you choose to print ?'))
#    print "Start printing station %s ..." %station
#    plt.figure('evolution_tp%i' %tid)
#    plt.suptitle('Evolution of Template %i / Station %s during the iterative matched-filter search' %(tid, station), fontsize=24)
#    Nrow = lastSTEP+1-firstSTEP
#    Ncol = len(T[-1].components)
#    for i in range(Nrow):
#        for c in range(Ncol):
#            plt.subplot(Nrow, Ncol, i*Ncol + c + 1)
#            if i == 0:
#                plt.title(T[0].components[c])
#            data = T[i].select(station=station, channel=T[i].components[c])[0].data
#            time = np.arange(0., data.size/autodet.cfg.sampling_rate, 1./autodet.cfg.sampling_rate)
#            plt.plot(time, data, label = 'Step %i' %i)
#            if c == Ncol-1:
#                plt.legend(loc='upper right', fancybox=True, bbox_to_anchor=(1.3, 0.9), handlelength=0.1)
#            if i == Nrow-1:
#                plt.xlabel('Time (s)', fontsize=24)
#    plt.subplots_adjust(left=0.05, right=0.92)
#    plt.show()
#
#def print_kurtosis_template(T):
#    ns = len(T.stations)
#    nc = len(T.components)
#    K = np.zeros((ns, nc), dtype=np.float32)
#    for s in range(ns):
#        for c in range(nc):
#            K[s,c] = kurtosis(T.traces[s*nc+c].data)
#    print "Mean kurtosis for template %i: %.2f" %(T.template_ID, K.mean())
#    return K
#
#def print_meta_template(version, STEP):
#    """
#    print_meta_template(version, STEP) \n
#    """
#    STEP = int(STEP)
#    if STEP == 1:
#        db_path_T = 'TEMPLATES_%s/' %version
#    else:
#        db_path_T = 'TEMPLATES_%s_Z%i/' %(version, STEP-1)
#    list_templates = '/nobackup1/ebeauce/CIFALPS/DETECTION_ALPS/templates_%s.txt' %version
#    tp_list = []
#    with open(list_templates, 'r') as f:
#        line = f.readline()[:-1]
#        while line != '':
#            tp_list.append(line[line.rfind('/')+1:].strip())
#            line = f.readline()[:-1]
#    templates = []
#    for tp in tp_list:
#        templates.append(autodet.db_h5py.read_template(tp, db_path=autodet.cfg.dbpath + db_path_T))
#        print "Template %i: Latitude: %.2f, longitude: %.2f, depth: %.2fkm" %(templates[-1].template_ID,\
#                                                                              templates[-1].latitude,\
#                                                                              templates[-1].longitude,\
#                                                                              templates[-1].depth)
#
#def print_count_multi(template_list, version, STEP, type_thrs='MAD'):
#    from subprocess import Popen, PIPE
#    path_to_file1 = '/nobackup1/ebeauce/CIFALPS/DETECTION_ALPS/%s' %template_list
#    tp_IDs = []
#    with open(path_to_file1, 'r') as f:
#        line = f.readline()[:-1]
#        while line != '':
#            tp_IDs.append(int(line[len('template'):].strip()))
#            line = f.readline()[:-1]
#    db_path = autodet.cfg.dbpath+'MULTIPLETS_%s_Z%i_MAD/' %(version, STEP)
#    for template_ID in tp_IDs:
#        print "Looking for %s%i_*" %(db_path+'*multiplets', template_ID)
#        files_list = Popen('ls '+db_path+'*multiplets%i_*meta.h5' %template_ID, stdout=PIPE, shell=True)
#        files = []
#        while True:
#            line = files_list.stdout.readline()[:-1]
#            if line == '':
#                break
#            else:
#                files.append(str(line))
#        Nmulti = 0
#        for i in range(len(files)):
#            with h5.File(files[i], mode='r') as fm0:
#                Nmulti += len(fm0['origin_times'][:])
#        # correction: add number of detections to the stacks database
#        S = autodet.db_h5py.read_stack_multiplets('stack%i' %template_ID, db_path_S='STACKS_%s_Z%i_%s/' %(version, STEP, type_thrs))
#        S.Nmulti = Nmulti
#        autodet.db_h5py.write_stack_multiplets('stack%i' %template_ID, S, db_path=autodet.cfg.dbpath+'STACKS_%s_Z%i_%s/' %(version, STEP, type_thrs))
#        print "%i detections associated to template %i" %(Nmulti, template_ID)
#
#
def plot_match(filename, index, db_path_T='template_db_1/', db_path_M='matched_filter_1/', db_path=autodet.cfg.dbpath, show=True):
    """
    plot_match(filename, index, db_path_T='template_db_1/', db_path_M='matched_filter_1/', db_path=autodet.cfg.dbpath, show=True)\n
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

#def print_multi_ZOOM(ID, version='V1', STEP=4, type_thrs='RMS', db_path=autodet.cfg.dbpath, show=True):
#    """
#    print_multi(ID, version='V1', STEP=5, type_thrs='MAD', db_path=autodet.cfg.dbpath) \n
#    ID is either [filename, inner idx] or [tp_ID, global idx]
#    """
#    font = {'family' : 'serif', 'weight' : 'normal', 'size' : 14}
#    plt.rc('font', **font)
#    plt.rcParams.update({'ytick.labelsize'  :  8})
#    plt.rcParams['pdf.fonttype'] = 42 #TrueType
#    filename = ID[0]
#    idx = ID[1]
#    db_path_T = 'TEMPLATES_%s_STEP%i/' %(version, STEP-1)
#    db_path_M = 'MULTIPLETS_%s_STEP%i_%s/' %(version, STEP, type_thrs)
#    M, T = autodet.db_h5py.read_multiplet(filename, idx, return_tp=True, db_path=db_path, db_path_T=db_path_T, db_path_M=db_path_M)
#    st_sorted = np.asarray(T.metadata['stations'])
#    st_sorted = np.sort(st_sorted)
#    I_s = np.argsort(T.metadata['stations'])
#    ns = len(T.metadata['stations'])
#    nc = len(M.components)
#    print nc,ns
#    plt.figure('multiplet%i_%s_TP%i' %(idx, M[0].stats.starttime.strftime('%Y-%m-%d'), M.template_ID))
#    t = np.arange(T[0].data.size)
#    plt.suptitle('Template %i (%.2f/%.2f/%.2fkm): Detection on %s' %(M.template_ID,\
#                                                                     M.latitude,\
#                                                                     M.longitude,\
#                                                                     M.depth,\
#                                                                     M[0].stats.starttime.strftime('%Y-%m-%d/%H:%M:%S')))
#    try:
#        mv = T.s_moveout
#        mv = np.repeat(mv, nc).reshape(ns, nc)
#    except:
#        mv = np.hstack((T.metadata['s_moveouts'].reshape(-1,1),
#                        T.metadata['s_moveouts'].reshape(-1,1),
#                        T.metadata['p_moveouts'].reshape(-1,1)))
#    for s in range(ns):
#        for c in range(nc):
#            plt.subplot(ns, nc, s*nc+c+1)
#            idx1 = np.int32(min(10., autodet.cfg.multiplet_len/4.)*autodet.cfg.sampling_rate ) + mv[I_s[s],c]
#            idx2 = idx1 + T[0].data.size
#            if idx2 > M[0].data.size:
#                idx2 = M[0].data.size
#            try:
#                t2 = np.arange(idx1,idx2)
#                plt.plot(M.select(station=st_sorted[s])[c].data[idx1:idx2], color='C0', label='%s.%s' %(st_sorted[s], M.components[c][-1]), lw=0.5)
#                Max = M.select(station=st_sorted[s])[c].data[idx1:idx2].max()
#                if Max == 0.:
#                    Max = 1.
#                data_toplot = T.select(station=st_sorted[s])[c].data[:t2.size]/T.select(station=st_sorted[s])[c].data[:t2.size].max() * Max
#                plt.plot(data_toplot, color='C2', lw=1.0, alpha=1)
#            except IndexError:
#                # more components in multiplets than in template
#                pass
#            except ValueError:
#                # empty slice
#                pass
#            xpos = np.arange(0, t.size, np.int32(2.*autodet.cfg.sampling_rate))
#            plt.xlim(xpos.min(), t.size)
#            if s == ns-1: 
#                xticks = [str(float(X)/autodet.cfg.sampling_rate) for X in xpos]
#                plt.xticks(xpos, xticks)
#                plt.xlabel('Time (s)')
#            else:
#                plt.xticks([])
#            #plt.legend(loc='upper left', frameon=False, handlelength=0.1, borderpad=0.)
#            plt.legend(loc='upper left', framealpha=1., handlelength=0.1, borderpad=0.2)
#    plt.subplots_adjust(bottom = 0.06, top = 0.94, hspace = 0.04, wspace = 0.16)
#    if show:
#        plt.show()
#
#def print_multi_stack(idx, version, STEP, type_thrs='MAD', best=False):
#    """
#    print_multi_stack(idx, version, STEP, type_thrs='MAD', best=False) \n
#    idx = index of the template's multiplets
#    """
#    from os.path import isfile
#    font = {'family' : 'serif', 'weight' : 'normal', 'size' : 14}
#    plt.rc('font', **font)
#    db_path_M = 'MULTIPLETS_%s_Z%i_%s/' %(version, STEP, type_thrs)
#    db_path_T = 'TEMPLATES_%s_Z%i/' %(version, STEP-1)
#    if isfile(autodet.cfg.dbpath+'STACKS_%s_Z%i_%s/stack%imeta.h5' %(version, STEP, type_thrs, idx)):
#        M = autodet.db_h5py.read_stack_multiplets('stack%i' %idx, db_path_S='STACKS_%s_Z%i_%s/' %(version, STEP, type_thrs))
#    else:
#        M = stack_multiplets(idx, db_path=autodet.cfg.dbpath+db_path_M, best=best)
#    #M.filter('bandpass', freqmin=autodet.cfg.freq_bands[0][0], freqmax=20., zerophase=True)
#    T = autodet.db_h5py.read_template('template%i' %M.template_ID, db_path=autodet.cfg.dbpath+db_path_T)
#    ns = len(T.stations)
#    nc = len(M.components)
#    print nc,ns
#    fig = plt.figure('stack_TP%i_%s' %(M.template_ID, type_thrs))
#    t = np.arange(M[0].stats.npts)
#    plt.suptitle('Multiplets from the source at %.2f/%.2f/%.2fkm, template %i (%i multiplets)' %(M.latitude,\
#                                                                                                 M.longitude,\
#                                                                                                 M.depth,\
#                                                                                                 T.template_ID,\
#                                                                                                 M.Nmulti), fontsize=24)
#    for s in range(ns):
#        for c in range(nc):
#            plt.subplot(ns, nc, s*nc+c+1)
#            plt.plot(t, M.select(station=T.stations[s])[c].data, color='b', label='%s.%s' %(T.stations[s], M.components[c]))
#            idx1 = np.int32( (autodet.cfg.multiplet_len/4.+T.s_moveout[s])*autodet.cfg.sampling_rate )
#            idx2 = idx1 + T[0].data.size
#            if idx2 > t.size:
#                idx2 = t.size
#            try:
#                t2 = np.arange(idx1,idx2)
#                Max = M.select(station=T.stations[s])[c].data.max()
#                if Max == 0.:
#                    Max = 1.
#                data_toplot = T.select(station=T.stations[s])[c].data[:t2.size]/T.select(station=T.stations[s])[c].data[:t2.size].max() * Max
#                plt.plot(np.arange(idx1,idx2), data_toplot, color='r', ls='--')
#            except IndexError:
#                # more components in multiplets than in template
#                plt.axvline(idx1, color='r', ls='--')
#                plt.axvline(idx2, color='r', ls='--')
#            xpos = np.arange(0, t.size, np.int32(5.*autodet.cfg.sampling_rate))
#            xticks = [str(float(X)/autodet.cfg.sampling_rate) for X in xpos]
#            plt.xticks(xpos, xticks)
#            plt.legend(loc='upper left', frameon=False, handlelength=0.1, borderpad=0.)
#            if s == ns-1:
#                plt.xlabel('Time (s)')
#    plt.subplots_adjust(bottom = 0.06, top = 0.94, hspace = 0.04, wspace = 0.12)
#    plt.show()
#
#def print_best_multi(tp_ID, version, STEP, type_thrs='MAD'):
#    OT, C, CC = temporal_distribution(tp_ID, 3600.*24., version=version, STEP=STEP, type_thrs=type_thrs, corr=True)
#    ot = read_RE_IDs(tp_ID, version, STEP, type_thrs=type_thrs)
#    meta_ev = ot[CC.argmax()]
#    print "Metadata of the best detection: ", meta_ev
#    print_multi([meta_ev[1], meta_ev[2]], version=version, STEP=STEP, type_thrs=type_thrs)
#    plt.show()
#
#def print_multi_red(Ms, Mi): 
#    font = {'family' : 'serif', 'weight' : 'normal', 'size' : 14}
#    plt.rc('font', **font)
#    plt.figure('Union_families')
#    SNR_best = np.argsort(np.sum(Ms.SNR, axis=-1))[-12:]
#    stations = np.asarray(Ms.stations)[SNR_best].tolist()
#    ns = len(stations)
#    nc = len(Ms.components)
#    for s in range(ns):
#        for c in range(nc):
#            plt.subplot(ns, nc, s*nc+c+1)
#            data_s = Ms.select(station=stations[s], channel=Ms.components[c])[0].data
#            data_i = Mi.select(station=stations[s], channel=Ms.components[c])[0].data
#            plt.plot(data_i, color='k')
#            plt.plot(data_s, color='g', label='%s.%s \n SNR: %.2f \n -> %.2f' %(stations[s], Ms.components[c], Mi.SNR[SNR_best[s],c], Ms.SNR[SNR_best[s],c]))
#            plt.legend(loc='upper right', fancybox=True, handlelength=0.1, borderpad=0.1)
#    plt.subplots_adjust(right=0.98, left=0.04, bottom = 0.06, top = 0.94, hspace = 0.04, wspace = 0.12)
#    plt.suptitle('TP %i (%.2f/%.2f/%.2fkm) and TP %i (%.2f/%.2f/%.2fkm): SNR %.2f -> %.2f' \
#                 %(Ms.template_ID1, Ms.latitude1, Ms.longitude1, Ms.depth1, \
#                   Ms.template_ID2, Ms.latitude2, Ms.longitude2, Ms.depth2, \
#                   Mi.SNR.mean(), Ms.SNR.mean()))
#    plt.show()
#
#def print_test_relocSP(idx, net, MV, Ns=12, db_path_M='MULTIPLETS_Z1_', db_path_T='TEMPLATES/', db_path_S=None, band=None, type_thrs=None, best=False, LG=False):
#    """
#    print_multi(M, T) \n
#    """
#    font = {'family' : 'serif', 'weight' : 'normal', 'size' : 14}
#    plt.rc('font', **font)
#    if type_thrs is not None:
#        T, M, SNR_dic = reloc_SP2(idx, net, MV, Ns=Ns, db_path_M=db_path_M, db_path_T=db_path_T, db_path_S=db_path_S, band=band, type_thrs=type_thrs, best=best, long_template=LG)
#    else:
#        T, M, SNR_dic = reloc_SP(idx, net, MV, db_path_M=db_path_M, db_path_T=db_path_T, db_path_S=db_path_S, band=band)
#    stations = T.stations
#    idx_sort = np.argsort(stations)
#    #stations = T.stations_reloc
#    ns = len(stations)
#    nc = len(T.components)
#    T_old = autodet.db_h5py.read_template('template%i' %T.template_ID, db_path=autodet.cfg.dbpath+db_path_T)
#    tp_len = T_old[0].data.size
#    fig = plt.figure('stacked_multiplets_from_TP%i' %T.template_ID)
#    t = np.arange(M[0].stats.npts)
#    plt.suptitle('%i Multiplets from template %i, relocalisation: %.2f/%.2f/%.2fkm --> %.2f/%.2f/%.2fkm' %(M.Nmulti, T.template_ID,\
#                                                                                                     M.latitude,M.longitude,M.depth,\
#                                                                                                     T.latitude,T.longitude,T.depth), fontsize=24)
#    for s in range(ns):
#        for c in range(nc):
#            plt.subplot(ns, nc, s*nc+c+1)
#            #if c == 2:
#            #    M.select(station=stations[idx_sort[s]])[c].filter('bandpass', freqmin=4., freqmax=20., zerophase=True)
#            #else:
#            #    M.select(station=stations[idx_sort[s]])[c].filter('bandpass', freqmin=1.5, freqmax=9., zerophase=True)
#            NORM = np.sqrt(np.var(M.select(station=stations[idx_sort[s]])[c].data))
#            plt.plot(t, M.select(station=stations[idx_sort[s]])[c].data/NORM, color='b', label='%s.%s' %(stations[idx_sort[s]], T.components[c]))
#            #plt.plot(t, M.select(station=stations[idx_sort[s]])[c].data, color='b', label='%s.%s' %(stations[idx_sort[s]], T.components[c]))
#            idx_S = T.time + np.int32(T.s_moveout_p[idx_sort[s]] * autodet.cfg.sampling_rate)
#            idx_P = T.time + np.int32(T.p_moveout[idx_sort[s]] * autodet.cfg.sampling_rate)
#            id1 = idx_S - tp_len/2
#            id2 = idx_S + tp_len/2
#            plt.axvline(idx_P, ls='--', color='g')
#            plt.axvline(idx_S, ls='--', color='r')
#            plt.axvline(id1, color='k')
#            plt.axvline(id2, color='k')
#            xpos = np.arange(0, t.size, np.int32(5.*autodet.cfg.sampling_rate))
#            xticks = [str(float(X)/autodet.cfg.sampling_rate) for X in xpos]
#            plt.xticks(xpos, xticks)
#            if s == ns-1:
#                plt.xlabel('Time (s)')
#            plt.legend(loc='upper left', frameon=False, handlelength=0.1, borderpad=0.)
#    plt.subplots_adjust(bottom = 0.06, top = 0.94, hspace = 0.04, wspace = 0.12)
#    plt.figure('SNR')
#    idx = np.argsort(SNR_dic.values())[::-1]
#    snr = np.asarray(SNR_dic.values())
#    labels_pos = range(idx.size)
#    labels = []
#    for s in range(idx.size):
#        labels.append(SNR_dic.keys()[idx[s]])
#    plt.plot(snr[idx], color='b', marker='o', lw=2)
#    plt.axhline(M.SNR_thrs, color='g', lw=2, label='SNR threshold')
#    plt.xticks(labels_pos, labels, rotation=45)
#    plt.grid(axis='x')
#    plt.ylabel('SNR')
#    plt.legend(loc='best')
#    plt.show()
#
#def print_reloc_manual(tp_idx, net, MV, version, STEP, repicking=False, write_template=False, normRMS=False, mode=2, show=True):
#    from matplotlib.colors import LogNorm
#    db_path_M = 'MULTIPLETS_%s_STEP%i_' %(version, STEP)
#    db_path_T = 'TEMPLATES_%s_STEP%i/' %(version, STEP-1)
#    S, mvP, mvS, RMS, stations, t0 = reloc_manual(tp_idx, net, MV, version, STEP, repicking=repicking, normRMS=normRMS, mode=mode)
#    with open(autodet.cfg.picks_path+'%s/STEP%i/picksP_tp%i.dat' %(version, STEP, tp_idx), 'rb') as f:
#        picksP = np.fromfile(f, dtype=np.int32).reshape((-1,2))
#    with open(autodet.cfg.picks_path+'%s/STEP%i/picksS_tp%i.dat' %(version, STEP, tp_idx), 'rb') as f:
#        picksS = np.fromfile(f, dtype=np.int32).reshape((-1,2))
#    with open(autodet.cfg.picks_path+'%s/STEP%i/stations_tp%i.dat' %(version, STEP, tp_idx), 'r') as f:
#        stations_reloc = np.loadtxt(f, dtype='|S4').tolist()
#    # ---- find argmin
#    idx_st = []
#    for s in range(min(len(stations), len(stations_reloc))):
#        idx_st.append(stations_reloc.index(stations[s]))
#    idx_st = np.int32(idx_st) # maps the picks' indices to the stations' indices finally used
#    print picksP
#    print picksS
#    T = autodet.db_h5py.read_template('template%i' %tp_idx, db_path=autodet.cfg.dbpath+db_path_T)
#    M = stack_multiplets(tp_idx, db_path=autodet.cfg.dbpath+db_path_M+'RMS/')
#    plt.figure('manual_relocation')
#    t = np.arange(M[0].data.size)
#    ns = len(stations)
#    nc = len(T.components)
#    #t0 = np.int32(picksP[idx_st,0][mvP.argmin()])
#    print "t0 print = %i" %t0
#    for s in range(ns):
#        for c in range(nc):
#            plt.subplot(ns, nc, s*nc+c+1)
#            plt.plot(t, M.select(station=stations[s])[c].data, color='b', label='%s.%s' %(stations[s], M.components[c]))
#            try:
#                dP = np.abs(picksP[idx_st[s],0] - picksP[idx_st[s],1])
#                idP = t0 + mvP[s]
#                idS = t0 + mvS[s]
#                plt.axvline(idP, ls='--', color='g', lw=2)
#                plt.axvline(picksP[idx_st[s],0]-dP, ls='-.', color='g', lw=1)
#                plt.axvline(picksP[idx_st[s],1]+dP, ls='-.', color='g', lw=1)
#                plt.axvline(idS, ls='--', color='r', lw=2) 
#                dS = np.abs(picksS[idx_st[s],0] - picksS[idx_st[s],1])
#                plt.axvline(picksS[idx_st[s],0]-dS, ls='-.', color='r', lw=1)
#                plt.axvline(picksS[idx_st[s],1]+dS, ls='-.', color='r', lw=1)
#            except IndexError:
#                # less than 12 stations have been used for relocation
#                pass
#            xpos = np.arange(0, t.size, np.int32(5.*autodet.cfg.sampling_rate))
#            xticks = [str(float(X)/autodet.cfg.sampling_rate) for X in xpos]
#            plt.xticks(xpos, xticks)
#            plt.legend(loc='upper left', frameon=False, handlelength=0.1, borderpad=0.)
#            if s == ns-1:
#                plt.xlabel('Time (s)')
#    plt.subplots_adjust(bottom = 0.06, top = 0.94, hspace = 0.04, wspace = 0.12)
#    plt.figure('objective_function')
#    plt.pcolormesh(RMS, norm=LogNorm())
#    Pmin = np.where(RMS == RMS.min())
#    print Pmin
#    plt.plot(Pmin[1], Pmin[0], marker='o', color='r', markersize=5)
#    plt.colorbar()
#    plt.xlim((0, RMS.shape[1]-1))
#    plt.ylim((0, RMS.shape[0]-1))
#    if show:
#        plt.show()
#    if write_template:
#        import copy
#        mvS_min = mvS.min()
#        T2 = copy.deepcopy(T)
#        waveforms = np.zeros((ns, nc, T2[0].data.size), dtype=np.float32)
#        for s in range(ns):
#            for c in range(nc):
#                id1 = t0 + mvS[s] - np.int32((autodet.cfg.template_len/2. + 2.) * autodet.cfg.sampling_rate)
#                id2 = id1 + np.int32((autodet.cfg.template_len + 3.) * autodet.cfg.sampling_rate)
#                if id2 > t.size:
#                    Nzeros = id2-t.size
#                    print Nzeros, id1, id2, stations[s]
#                    waveforms[s,c,:] = np.hstack((M.select(station=stations[s])[c].data[id1:], np.zeros(Nzeros, dtype=np.float32)))
#                elif id1 < 0:
#                    Nzeros = -id1
#                    print Nzeros
#                    waveforms[s,c,:] = np.hstack((np.zeros(Nzeros, dtype=np.float32), M.select(station=stations[s])[c].data[:id2]))
#                else:
#                    waveforms[s,c,:] = M.select(station=stations[s])[c].data[id1:id2]
#                T2[s*nc+c].data = waveforms[s,c,:]
#            T2.s_moveout[s] = np.float32(mvS[s] - mvS_min)/autodet.cfg.sampling_rate
#        T2.stations = stations
#        T2.p_moveout = np.float32(mvP) / autodet.cfg.sampling_rate
#        T2.s_moveout_p = np.float32(mvS) / autodet.cfg.sampling_rate
#        T2.latitude = MV.latitude[S[0], S[1], S[2]]
#        T2.longitude = MV.longitude[S[0], S[1], S[2]]
#        T2.depth = MV.depth[S[0], S[1], S[2]]
#        T2.source_idx = S[0] * (MV.depth.shape[1] * MV.depth.shape[2]) + S[1] * MV.depth.shape[2] + S[2]
#        T2.start_time = udt(0)
#        autodet.db_h5py.write_template('template%i' %T2.template_ID, T2, waveforms, db_path=autodet.cfg.dbpath+'TEMPLATES_%s_Z%i/' %(version, STEP))
#
#def print_statistic_2tps(tid1, tid2, version='V1', STEP=5, type_thrs='MAD', db_path=autodet.cfg.dbpath):
#    db_path_M = 'MULTIPLETS_%s_Z%i_%s/' %(version, STEP, type_thrs)
#    cat1 = autodet.db_h5py.read_catalog_multiplets('multiplets%i' %tid1, db_path=db_path, db_path_M=db_path_M)
#    cat2 = autodet.db_h5py.read_catalog_multiplets('multiplets%i' %tid2, db_path=db_path, db_path_M=db_path_M)
#    plt.figure('tps_%i_%i' %(tid1, tid2))
#    plt.plot(cat1['origin_times'], cat1['magnitudes'], marker='o', ls='', color='b', label='Template %i' %tid1)
#    plt.plot(cat2['origin_times'], cat2['magnitudes'], marker='v', ls='', color='r', label='Template %i' %tid2)
#    plt.legend(loc='best')
#    plt.show()
#
##==================================================================================
##                       PLOT TEMPORAL DISTRIBUTION
##==================================================================================
#def plot_CC(TP_IDX, DD, net, path_tp='TEMPLATES_V1_Z0/', light=False, db_path=autodet.cfg.dbpath, \
#            returnCC=False, show=True, device='cpu'): 
#    font = {'family' : 'serif', 'weight' : 'bold', 'size' : 24}
#    plt.rc('font', **font)
#    path_to_templates = db_path + path_tp
#    CC = autodet.multiplet_search.CC_1day('template%i' %TP_IDX, path_to_templates, DD, net, best_St=False, device=device)
#    buf = np.int32(autodet.cfg.data_buffer * autodet.cfg.sampling_rate)
#    CC = CC[0][buf:-buf]
#    moy = CC.mean()
#    med = np.median(CC)
#    print "Moy: %.2f, Med: %.2f" %(moy, med)
#    plt.figure('CC_tp%i_%s' %(TP_IDX, DD.date.strftime('%Y-%m-%d')))
#    t = np.arange(CC.size, dtype=np.float32)
#    t /= (autodet.cfg.sampling_rate * 3600.)
#    plt.subplot(1,2,1)
#    if light:
#        plt.plot(t[::2], CC[::2])
#    else:
#        plt.plot(t, CC)
#    th_RMS = 8. * np.sqrt(np.var(CC))
#    th_MAD = 8. * autodet.common.mad(CC)
#    nRMS = np.where( (CC - moy) > th_RMS)[0].size
#    nMAD = np.where( (CC - med) > th_MAD )[0].size
#    #------------------------------------------------
#    for idx in np.where( (CC - moy) > th_RMS)[0]:
#        nsec = np.float32(idx)/autodet.cfg.sampling_rate - autodet.cfg.data_buffer
#        hour = nsec // (3600.)
#        minutes = (nsec % 3600.)/60.
#        seconds = nsec % 60.
#        print "%.0f:%.0f:%.2f" %(hour, minutes, seconds)
#    #------------------------------------------------
#    plt.axhline(th_RMS, color='r', lw=2, ls='--', label='8 x RMS: %i multiplets' %nRMS)
#    plt.axhline(th_MAD, color='g', lw=2, ls='--', label='8 x MAD: %i multiplets' %nMAD)
#    #plt.xlim((0., 24.))
#    plt.xlim((-1., 25.))
#    if np.abs(CC).max() < 0.25:
#        plt.ylim((-0.5,0.5))
#    plt.xlabel('Time (hours)')
#    plt.ylabel('Coherency coefficient')
#    plt.legend(loc='best')
#    plt.subplot(1,2,2)
#    n, bins, patches = plt.hist(CC, 300, normed=True)
#    plt.axvline(th_RMS, color='r', lw=2, ls='--', label='8 x RMS')
#    plt.axvline(th_MAD, color='g', lw=2, ls='--', label='8 x MAD')
#    plt.xlim((-0.2, 0.2))
#    plt.xlabel('Coherency coefficient')
#    plt.ylabel('Density of probability')
#    plt.legend(loc='best')
#    if show:
#        plt.show()
#    if returnCC:
#        return CC
#
#def plot_cumul(template_ID, DT, version='V1', STEP=5, type_thrs='RMS', db_path=autodet.cfg.dbpath):
#    from obspy import UTCDateTime as udt 
#    font = {'family' : 'serif', 'weight' : 'bold', 'size' : 24}
#    plt.rc('font', **font)
#    OT, C, corr = temporal_distribution(template_ID, DT, version=version, STEP=STEP, type_thrs=type_thrs, corr=True, db_path=db_path)
#    C_cumul = np.zeros(C.size, dtype=np.float32)
#    C_cumul[0] = C[0]
#    for i in range(1, C.size):
#        C_cumul[i] = C_cumul[i-1] + C[i]
#    C_cumul /= C.sum()
#    time = np.arange(C.size)
#    plt.figure('cumulative_event_count_TP%i' %template_ID)
#    plt.title('Cumulative event count (template %i\'s multiplets)' %template_ID)
#    plt.plot(time, C_cumul)
#    start_day = udt('2012,08,01')
#    time_scale = [start_day]
#    idx = []
#    n = 0
#    for t in np.arange(time.size):
#        T = udt(time_scale[-1] + n*DT)
#        if T.month == time_scale[-1].month:
#            n += 1
#            continue
#        else:
#            idx.append(t)
#            time_scale.append(T)
#            n = 0
#    tlabels = [T.strftime('%b%Y') for T in time_scale]
#    plt.xticks(idx, tlabels, rotation=45)
#    plt.grid(axis='x')
#    plt.show()
#
#def plot_freq_1TP(template_ID, version='V3', STEP=4, type_thrs='MAD', plotCC=False, plotLOC=False, db_path=autodet.cfg.dbpath):
#    from obspy import UTCDateTime as udt
#    from matplotlib.colors import Normalize, LogNorm
#    from matplotlib.cm import ScalarMappable
#    #matplotlib.rcParams['svg.fonttype'] = 'none'
#    import matplotlib.colorbar as clb
#    font = {'family' : 'serif', 'weight' : 'bold', 'size' : 24}
#    plt.rc('font', **font)
#    plt.rcParams['pdf.fonttype'] = 42 #TrueType
#    db_path_M = 'MULTIPLETS_%s_Z%i_' %(version, STEP)
#    OT, C, corr = temporal_distribution(template_ID, 3600.*24., version=version, STEP=STEP, type_thrs=type_thrs, corr=True, db_path=db_path)
#    corr /= np.median(corr)
#    if plotLOC:
#        import pickle
#        with open(autodet.cfg.scripts_path + 'RELOC_%s/reloc_multiplets%i.pickle' %(version, template_ID), 'rb') as f:
#            RL = pickle.load(f)
#        TS = RL[0].squeeze()
#        T = autodet.db_h5py.read_template('template%i' %template_ID, db_path=db_path+'TEMPLATES_V3_Z3/')
#        #Dt = np.float32(TS[:,np.argmin(T.s_moveout)])/autodet.cfg.sampling_rate
#        Dt = np.float32(TS[:,np.argmax(T.s_moveout)])/autodet.cfg.sampling_rate
#        #Dt = np.sum(np.power(TS, 2), axis=-1)
#        #print Dt.shape
#    if plotCC:
#        # make color scale
#        cm = plt.get_cmap('brg')
#        cNorm = LogNorm(vmin=corr.min(),vmax=2.)
#        scalarMap = ScalarMappable(norm=cNorm,cmap=cm)
#        colors = scalarMap.to_rgba(corr)
#    elif plotLOC:
#        # make color scale
#        cm = plt.get_cmap('brg')
#        #Dt_med = np.median(Dt)
#        #cNorm = Normalize(vmin=-np.abs(Dt).max(),vmax=np.abs(Dt).max())
#        cNorm = Normalize(vmin=-2.*np.median(np.abs(Dt)),vmax=2.*np.median(np.abs(Dt)))
#        #cNorm = Normalize(vmin=-0.1, vmax=0.1)
#        scalarMap = ScalarMappable(norm=cNorm,cmap=cm)
#        colors = scalarMap.to_rgba(Dt)
#    else:
#        colors = ['k']*len(corr)
#    n = 1
#    C[np.where(C != 0)[0][0]] -= 1 #remove the first detection
#    print C.sum()
#    plt.figure('frequencies_tp%i' %template_ID)
#    plt.title('Temporal distribution of the template %i\'s multiplets (%i detections)' %(template_ID, C.sum()+1))
#    for t in range(1, C.size):
#        if C[t] == 0:
#            continue
#        if C[t] == 1:
#            plt.plot(t, (OT[n] - OT[n - 1])/(3600.*24.), marker='o', markerfacecolor='none', markeredgecolor=colors[n], markeredgewidth=2)
#            n += 1
#            continue
#        for i in range(C[t]):
#            plt.plot(t, (OT[n + i] - OT[n + i - 1])/(3600.*24.), marker='o', markerfacecolor='none', markeredgecolor=colors[n], markeredgewidth=2)
#        n += C[t]
#    start_day = udt('2012,08,01')
#    time_scale = [start_day]
#    idx = [1]
#    n = 0
#    for t in np.arange(C.size):
#        T = udt(time_scale[-1] + n*(3600.*24.))
#        if C[t] == C.max():
#            print "Max event count on %s (%i detections)" %(T.strftime('%Y-%m-%d'), C.max())
#        if T.month == time_scale[-1].month:
#            n += 1
#            continue
#        else:
#            idx.append(t)
#            time_scale.append(T)
#            n = 1
#    tlabels = [T.strftime('%b%Y') for T in time_scale]
#    plt.xticks(idx, tlabels, rotation=45)
#    plt.xlim((1, C.size))
#    plt.semilogy()
#    plt.grid(axis='x')
#    plt.ylabel('Recurrence interval (days)')
#    if plotCC:
#        ax, _ = clb.make_axes(plt.gca(), shrink=0.5)
#        cbar = clb.ColorbarBase(ax, cmap = cm, norm=cNorm, label='CC')
#    elif plotLOC: 
#        ax, _ = clb.make_axes(plt.gca(), shrink=0.5)
#        cbar = clb.ColorbarBase(ax, cmap = cm, norm=cNorm, label='Dt')
#    plt.show()

#def plot_autocorr(TP_IDX, day_crisis, day_quiet, W_day, DT, version='V1', STEP=5, type_thrs='MAD', L_hours = 5., smooth=False):
#    """
#    plot_autocorr(TP_IDX, day_crisis, day_quiet, W_day, DT, db_path_M='MULTIPLETS_V3_Z4_') \n
#    DT: temporal resolution of the event count temporal serie, in seconds
#    """
#    import datetime as dt
#    import mtspec
#    from scipy import stats
#    #----------------------------------------------------------------------
#    font = {'family' : 'serif', 'weight' : 'bold', 'size' : 20}
#    plt.rc('font', **font)
#    plt.rcParams['pdf.fonttype'] = 42 #TrueType
#    #----------------------------------------------------------------------
#    db_path_M = 'MULTIPLETS_%s_STEP%i_%s/' %(version, STEP, type_thrs)
#    W = int(W_day*3600.*24./DT) # time window over which is calculated the autocorrelation
#    L = int(L_hours*3600./DT) # max lag = 1hour
#    start_day = udt('2012,08,01')
#    end_day = udt('2013,08,01')
#    Nbins = int((end_day.timestamp - start_day.timestamp)/DT) + 1
#    #OT, C = temporal_distribution(TP_IDX, DT, db_path_M=db_path_M)
#    catalog = autodet.db_h5py.read_catalog_multiplets('multiplets%i' %TP_IDX, db_path_M=db_path_M)
#    C, times = np.histogram(catalog['origin_times'], bins=Nbins, range=(start_day.timestamp, end_day.timestamp))
#    s_sm = 1
#    C = gaussian_filter1d(np.float32(C), s_sm)
#    lower_relevant_period = s_sm * DT
#    #----------------------------
#    #----------------------------
#    day_crisis = udt(day_crisis)
#    ib_crisis = max(int((day_crisis.timestamp - start_day.timestamp)/DT)-W/2-L, 0)
#    ie_crisis = ib_crisis + W + 2*L
#    event_count_number_crisis = C[ib_crisis:ie_crisis]
#    A_crisis = A(event_count_number_crisis, L=L)
#    #----------------------------
#    day_quiet = udt(day_quiet)
#    idx_quiet = int((day_quiet.timestamp - start_day.timestamp)/DT)
#    event_count_number_quiet  = C[idx_quiet-W/2-L:idx_quiet+W/2+L]
#    A_quiet = A(event_count_number_quiet, L=L)
#    #---------------------------- 
#    psd_crisis, freq = mtspec.multitaper.mtspec(event_count_number_crisis, DT, 3, quadratic=True)
#    psd_quiet, freq  = mtspec.multitaper.mtspec(event_count_number_quiet,  DT, 3, quadratic=True)
#    FFT_crisis = np.sqrt(psd_crisis)
#    FFT_quiet = np.sqrt(psd_quiet)
#    period = np.power(freq[1:], -1.)
#    period /= (3600.*24.) # conversion to days
#    new_periods = log_to_lin(np.linspace(np.log10(period.min()), np.log10(period.max()), 200))
#    if smooth:
#        FFT_crisis =  smooth_resample_spectrum(FFT_crisis[1:], period, new_periods, gaussian_width=2)
#        FFT_quiet  =  smooth_resample_spectrum(FFT_quiet[1:],  period, new_periods, gaussian_width=2)
#    #--------------------------------------------------------------------------------
#    plt.figure('event_count_number_tp%i_%s' %(TP_IDX, day_crisis.strftime('%d%b%Y')))
#    time = np.arange(0., event_count_number_quiet.size*DT, DT)
#    plt.plot(time, event_count_number_crisis, color='C0', label='%s - %s' %((day_crisis - dt.timedelta(days=W_day/2)).strftime('%Y-%m-%d'),\
#                                                                            (day_crisis + dt.timedelta(days=W_day/2)).strftime('%Y-%m-%d')))
#    plt.plot(time, event_count_number_quiet, color='C2', label='%s - %s' %((day_quiet - dt.timedelta(days=W_day/2)).strftime('%Y-%m-%d'),\
#                                                                           (day_quiet + dt.timedelta(days=W_day/2)).strftime('%Y-%m-%d')))
#    tpos = np.arange(0., 10.*3600.*24., 3600.*24.)
#    tlabels = range(tpos.size)
#    plt.xticks(tpos, tlabels)
#    plt.xlabel('Time (days)')
#    plt.ylabel('Number of events per unit time')
#    plt.legend(loc='best', fancybox=True)
#    #--------------------------------------------------------------------------------
#    plt.figure('autocorrelation_tp%i_%s' %(TP_IDX, day_crisis.strftime('%d%b%Y')))
#    plt.subplot(1, 2, 1)
#    plt.plot(A_crisis, color='C0', label='%s - %s' %((day_crisis - dt.timedelta(days=W_day/2)).strftime('%Y-%m-%d'),\
#                                                    (day_crisis + dt.timedelta(days=W_day/2)).strftime('%Y-%m-%d')))
#    plt.plot(A_quiet, color='C2', label='%s - %s' %((day_quiet - dt.timedelta(days=W_day/2)).strftime('%Y-%m-%d'),\
#                                                    (day_quiet + dt.timedelta(days=W_day/2)).strftime('%Y-%m-%d')))
#    tpos = np.arange(0., 2*L+1, L/2)
#    tlabels = [(ticks*DT/3600. - L_hours) for ticks in tpos]
#    plt.xticks(tpos, tlabels)
#    plt.xlabel('Lag (hour)')
#    plt.ylabel('Event count autocorrelation')
#    plt.legend(loc='best', fancybox=True)
#    plt.subplot(1,2,2)
#    if smooth:
#        idx_reg = new_periods > 0.1
#        s, intercept, r_value, p_value, std_err = stats.linregress(np.log10(new_periods[idx_reg]), np.log10(FFT_crisis[idx_reg]))
#        plt.plot(new_periods, FFT_crisis, color='C0', label='%s - %s' %((day_crisis - dt.timedelta(days=W_day/2)).strftime('%Y-%m-%d'),\
#                                                                            (day_crisis + dt.timedelta(days=W_day/2)).strftime('%Y-%m-%d')))
#        plt.plot(new_periods, FFT_quiet, color='C2', label='%s - %s' %((day_quiet - dt.timedelta(days=W_day/2)).strftime('%Y-%m-%d'),\
#                                                                            (day_quiet + dt.timedelta(days=W_day/2)).strftime('%Y-%m-%d')))
#        straight_line = log_to_lin(intercept + s * np.log10(new_periods[idx_reg]))
#        plt.plot(new_periods[idx_reg], straight_line, color='r', ls='--', lw=2, label='slope = %.2f' %s)
#    else:
#        idx_reg = period > 0.1
#        s, intercept, r_value, p_value, std_err = stats.linregress(np.log10(period[idx_reg]), np.log10(FFT_crisis[1:][idx_reg]))
#        plt.plot(period, FFT_crisis[1:], color='C0', label='%s - %s' %((day_crisis - dt.timedelta(days=W_day/2)).strftime('%Y-%m-%d'),\
#                                                                            (day_crisis + dt.timedelta(days=W_day/2)).strftime('%Y-%m-%d')))
#        plt.plot(period, FFT_quiet[1:], color='C2', label='%s - %s' %((day_quiet - dt.timedelta(days=W_day/2)).strftime('%Y-%m-%d'),\
#                                                                            (day_quiet + dt.timedelta(days=W_day/2)).strftime('%Y-%m-%d')))
#        straight_line = log_to_lin(intercept + s * np.log10(period[idx_reg]))
#        plt.plot(period[idx_reg], straight_line, color='C3', ls='--', lw=2, label='slope = %.2f' %s)
#    plt.loglog()
#    plt.legend(loc='best', fancybox=True)
#    plt.xlim(0.01, W_day)
#    plt.xlabel('Period (days)')
#    plt.show()
#
#def plot_slopes(tp_ID, DT=3600.*24., version='V3'):
#    from obspy import UTCDateTime as udt
#    with open('/nobackup1/ebeauce/CIFALPS/DETECTION_ALPS/SLOPES_%s/slopes_%i.dat' %(version, tp_ID), 'rb') as f:
#        sl = np.fromfile(f, dtype=np.float32)
#    plt.plot(sl)
#    start_day = udt('2012,08,01')
#    std = start_day.timestamp
#    time_axis = [start_day]
#    n = 0
#    idx = [0]
#    for t in np.arange(366):
#        T = udt(time_axis[-1] + n*(3600.*24.))
#        if T.month == time_axis[-1].month:
#            n += 1
#            continue
#        else:
#            idx.append((T.timestamp - std)/DT)
#            time_axis.append(T)
#            n = 0
#    tlabels = [T.strftime('%b%Y') for T in time_axis]
#    plt.xticks(idx, tlabels, rotation=45)
#    plt.grid(axis='x')
#    plt.xlim((0, (udt('2013,08,02').timestamp - std)/DT))
#    plt.ylabel('Power-law exponent (spectrum\'s slope)')
#    plt.title('Degree of clustering of the source %i\'s events' %tp_ID)
#    plt.show()
#
##==================================================================================
##                                 DIVERS
##==================================================================================
#def log_to_lin(X_log):
#    X_lin = np.power(10.*np.ones(X_log.size, dtype=np.float64), X_log)
#    return X_lin
#
#def smooth_resample_spectrum(spectrum, initial_freq, new_freq, gaussian_width=4, interpolation_method='linear'):
#    """
#    smooth_resample_spectrum(spectrum, initial_freq, new_freq, gaussian_width, interpolation_order)\n
#    spectrum: initial spectrum,\n
#    initial_freq: frequencies at which the spectrum is initially evaluated,\n
#    new_freq: frequencies at which we want to re-sample the spectrum,\n
#    gaussian_width: standard deviation of the gaussian filter (scipy.ndimage.filteautodet.gaussian_filter),\n
#    interpolation_method: cf. scipy documentation: kind : str or int, optional
#        Specifies the kind of interpolation as a string ('linear', 'nearest', 'zero', 'slinear', 'quadratic', 'cubic' where 'zero', 'slinear', 'quadratic' and 'cubic' refer to a spline interpolation of zeroth, first, second or third order) or as an integer specifying the order of the spline interpolator to use. Default is 'linear'.
#    """
#    interpolator = interp1d(np.log10(initial_freq), np.log10(spectrum), kind=interpolation_method, fill_value='extrapolate')
#    resampled_spectrum = interpolator(np.log10(new_freq))
#    smoothed_resampled_spectrum = gaussian_filter1d(resampled_spectrum, gaussian_width)
#    return np.float32(log_to_lin(smoothed_resampled_spectrum))
#
#def wavelet(template):
#    from obspy.imaging.cm import obspy_sequential
#    from obspy.signal.tf_misfit import cwt
#    ns = len(template.stations)
#    nc = len(template.traces)/ns
#    fmin, fmax = autodet.cfg.freq_bands[0]
#    dt = template.traces[0].stats.delta
#    time = np.linspace(0., dt * float(template.traces[0].data.size), template.traces[0].data.size)
#    for s in range(ns):
#        for c in range(nc):
#            plt.subplot(ns, 2, s*nc+c+1)
#            scalogram = cwt(template.traces[s*nc+c].data, dt, 8, fmin, fmax)
#            if s == 0 and c == 0:
#                x,y = np.meshgrid(time, np.logspace(np.log10(fmin), np.log10(fmax), scalogram.shape[0]))
#            plt.pcolormesh(x,y,np.abs(scalogram), cmap=obspy_sequential)
#    plt.xlabel('Time')
#    plt.ylabel('Frequency')
#    plt.show()
#
#def kurtosis(data):
#    var = np.var(data)
#    if var != 0.:
#        return np.power((data - np.mean(data))/np.sqrt(var), 4.).sum()/data.size
#    else:
#        return 0.
#
#def sliding_kurto(data, w):
#    C = 1. - 1./(w * autodet.cfg.sampling_rate)
#    K = np.zeros(data.size, dtype=np.float32)
#    grad = np.zeros(data.size, dtype=np.float32)
#    S = np.sqrt(np.var(data))
#    m = 0.
#    s = np.float32(S)
#    for i in range(1,data.size):
#        m = C*m + (1.-C)*data[i]
#        s = C*s + (1.-C)*np.power(data[i] - m, 2)
#        if S != np.sqrt(np.var(data)):
#            print 'Prob'
#        if s > S:
#            K[i] = C*K[i-1] + (1.-C)*np.power(data[i] - m, 4)/np.power(s,2)
#        else:
#            K[i] = C*K[i-1] + (1.-C)*np.power(data[i] - m, 4)/np.power(S,2)
#    grad[:-1] = K[1:] - K[:-1]
#    return K, grad
#
#def azimuth(template, net):
#    from obspy.geodetics.base import calc_vincenty_inverse
#    azimuths = np.zeros(template.metadata['stations'].size, dtype=np.float32)
#    backazimuths = np.zeros(template.metadata['stations'].size, dtype=np.float32)
#    for s in range(template.metadata['stations'].size):
#        dist, az, baz = calc_vincenty_inverse(lat1=net.latitude[net.stations.index(template.metadata['stations'][s])],
#                                              lon1=net.longitude[net.stations.index(template.metadata['stations'][s])],
#                                              lat2=template.metadata['latitude'],
#                                              lon2=template.metadata['longitude'])
#        azimuths[s] = az
#        backazimuths[s] = baz
#    return azimuths, backazimuths
#
##def interactive_picking(click):
##    """
##    interactive_picking(click) \n
##    This function expects the user to first pick the phase and right after another sample defining
##    the uncertainty interval, assuming a symmetric error around the picked time.
##    """
##    ns = click.canvas.figure.get_axes()[0].numRows
##    #ns = 12
##    #print click.inaxes.rowNum
##    global picksP, picksS
##    if len(picksP) < 2*ns:
##        picksP.append(click.xdata)
##        orderP.append(click.inaxes.rowNum) 
##        click.inaxes.patch.set_facecolor('grey')
##        click.canvas.draw()
##    elif len(picksS) < 2*ns:
##        picksS.append(click.xdata)
##        orderS.append(click.inaxes.rowNum)
##        click.inaxes.patch.set_facecolor('grey')
##        click.canvas.draw()
##    if len(picksS) == 2*ns:
##        #click.canvas.disconnect()
##        plt.close()
#
#def interactive_picking(click):
#    """
#    interactive_picking(click) \n
#    This function expects the user to first pick the phase and right after another sample defining
#    the uncertainty interval, assuming a symmetric error around the picked time.
#    """
#    ns = click.canvas.figure.get_axes()[0].numRows
#    #ns = 12
#    #print click.inaxes.rowNum
#    global picks
#    picks.append(click.xdata)
#    if len(picks) == 2:
#        plt.close()
#
#def select_stations(click):
#    global selection
#    selection.append(click.inaxes.rowNum)
#    click.inaxes.patch.set_facecolor('grey')
#    click.canvas.blit(click.inaxes.bbox)
#    #click.canvas.draw()
#
#def templates_list(tp_file, to_rm=None):
#    path_tp_file = autodet.cfg.tpfiles_path + tp_file
#    IDs_rm = []
#    if to_rm is not None:
#        path_tp_to_rm = autodet.cfg.tpfiles_path + to_rm
#        with open(path_tp_to_rm, 'r') as f:
#            line = f.readline()[:-1]
#            while line != '':
#                IDs_rm.append(int(line.strip()[len('template'):]))
#                line = f.readline()[:-1]
#    tp_IDs = []
#    tp_names = []
#    with open(path_tp_file, 'r') as f:
#        line = f.readline()[:-1]
#        while line != '':
#            tid = int(line.strip()[len('template'):])
#            if tid in IDs_rm:
#                line = f.readline()[:-1]
#                continue
#            tp_names.append(line.strip())
#            tp_IDs.append(tid)
#            line = f.readline()[:-1]
#    return tp_IDs, tp_names
#
#def read_RE_IDs(tp_ID, version, STEP, format_array=False, type_thrs='MAD', study=autodet.cfg.substudy):
#    import pickle
#    filename = autodet.cfg.RE_IDs_path + study + '/%s/STEP%i/%s/tp%i_IDs.pickle' %(version, STEP, type_thrs, tp_ID)
#    with open(filename, 'rb') as f:
#        OT = pickle.load(f)
#    if format_array:
#        ot = []
#        filenames = []
#        indices = []
#        for n in range(len(OT)):
#            ot.append(OT[n][0])
#            filenames.append(OT[n][1])
#            indices.append(OT[n][2])
#        OT = [np.float64(ot), np.asarray(filenames), np.int32(indices)]
#    return OT
#
#def search_mag(time, lat, lon, dep, catalog):
#    if not isinstance(catalog, list):
#        catalog = [catalog]
#    T = udt(time).timestamp
#    M = None
#    for i in range(len(catalog)):
#        with open(catalog[i], 'r') as f:
#            line = f.readline()[:-1]
#            while line != '':
#                Tcat = udt(line).timestamp
#                if abs(T-Tcat) > 60.:
#                    for i in range(7):
#                        line = f.readline()[:-1]
#                    continue
#                line = f.readline()[:-1]
#                mag = float(line)
#                line = f.readline()[:-1]
#                mag_type = line
#                line = f.readline()[:-1]
#                lat_cat = float(line)
#                line = f.readline()[:-1]
#                lon_cat = float(line)
#                line = f.readline()[:-1]
#                dep_cat = float(line)
#                line = f.readline()[:-1]
#                region = line
#                if abs(lat_cat - lat) < 0.3 and abs(lon_cat - lon) < 0.3 and abs(dep_cat - dep) < 20.:
#                    M = mag
#                    return M, lat_cat, lon_cat, dep_cat, Tcat, region
#                else:
#                    line = f.readline()[:-1]
#    return M, 0, 0, 0, 0, 0
#
#def write_tempdistrib(tp_file, dt=60., version='V1', STEP=7, type_thrs='RMS', db_path=autodet.cfg.dbpath):
#    tp_IDs, tp_names = templates_list(tp_file)
#    tpdistrib_path = autodet.cfg.tpdistrib_path + '%s/STEP%i/%s/' %(version, STEP, type_thrs)
#    for tid in tp_IDs:
#        OT, C = temporal_distribution(tid, dt, db_path_M='MULTIPLETS_%s_Z%i_' %(version, STEP), type_thrs=type_thrs_thrs, db_path=db_path)
#        with open(tpdistrib_path + 'C_dt%.0f_tp%i.dat' %(dt, tid), 'wb') as fC:
#            C.tofile(fC)
#        with open(tpdistrib_path + 'OT_tp%i.dat' %tid, 'wb') as fOT:
#            OT.tofile(fOT)
