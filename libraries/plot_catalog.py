import numpy as np
import matplotlib.pyplot as plt
import sys
sys.path.append('/nobackup1/ebeauce/automatic_detection/')
import automatic_detection as autodet
import h5py
from obspy import UTCDateTime as udt
from os.path import isfile
from scipy.signal import hilbert, tukey
from matplotlib.colors import Normalize, LogNorm
from matplotlib.cm import ScalarMappable
from matplotlib import ticker
import matplotlib.colorbar as clb
from mpl_toolkits.axes_grid1 import make_axes_locatable

def plot_freq_1TP(tid, db_path_T='template_db_2/', db_path_M='matched_filter_2/', plotCC=False, plotMAG=True, db_path=autodet.cfg.dbpath):
    #------------------------------------------------------------
    font = {'family' : 'serif', 'weight' : 'bold', 'size' : 24}
    plt.rc('font', **font)
    plt.rcParams['pdf.fonttype'] = 42 #TrueType
    #------------------------------------------------------------
    catalog = autodet.db_h5py.read_catalog_multiplets('multiplets{:d}'.format(tid), db_path=db_path, db_path_M=db_path_M) 
    unique_events = catalog['unique_events']
    #magnitudes = catalog['magnitudes'][unique_events]
    magnitudes = np.zeros(unique_events.sum(), dtype=np.float32)
    cm = plt.get_cmap('RdBu_r')
    vmin = np.percentile(magnitudes, 5.)
    vmax = np.percentile(magnitudes, 99.5)
    cNorm = Normalize(vmin = vmin, vmax = vmax)
    scalarMap = ScalarMappable(norm=cNorm, cmap=cm)
    colors = scalarMap.to_rgba(magnitudes[1:])
    corr = catalog['correlation_coefficients']
    OT   = catalog['origin_times'][unique_events]
    corr /= np.median(corr)
    if plotCC:
        # make color scale
        cm = plt.get_cmap('brg')
        cNorm = LogNorm(vmin=corr.min(),vmax=2.)
        scalarMap = ScalarMappable(norm=cNorm,cmap=cm)
        colors = scalarMap.to_rgba(corr)
    #else:
    #    colors = ['k']*len(corr)
    start_day = udt('2012,08,01')
    end_day = udt('2013,08,01')
    #-------------------------------------------------------
    plt.figure('frequencies_tp{:d}'.format(tid))
    plt.gca().set_rasterization_zorder(1)
    #plt.title('Temporal distribution of the template %i\'s multiplets (%i detections)' %(tid, OT.size))
    t = OT - start_day.timestamp
    WT = (OT[1:] - OT[:-1])/(3600.*24.) # waiting times
    I = np.argsort(magnitudes[1:])
    sc=plt.scatter(t[1:][I], WT[I], 50, marker='o', color=colors[I], edgecolor='k', zorder=0)
    time_scale = [start_day]
    for i in range(12):
        T = udt(time_scale[-1])
        try:
            T.month += 1
        except ValueError:
            # month has to be between 1 in 12
            T.year += 1
            T.month = 1
        time_scale.append(T)
    tlabels = [T.strftime('%b%Y') for T in time_scale]
    idx = [T.timestamp - start_day.timestamp for T in time_scale]
    plt.xticks(idx, tlabels, rotation=45)
    #D1 = udt('2012,09,25')
    #D2 = udt('2012,11,02')
    #plt.xlim( (D1.timestamp - start_day.timestamp, D2.timestamp - start_day.timestamp) )
    plt.xlim((0., end_day.timestamp - start_day.timestamp))
    plt.semilogy()
    plt.ylim((10**np.floor(np.log10(WT.min())), 10**(np.ceil(np.log10(WT.max())))))
    plt.grid(axis='x')
    plt.ylabel('Recurrence interval (days)')
    if plotCC:
        ax, _ = clb.make_axes(plt.gca(), shrink=0.5)
        cbar = clb.ColorbarBase(ax, cmap = cm, norm=cNorm, label='CC') 
    elif plotMAG:
        cbmin = magnitudes.min()
        cbmax = magnitudes.max()
        ax, _ = clb.make_axes(plt.gca(), shrink=0.5, orientation='vertical', pad=0.15, aspect=40, anchor=(1.1,0.75))
        ticks_pos = np.arange(np.round(cbmin, decimals=1), np.round(cbmax, decimals=1), 0.5)
        ticks_pos = np.hstack( (ticks_pos[0]-0.0001, ticks_pos, ticks_pos[-1]+0.0001) )
        cbar = clb.ColorbarBase(ax, cmap = cm, norm=cNorm, \
                                label='Magnitude', orientation='vertical', 
                                boundaries=np.linspace(cbmin, cbmax, 100),\
                                ticks=ticks_pos)
    plt.subplots_adjust(bottom=0.4, top=0.8)
    plt.show()

def plot_freq_1TP_V2(tid, db_path_T='template_db_2/', db_path_M='matched_filter_2/', db_path=autodet.cfg.dbpath):
    #------------------------------------------------------------
    font = {'family' : 'serif', 'weight' : 'bold', 'size' : 24}
    plt.rc('font', **font)
    plt.rcParams['pdf.fonttype'] = 42 #TrueType
    #------------------------------------------------------------
    catalog = autodet.db_h5py.read_catalog_multiplets('multiplets{:d}'.format(tid), db_path=db_path, db_path_M=db_path_M) 
    unique_events = catalog['unique_events']
    #unique_events = np.ones(catalog['origin_times'].size, dtype=np.bool)
    OT = catalog['origin_times'][unique_events]
    s = catalog['slopes'][unique_events]
    #magnitudes = catalog['magnitudes'][unique_events]
    #-------------------------------------------------------
    start_day = udt('2012,08,01')
    end_day   = udt('2013,08,01')
    time_scale = [start_day]
    for i in range(12):
        T = udt(time_scale[-1])
        try:
            T.month += 1
        except ValueError:
            # month has to be between 1 in 12
            T.year += 1
            T.month = 1
        time_scale.append(T)
    tlabels = [T.strftime('%b%Y') for T in time_scale]
    idx = [T.timestamp - start_day.timestamp for T in time_scale]
    #-------------------------------------------------------
    fig = plt.figure('frequencies_tp{:d}'.format(tid))
    #plt.subplot(2,1,1)
    plt.subplot2grid((5,1), (0,0))
    plt.gca().set_rasterization_zorder(1)
    # load slopes
    S_max = 0.
    S = []
    lv = 0.01
    s[np.where(s != s)[0]] = 0. # put NaN to 0
    s[np.where(s <= lv)[0]] = lv
    # make color scale
    cm = plt.get_cmap('hot')
    cNorm = Normalize(vmin=0., vmax=1.0)
    scalarMap = ScalarMappable(norm=cNorm, cmap=cm)
    colors = scalarMap.to_rgba(s)
    alphas = s/s.max()
    #alphas[np.where(alphas < 1./5.)] = 1./5.
    alphas = np.ones(colors.shape[0])
    colors[:,-1] = alphas
    I = np.argsort(colors[:,-1])

    plt.scatter(OT - start_day.timestamp, np.ones(s.size), 40, marker='o', color=colors, zorder=0)
    plt.yticks([])
    plt.xticks(idx, ['']*len(idx))
    plt.xlim((0., end_day.timestamp - start_day.timestamp))
    plt.grid(axis='x')
    #------------------------------------------------------- 
    #ax, _ = clb.make_axes(plt.gca(), shrink=0.8, orientation='horizontal', pad=0.30, aspect=40, anchor=(0.3,2.))
    cbar_ax = fig.add_axes([0.3, 0.85, 0.4, 0.02])
    cbar = clb.ColorbarBase(cbar_ax, cmap = cm, norm=cNorm, label='Clustering coefficient', orientation='horizontal')
    cbar_ax.xaxis.set_ticks_position('top')
    #====================================================
    #====================================================
    cm = plt.get_cmap('RdBu_r')
    vmin = np.percentile(magnitudes, 5.)
    vmax = np.percentile(magnitudes, 99.5)
    cNorm = Normalize(vmin = vmin, vmax = vmax)
    scalarMap = ScalarMappable(norm=cNorm, cmap=cm)
    colors = scalarMap.to_rgba(magnitudes[1:])
    #plt.subplot(2,1,2)
    plt.subplot2grid((5,1), (1,0), rowspan=4)
    plt.gca().set_rasterization_zorder(1)
    t = OT - start_day.timestamp
    WT = (OT[1:] - OT[:-1])/(3600.*24.) # waiting times
    I = np.argsort(magnitudes[1:])
    sc = plt.scatter(t[1:][I], WT[I], 50, marker='o', color=colors[I], edgecolor='k', zorder=0)
    plt.xticks(idx, tlabels, rotation=45)
    plt.xlim((0., end_day.timestamp - start_day.timestamp))
    plt.semilogy()
    plt.ylim((10**np.floor(np.log10(WT.min())), 10**(np.ceil(np.log10(WT.max())))))
    plt.grid(axis='x')
    plt.ylabel('Recurrence interval (days)')
    #-------------------------------------------------------
    cbmin = np.floor(10. * catalog['magnitudes'].min()) / 10.
    cbmax = np.ceil(10. * catalog['magnitudes'].max()) / 10.
    #ax, _ = clb.make_axes(plt.gca(), shrink=0.5, orientation='vertical', pad=0.15, aspect=40, anchor=(1.1,0.75))
    #ticks_pos = np.round(np.arange(cbmin, cbmax, (cbmax - cbmin)/5.), decimals=1)
    ticks_pos = np.floor(10. * np.arange(cbmin, cbmax, (cbmax - cbmin)/5.)) / 10.
    #ticks_pos = np.hstack( (ticks_pos[0]-0.0001, ticks_pos, ticks_pos[-1]+0.0001) )
    #cbar = clb.ColorbarBase(ax, cmap = cm, norm=cNorm, \
    #                        label='Magnitude', orientation='vertical', 
    #                        boundaries=np.linspace(cbmin, cbmax, 100),\
    #                        ticks=ticks_pos)
    cbar_ax = fig.add_axes([0.91, 0.41, 0.01, 0.30])
    cbar = clb.ColorbarBase(cbar_ax, cmap = cm, norm=cNorm, \
                            label='Magnitude', orientation='vertical')
    plt.subplots_adjust(bottom=0.4, top=0.8, hspace=0.)
    plt.show()

def plot_recurrence_times(db_path_T='template_db_3/', db_path_M='matched_filter_2/', magnitudes=True, db_path=autodet.cfg.dbpath, path_template_list='template_db_3/', well_relocated_templates=False):
    #------------------------------------------------------------
    font = {'family' : 'serif', 'size' : 20}
    plt.rc('font', **font)
    plt.rcParams['pdf.fonttype'] = 42 #TrueType
    #------------------------------------------------------------
    with h5py.File(db_path + path_template_list + 'database_index.h5', mode='r') as f:
        if well_relocated_templates:
            tids = f['well_relocated_template_indexes'][()]
        else:
            tids = f['template_indexes'][()]
    # python list version
    OT = []
    WT = []
    M  = []
    Ntot = 0
    for tid in tids:
        catalog = autodet.db_h5py.read_catalog_multiplets('multiplets{:d}'.format(tid), db_path=db_path, db_path_M=db_path_M)
        mask = catalog['unique_events']
        #mask = np.ones(catalog['origin_times'].size, dtype=np.bool)
        origin_times = catalog['origin_times'][mask]
        Ntot += origin_times.size
        if origin_times.size <= 1:
            continue
        OT.extend(origin_times[1:])
        if magnitudes:
            mag_ = catalog['magnitudes'][mask][1:]
            if mag_.size == 0:
                continue
            if mag_.min() < -2.:
                mag_ = -10. * np.ones(mag_.size, dtype=np.float32)
        else:
            mag_ = -10. * np.ones(origin_times.size-1, dtype=np.float32)
        M.extend(mag_)
        wt = origin_times[1:] - origin_times[:-1]
        WT.extend(wt)
    # python list version: convert to numpy arrays
    OT = np.asarray(OT)
    WT = np.asarray(WT)
    M  = np.asarray(M)
    mask_mag = M == -10.
    WT /= (3600.*24.) # conversion to days
    #--------------------------------
    cm = plt.get_cmap('RdBu_r')
    if magnitudes:
        cNorm = Normalize(vmin = np.percentile(M[~mask_mag], 2.), vmax=np.percentile(M[~mask_mag], 99.5))
        #vmin = np.percentile(M[~mask_mag], 5.)
        #vmax = np.percentile(M[~mask_mag], 99.5)
        #cNorm = Normalize(vmin = vmin, vmax = vmax)
        scalarMap = ScalarMappable(norm=cNorm, cmap=cm)
        colors = scalarMap.to_rgba(M[~mask_mag])
    #--------------------------------
    start_day = udt('2012,08,01')
    end_day   = udt('2013,08,01')
    #-------------------------------------------------------
    fig = plt.figure('catalog_recurrence_times')
    ax  = plt.gca()
    ax.set_rasterization_zorder(1)
    plt.title('Temporal distribution of the detections ({:d} detections)'.format(Ntot))
    #------------------------------
    #-------- plot events without mag -----
    t = OT[mask_mag] - start_day.timestamp
    I = np.argsort(M[mask_mag])
    sc2 = plt.scatter(t[I], WT[mask_mag][I], 50, marker='v', color='k', zorder=0)
    #sc2.set_rasterized(True)
    #------------------------------
    #------- plot events with mag -------
    if magnitudes:
        t = OT[~mask_mag] - start_day.timestamp
        I = np.argsort(M[~mask_mag])
        sc=plt.scatter(t[I], WT[~mask_mag][I], 50, marker='o', color=colors[I], edgecolor='k')
    time_scale = [start_day]
    for i in range(12):
        T = udt(time_scale[-1])
        try:
            T.month += 1
        except ValueError:
            # month has to be between 1 in 12
            T.year += 1
            T.month = 1
        time_scale.append(T)
    tlabels = [T.strftime('%b%Y') for T in time_scale]
    idx = [T.timestamp - start_day.timestamp for T in time_scale]
    plt.xticks(idx, tlabels, rotation=45)
    #--------------------------
    #D1 = udt('2012,09,25')
    #D2 = udt('2012,11,02')
    #plt.xlim( (D1.timestamp - start_day.timestamp, D2.timestamp - start_day.timestamp) )
    plt.xlim((0., end_day.timestamp - start_day.timestamp))
    plt.semilogy()
    plt.ylim((10**np.floor(np.log10(WT.min())), 10**(np.ceil(np.log10(WT.max())))))
    plt.grid(axis='x')
    plt.gca().set_rasterized(True)
    plt.ylabel('Recurrence Times (days)')
    if magnitudes:
        # plot the colorbar
        cbar_ax  = fig.add_axes([0.91, 0.4, 0.02, 0.4])
        cbar_mag = clb.ColorbarBase(cbar_ax, cmap = cm, norm=cNorm, label='Magnitude', orientation='vertical')
    plt.subplots_adjust(bottom=0.4, top=0.8)
    plt.show()

def plot_averaged_clustering(db_path_T='template_db_3/', db_path_M='matched_filter_2/', well_relocated_templates=False):
    #-------------------------------------------------------
    font = {'family' : 'serif', 'weight' : 'normal', 'size' : 18}
    plt.rc('font', **font)
    #------------------------------------------------------------
    # read the database of template events
    tids     = autodet.db_h5py.get_template_ids_list('database_index', db_path = autodet.cfg.dbpath + db_path_T, well_relocated_templates=well_relocated_templates)
    tids_all = autodet.db_h5py.get_template_ids_list('database_index', db_path = autodet.cfg.dbpath + db_path_T, well_relocated_templates=False)
    #------------------------------------------------------------
    # load distances
    distances_all = np.loadtxt(autodet.cfg.dbpath + db_path_T + 'projection_templates.txt')
    if well_relocated_templates:
        distances = np.zeros(tids.size, dtype=np.float64)
        for t in range(len(tids)):
            distances[t] = distances_all[np.where(tids_all == tids[t])[0][0]]
    else:
        distances = distances_all
    # load clustering coefficients
    cc_max = 0.
    CC = []
    lv = 0.01
    TIMINGS = []
    for tid in tids:
        cc      = autodet.db_h5py.read_catalog_multiplets('multiplets{:d}'.format(tid), object_from_cat='sliding_clustering_coefficient_averaged', db_path=autodet.cfg.dbpath, db_path_M=db_path_M)
        timings = autodet.db_h5py.read_catalog_multiplets('multiplets{:d}'.format(tid), object_from_cat='timings_averaged_clustering', db_path=autodet.cfg.dbpath, db_path_M=db_path_M)
        print('Template {:d}: {:d} events'.format(tid, cc.size))
        if cc.size == 0:
            # can happen if the averaged values are plotted: templates can have 0 unique detections
            CC.append([])
            TIMINGS.append([])
            continue
        cc[np.where(cc != cc)[0]] = 0. # put NaN to 0
        cc[np.where(cc <= lv)[0]] = lv
        if cc.max() > cc_max:
            cc_max = cc.max()
        CC.append(cc)
        TIMINGS.append(timings)
    # make color scale
    cm = plt.get_cmap('hot')
    cNorm = Normalize(vmin=0., vmax=1.0)
    scalarMap = ScalarMappable(norm=cNorm, cmap=cm)
    # plot
    std = udt('2012,08,01').timestamp
    edd = udt('2013,08,01').timestamp
    Nbins = int((edd - std)/(3600.*24.))
    plt.figure('averaged_clustering_time_space')
    plt.gca().set_rasterization_zorder(1)
    Ntot = 0
    TIMES      = []
    DISTANCES  = []
    CLUSTERING = []
    COLORS     = np.array([[], [], [], []]).reshape(-1, 4)
    #CLUSTERING = np.zeros(0, dtype=np.float64)
    for i,template_ID in enumerate(tids):
        print('--------- {:d} / {:d} --------'.format(i, len(tids)))
        Ntot += len(TIMINGS[i])
        if len(TIMINGS[i]) == 0:
            continue
        T = TIMINGS[i] - std
        D = np.ones(T.size) * distances[i]
        colors = scalarMap.to_rgba(CC[i])
        alphas = CC[i]/cc_max
        alphas[np.where(alphas < 1./2.)] = 1./2.
        colors[:,-1] = alphas
        #------------------------------
        CLUSTERING.extend(CC[i])
        TIMES.extend(T)
        DISTANCES.extend(D)
        COLORS = np.vstack( (COLORS, colors) )
    CLUSTERING = np.asarray(CLUSTERING)
    TIMES      = np.asarray(TIMES)
    DISTANCES  = np.asarray(DISTANCES)
    #I = np.argsort(COLORS[:,-1])
    I = np.argsort(CLUSTERING)
    plt.scatter(TIMES[I], DISTANCES[I], 20, marker='o', color=COLORS[I,:], zorder=0)
    start_day = udt('2012,08,01')
    time_scale = [start_day]
    for i in range(12):
        T = udt(time_scale[-1])
        try:
            T.month += 1
        except ValueError:
            # month has to be between 1 in 12
            T.year += 1
            T.month = 1
        time_scale.append(T)
    tlabels = [T.strftime('%b%Y') for T in time_scale]
    idx = [T.timestamp - start_day.timestamp for T in time_scale]
    plt.xticks(idx, tlabels, rotation=45)
    plt.grid(axis='x')
    plt.xlim((0, udt('2013,08,02').timestamp - std))
    plt.ylabel('Position on CT05 - CT42 axis')
    plt.title('Temporal clustering')
    plt.subplots_adjust(bottom=0.30, left=0.1)
    #--------------------------------------
    net = autodet.dataset.Network('network.in')
    net.read()
    Labels = np.loadtxt(autodet.cfg.base + 'projection_labels.txt')
    Pos = Labels[:,1]
    Sta_names = []
    for idx in Labels[:,0]:
        Sta_names.append(net.stations[int(idx)])
    plt.yticks(Pos, Sta_names)
    ax, _ = clb.make_axes(plt.gca(), shrink=0.5, orientation='horizontal', pad=0.30, aspect=40)
    cbar = clb.ColorbarBase(ax, cmap = cm, norm=cNorm, label='Clustering coefficient', orientation='horizontal')
    plt.show()

def plot_clustering(db_path_T='template_db_2/', db_path_M='matched_filter_2/', well_relocated_templates=False):
    #-------------------------------------------------------
    font = {'family' : 'serif', 'weight' : 'normal', 'size' : 18}
    plt.rc('font', **font)
    #------------------------------------------------------------
    # read the database of template events
    tids     = autodet.db_h5py.get_template_ids_list('database_index', db_path = autodet.cfg.dbpath + db_path_T, well_relocated_templates=well_relocated_templates)
    tids_all = autodet.db_h5py.get_template_ids_list('database_index', db_path = autodet.cfg.dbpath + db_path_T, well_relocated_templates=False)
    #------------------------------------------------------------
    # load distances
    distances_all = np.loadtxt(autodet.cfg.dbpath + db_path_T + 'projection_templates.txt')
    if well_relocated_templates:
        distances = np.zeros(tids.size, dtype=np.float64)
        for t in range(len(tids)):
            distances[t] = distances_all[np.where(tids_all == tids[t])[0][0]]
    else:
        distances = distances_all
    # load slopes
    S_max = 0.
    S = []
    lv = 0.01
    for tid in tids:
        #s = autodet.db_h5py.read_catalog_multiplets('multiplets{:d}'.format(tid), object_from_cat='slopes', db_path=autodet.cfg.dbpath, db_path_M=db_path_M)
        s = autodet.db_h5py.read_catalog_multiplets('multiplets{:d}'.format(tid), object_from_cat='sliding_clustering_coefficient_averaged', db_path=autodet.cfg.dbpath, db_path_M=db_path_M)
        print('Template {:d}: {:d} events'.format(tid, s.size))
        if s.size == 0:
            # can happen if the averaged values are plotted: templates can have 0 unique detections
            S.append([])
            continue
        s[np.where(s != s)[0]] = 0. # put NaN to 0
        s[np.where(s <= lv)[0]] = lv
        if s.max() > S_max:
            S_max = s.max()
        S.append(s)
    # make color scale
    cm = plt.get_cmap('hot')
    cNorm = Normalize(vmin=0., vmax=1.0)
    scalarMap = ScalarMappable(norm=cNorm, cmap=cm)
    # plot
    std = udt('2012,08,01').timestamp
    edd = udt('2013,08,01').timestamp
    Nbins = int((edd - std)/(3600.*24.))
    plt.figure('clustering_time_space')
    plt.gca().set_rasterization_zorder(1)
    Ntot = 0
    TIMES      = []
    DISTANCES  = []
    CLUSTERING = []
    COLORS     = np.array([[], [], [], []]).reshape(-1, 4)
    #CLUSTERING = np.zeros(0, dtype=np.float64)
    for i,template_ID in enumerate(tids):
        print('--------- {:d} / {:d} --------'.format(i, len(tids)))
        OT = autodet.db_h5py.read_catalog_multiplets('multiplets{:d}'.format(template_ID), object_from_cat='origin_times', db_path=autodet.cfg.dbpath, db_path_M=db_path_M)
        #if OT.size == 0:
        #    continue
        unique_events = autodet.db_h5py.read_catalog_multiplets('multiplets{:d}'.format(template_ID), object_from_cat='unique_events', db_path=autodet.cfg.dbpath, db_path_M=db_path_M)
        #C, times = np.histogram(OT, bins=Nbins, range=(std, edd))
        Ntot += OT[unique_events].size
        T = np.asarray(OT[unique_events]) - std
        D = np.ones(T.size) * distances[i]
        #t_ids = np.int32(T/(3600.*24.))
        colors = scalarMap.to_rgba(S[i][unique_events])
        alphas = S[i][unique_events]/S_max
        alphas[np.where(alphas < 1./2.)] = 1./2.
        colors[:,-1] = alphas
        #CLUSTERING = np.hstack( (CLUSTERING, S[i][unique_events]) )
        #TIMES = np.hstack( (TIMES, T) )
        #DISTANCES = np.hstack( (DISTANCES, D) )
        CLUSTERING.extend(S[i][unique_events])
        TIMES.extend(T)
        DISTANCES.extend(D)
        COLORS = np.vstack( (COLORS, colors) )
    CLUSTERING = np.asarray(CLUSTERING)
    TIMES      = np.asarray(TIMES)
    DISTANCES  = np.asarray(DISTANCES)
    #I = np.argsort(COLORS[:,-1])
    I = np.argsort(CLUSTERING)
    plt.scatter(TIMES[I], DISTANCES[I], 20, marker='o', color=COLORS[I,:], zorder=0)
    print("Total number of detections: {:d}".format(Ntot))
    start_day = udt('2012,08,01')
    time_scale = [start_day]
    for i in range(12):
        T = udt(time_scale[-1])
        try:
            T.month += 1
        except ValueError:
            # month has to be between 1 in 12
            T.year += 1
            T.month = 1
        time_scale.append(T)
    tlabels = [T.strftime('%b%Y') for T in time_scale]
    idx = [T.timestamp - start_day.timestamp for T in time_scale]
    plt.xticks(idx, tlabels, rotation=45)
    plt.grid(axis='x')
    #D1 = udt('2012,01,31')
    #D2 = udt('2013,03,01')
    #plt.xlim( (D1.timestamp - std, D2.timestamp - std) )
    plt.xlim((0, udt('2013,08,02').timestamp - std))
    plt.ylabel('Position on CT10 - CT45 axis')
    #plt.title('Total of %i detections' %Ntot)
    plt.title('Temporal clustering')
    plt.subplots_adjust(bottom=0.30, left=0.1)
    #--------------------------------------
    net = autodet.dataset.Network('network.in')
    net.read()
    Labels = np.loadtxt(autodet.cfg.base + 'projection_labels.txt')
    Pos = Labels[:,1]
    Sta_names = []
    for idx in Labels[:,0]:
        Sta_names.append(net.stations[int(idx)])
    plt.yticks(Pos, Sta_names)
    ax, _ = clb.make_axes(plt.gca(), shrink=0.5, orientation='horizontal', pad=0.30, aspect=40)
    cbar = clb.ColorbarBase(ax, cmap = cm, norm=cNorm, label='Clustering coefficient', orientation='horizontal')
    plt.show()

def plot_map_clustering(db_path_T='template_db_2/', db_path_M='matched_filter_2/', well_relocated_templates=False, smoothing_param=None):
    #-------------------------------------------------------
    font = {'family' : 'serif', 'weight' : 'normal', 'size' : 18}
    plt.rc('font', **font)
    plt.rcParams['pdf.fonttype'] = 42 #TrueType
    plt.rcParams['svg.fonttype'] = 'none'
    #------------------------------------------------------------
    # read the database of template events
    tids     = autodet.db_h5py.get_template_ids_list('database_index', db_path = autodet.cfg.dbpath + db_path_T, well_relocated_templates=well_relocated_templates)
    tids_all = autodet.db_h5py.get_template_ids_list('database_index', db_path = autodet.cfg.dbpath + db_path_T, well_relocated_templates=False)
    #------------------------------------------------------------
    # load distances
    distances_all = np.loadtxt(autodet.cfg.dbpath + db_path_T + 'projection_templates.txt')
    if well_relocated_templates:
        distances = np.zeros(tids.size, dtype=np.float64)
        for t in range(len(tids)):
            distances[t] = distances_all[np.where(tids_all == tids[t])[0][0]]
    else:
        distances = distances_all
    # make color scale
    cm = plt.get_cmap('hot')
    cm_detections = plt.get_cmap('viridis')
    #cNorm = Normalize(vmin=0., vmax=1.0)
    # plot
    std = udt('2012,08,01').timestamp
    edd = udt('2013,08,01').timestamp
    Nbins = int((edd - std)/(3600.*24.))
    TIMES      = []
    DISTANCES  = []
    CLUSTERING = []
    ECNs       = []
    #------------------------------
    W_day         = 10  # size of the sliding window, in days
    W_sec  = W_day *3600.*24.
    n_windows = int( (edd - std) / W_sec)
    for i, tid in enumerate(tids):
        print('--------- {:d} / {:d} --------'.format(i+1, len(tids)))
        CCs = autodet.db_h5py.read_catalog_multiplets('multiplets{:d}'.format(tid), object_from_cat='non_overlapping_CC', db_path=autodet.cfg.dbpath, db_path_M=db_path_M)
        ECNs.extend(autodet.db_h5py.read_catalog_multiplets('multiplets{:d}'.format(tid), object_from_cat='n_events_per_window', db_path=autodet.cfg.dbpath, db_path_M=db_path_M))
        T = np.linspace(std, edd, CCs.size) - std
        D = np.ones(T.size) * distances[i]
        CLUSTERING.extend(CCs)
        TIMES.extend(T)
        DISTANCES.extend(D)
    CLUSTERING = np.asarray(CLUSTERING).reshape(len(tids), -1)
    CLUSTERING[CLUSTERING < 0.] = 0.
    DISTANCES  = np.asarray(DISTANCES).reshape(len(tids), -1)
    TIMES      = np.asarray(TIMES).reshape(len(tids), -1)
    ECNs       = np.asarray(ECNs).reshape(len(tids), -1)
    I = np.argsort(DISTANCES[:,0])
    #-----------------------------------
    #         reorder the matrices
    TIMES      = TIMES[I, :] # just for consistency, but actually does nothing
    DISTANCES  = DISTANCES[I, :]
    CLUSTERING = CLUSTERING[I, :]
    ECNs       = ECNs[I, :]
    #-----------------------------------
    if smoothing_param is not None:
        #from scipy.ndimage.filters import gaussian_filter
        #CLUSTERING = gaussian_filter(CLUSTERING, sigma=smoothing_param)
        Kernel = np.zeros((len(tids), len(tids)), dtype=np.float32)
        sigma  = smoothing_param * np.std(distances)
        for t1 in range(len(tids)):
            for t2 in range(t1, len(tids)):
                Kernel[t1, t2] = np.exp(-(distances[t2] - distances[t1])**2/(2.*sigma))
        Kernel = (Kernel + Kernel.T) / 2.
        for t in range(len(tids)):
            norm = Kernel[t,:].sum()
            if norm != 0.:
                Kernel[t,:] /= norm
        for i in range(TIMES.shape[0]):
            for j in range(TIMES.shape[1]):
                #CLUSTERING[i, j] = np.sum(CLUSTERING[:, j] * Kernel[i, :])
                ECNs[i, j] = np.sum(ECNs[:, j] * Kernel[i, :])
    #-----------------------------------
    start_day = udt('2012,08,01')
    time_scale = [start_day]
    for i in range(12):
        T = udt(time_scale[-1])
        try:
            T.month += 1
        except ValueError:
            # month has to be between 1 in 12
            T.year += 1
            T.month = 1
        time_scale.append(T)
    tlabels = [T.strftime('%b%Y') for T in time_scale]
    idx_time_ticks = [T.timestamp - start_day.timestamp for T in time_scale]
    net = autodet.dataset.Network('network.in')
    net.read()
    Labels = np.loadtxt(autodet.cfg.base + 'projection_labels.txt')
    Pos = Labels[:,1]
    Sta_names = []
    for idx in Labels[:,0]:
        Sta_names.append(net.stations[int(idx)])
    #-----------------------------------
    plt.figure('map_clustering')
    ax1 = plt.subplot(2, 1, 1)
    ax1.set_rasterization_zorder(1)
    ECNs += 1
    cNorm_detections = LogNorm(vmin=ECNs.min(), vmax=ECNs.max())
    #plt.contourf(TIMES, DISTANCES, ECNs, cmap=cm_detections, levels=500, zorder=0, locator=ticker.LogLocator())
    im_ecn = plt.contourf(TIMES, DISTANCES, ECNs, cmap=cm_detections, locator=ticker.LogLocator(subs=(1.0, 3.0, 7.0,)), zorder=0)
    #im_ecn = plt.pcolor(TIMES, DISTANCES, ECNs, cmap=cm_detections, norm=cNorm_detections, zorder=0)
    plt.xticks(idx_time_ticks, ['']*len(idx_time_ticks))
    plt.grid(axis='x')
    plt.ylabel('Projected Position')
    plt.yticks(Pos, Sta_names)
    plt.ylim(Pos[Sta_names.index('CT10')], Pos[Sta_names.index('CT45')])
    plt.xlim(0., edd-std)
    #plt.colorbar(orientation='horizontal', label='Number of Earthquakes', aspect=40)
    divider = make_axes_locatable(ax1)
    cax     = divider.append_axes('top', size='5%', pad=1.35)
    plt.colorbar(im_ecn, cax=cax, orientation='horizontal', label='Number of Earthquakes')
    cax.xaxis.set_ticks_position('top')
    cax.xaxis.set_label_position('top')
    #------------------------------------------------
    ax2 = plt.subplot(2, 1, 2)
    ax2.set_rasterization_zorder(1)
    im_cc = plt.contourf(TIMES, DISTANCES, CLUSTERING, cmap=cm, levels=50, zorder=0)
    #plt.pcolormesh(TIMES, DISTANCES[I,:], CLUSTERING[I,:], cmap=cm, norm=cNorm, zorder=0)
    #plt.colorbar(orientation='horizontal', pad=0.35, label=r'Clustering Coefficient $\beta$', aspect=40)
    plt.xticks(idx_time_ticks, tlabels, rotation=90)
    plt.grid(axis='x')
    plt.ylabel('Projected Position')
    #--------------------------------------
    plt.yticks(Pos, Sta_names)
    plt.ylim(Pos[Sta_names.index('CT10')], Pos[Sta_names.index('CT45')])
    plt.xlim(0., edd-std)
    #plt.subplots_adjust(bottom=0.47, left=0.1)
    divider = make_axes_locatable(ax2)
    cax     = divider.append_axes('bottom', size='5%', pad=1.35)
    plt.colorbar(im_cc, cax=cax, orientation='horizontal', label=r'Clustering Coefficient $\beta$')
    plt.show()

#def plot_magnitude(tid, version, STEP, type_thrs='MAD', db_path=autodet.cfg.dbpath, show=True):
#    """
#    plot_magnitude(tid, version, STEP, type_thrs='MAD', db_path=autodet.cfg.dbpath) \n
#    """
#    font = {'family' : 'serif', 'weight' : 'normal', 'size' : 18}
#    plt.rc('font', **font)
#    # -----------------------------------------------------
#    fname = 'multiplets%i' %tid
#    db_path_M = 'MULTIPLETS_%s_Z%i_%s/' %(version, STEP, type_thrs)
#    catalog = autodet.db_h5py.read_catalog_multiplets(fname, db_path=db_path, db_path_M=db_path_M)
#    Ml = catalog['magnitudes']
#    if Ml[0] == -10:
#        print "No magnitude estimations available for Template %i !" %tid
#        return -10, -10
#    elif Ml.size < 30:
#        print "Not enough detections to make statistics for Template %i !" %tid
#        return -10, -10
#    if Ml.size < 100:
#        Q = clean_catalog(catalog['origin_times'], catalog['travel_times_S'])
#    else:
#        Q = np.ones(Ml.size, dtype=np.bool)
#    if Q.sum() < 30:
#        print "Not enough good quality detections to make statistics for Template %i" %tid
#        return -10, -10
#    fig = plt.figure()
#    plt.suptitle('Frequency-Magnitude distribution of Template %i\'s detections' %tid)
#    plt.subplot(1,2,1)
#    n, bins, _ = plt.hist(Ml[Q], bins=100)
#    plt.xlabel('Magnitude (Ml)')
#    plt.ylabel('Number of earthquakes')
#    N = np.zeros(n.size, dtype=np.float32)
#    for i in xrange(n.size):
#        N[i] = n[i+1:].sum()
#    plt.subplot(1,2,2)
#    idx = np.hstack((True, np.abs(N[1:] - N[:-1]) != 0))
#    if N[idx][-1] == 0:
#        Nlog = np.log10(N[idx][:-1])
#        M = bins[1:][idx][:-1]
#    else:
#        Nlog = np.log10(N[idx])
#        M = bins[1:][idx]
#    idx_robust = (M > 0.5) & (M < 2.5)
#    slopes, inter = WFit(M, Nlog, 0.5)
#    if np.median(-slopes[idx_robust]) < 0.6:
#        # Try something more robust, ie regression using all the points
#        slopes, inter = WFit(M, Nlog, 100.)
#    #-----------------------------------
#    sl_min = np.min(slopes[idx_robust])
#    it_min = inter[idx_robust][slopes[idx_robust].argmin()]
#    sl_max = np.max(slopes[idx_robust])
#    it_max = inter[idx_robust][slopes[idx_robust].argmax()]
#    sl = (sl_min + sl_max)/2.
#    dsl = (sl_max - sl_min)/2.
#    it = (it_min + it_max)/2.
#    #-----------------------------------
#    plt.plot(M, Nlog, ls='', marker='o')
#    plt.xlabel('Magnitude (Ml)')
#    plt.ylabel('Cumulative number of earthquakes log(N)')
#    plt.plot(M, it + sl*M, color='r', lw=2, label='Gutenberg-Richter law\n'+r'b value = %.2f$\pm$%.2f' %(-sl, dsl))
#    plt.plot(M, it_min + sl_min*M, color='r', lw=2, ls='--')
#    plt.plot(M, it_max + sl_max*M, color='r', lw=2, ls='--')
#    plt.legend(loc='upper right', fancybox=True)
#    if show:
#        plt.show()
#    else:
#        plt.close()
#    return -sl, it
#
#def plot_magnitude_all(version, STEP, type_thrs='RMS', db_path=autodet.cfg.dbpath, study=autodet.cfg.substudy, show=True):
#    """
#    plot_magnitude(tid, version, STEP, type_thrs='MAD', db_path=autodet.cfg.dbpath) \n
#    """
#    import linear_regression as lin_reg
#    font = {'family' : 'serif', 'weight' : 'normal', 'size' : 18}
#    plt.rc('font', **font)
#    # -----------------------------------------------------
#    tpl_list = 'templates%s_%s_STEP%i.txt' %(study, version, STEP)
#    tp_IDs, tp_names = templates_list(tpl_list)
#    M = []
#    for tid in tp_IDs:
#        fname = 'multiplets%i' %tid
#        db_path_M = 'MULTIPLETS_%s_STEP%i_%s/' %(version, STEP, type_thrs)
#        catalog = autodet.db_h5py.read_catalog_multiplets(fname, db_path=db_path, db_path_M=db_path_M)
#        Ml = catalog['magnitudes'][catalog['unique_events']]
#        if np.allclose(Ml, -10.*np.ones(Ml.size), atol=0.1):
#            print "No magnitude estimations available for Template %i !" %tid
#            continue
#        else:
#            M = np.hstack((M, Ml))
#    print M.size
#    Mmax = 10.
#    M0 = 1.
#    mask = (M > M0) & (M < Mmax)
#    #b_ = 1./(np.sum(M[mask])/np.float32(np.sum(mask)) - M0)
#    b_ = 1./(np.mean(M[mask]) - M0)
#    bb = b_ * np.log10(np.exp(1.))
#    fig = plt.figure()
#    plt.subplot(1,2,1)
#    n, bins, _ = plt.hist(M[M > -1.], bins=100)
#    M0_arg = np.argwhere(bins >= M0)[0]
#    plt.xlabel('Magnitude (Ml)')
#    plt.ylabel('Number of earthquakes')
#    N = np.cumsum(n[::-1])[::-1]
#    #N = np.zeros(n.size, dtype=np.float32)
#    #for i in xrange(n.size):
#    #    N[i] = n[i+1:].sum()
#    plt.subplot(1,2,2)
#    idx = np.hstack((True, np.abs(N[1:] - N[:-1]) != 0))
#    if N[idx][-1] == 0:
#        Nlog = np.log10(N[idx][:-1])
#        M = bins[1:][idx][:-1]
#    else:
#        Nlog = np.log10(N[idx])
#        M = bins[1:][idx]
#    idx_robust = M > 0.
#    #slopes, inter = WFit(M, Nlog, 1.0)
#    Mbins = (bins[:-1] + bins[1:]) / 2.
#    Mbins = Mbins[idx]
#    print Mbins.size, Nlog.size
#    mask2 = (Mbins > M0) & (Mbins < Mmax)
#    linreg_output = lin_reg.NLSQ_least_squares(Mbins[mask2], Nlog[mask2], np.ones(Mbins[mask2].size), loss='linear')
#    #-----------------------------------
#    plt.plot(M, Nlog, ls='', marker='o')
#    plt.plot(M, Nlog[M0_arg] - bb* M, lw=2, color='C3', label='Maximum Likelihood Estimate: b={:.2f}'.format(bb))
#    plt.plot(Mbins, linreg_output.x[0] + linreg_output.x[1] * Mbins, lw=2, color='C4', label='Linear regression: b={:.2f}'.format(abs(linreg_output.x[1])))
#    plt.axvline(M0, ls='--', color='k', lw=2)
#    plt.xlabel('Magnitude (Ml)')
#    plt.ylabel('Cumulative number of earthquakes log(N)')
#    plt.legend(loc='upper right', fancybox=True)
#    if show:
#        plt.show()
#    else:
#        plt.close()
#    return linreg_output
#
#def WFit(X, Y, Lcor):
#    """
#    WFit(X, Y, Lcor) \n
#    Distance-weighted least square regression, with analytical solution.
#    """
#    slopes = np.zeros(X.size, dtype=np.float32)
#    intercepts = np.zeros(X.size, dtype=np.float32)
#    for i in xrange(X.size):
#        # decreasing exponential
#        w2 = np.exp(-np.power(X-X[i], 2)/(2.*Lcor**2))
#        # sum of the squared distances
#        W2 = w2.sum()
#        xmean = np.sum(w2*X)/W2
#        ymean = np.sum(w2*Y)/W2
#        xvar = np.sum(w2*np.power(X-xmean, 2))
#        xycov = np.sum(w2*(X-xmean)*(Y-ymean))
#        best_slope = xycov/xvar
#        best_inter = ymean - best_slope*xmean
#        slopes[i] = best_slope
#        intercepts[i] = best_inter
#    return slopes, intercepts
#
#def plot_cumul(tid, DT, version='V1', STEP=5, type_thrs='MAD', db_path=autodet.cfg.dbpath):
#    font = {'family' : 'serif', 'weight' : 'bold', 'size' : 24}
#    plt.rc('font', **font)
#    plt.rcParams['pdf.fonttype'] = 42 #TrueType
#    #------------------------------------------------------------
#    db_path_M = 'MULTIPLETS_%s_Z%i_%s/' %(version, STEP, type_thrs)
#    catalog = autodet.db_h5py.read_catalog_multiplets('multiplets%i' %tid, db_path=db_path, db_path_M=db_path_M)
#    OT = catalog['origin_times']
#    start_day = udt('2012,08,01')
#    std = start_day.timestamp
#    end_day = udt('2013,08,01')
#    Nbins = (end_day.timestamp - start_day.timestamp)/DT
#    C, times = np.histogram(catalog['origin_times'], bins=Nbins, range=(start_day.timestamp, end_day.timestamp))
#    C_cumul = np.zeros(C.size, dtype=np.float32)
#    C_cumul[0] = C[0]
#    for i in xrange(1, C.size):
#        C_cumul[i] = C_cumul[i-1] + C[i]
#    C_cumul /= C.sum()
#    #time = np.arange(C.size)
#    time = times[:-1] - std
#    plt.figure('cumulative_event_count_TP%i' %tid)
#    plt.title('Cumulative event count (template %i\'s detections)' %tid)
#    plt.plot(time, C_cumul)
#    time_scale = [start_day]
#    idx = [0]
#    n = 0
#    for t in np.arange(time.size):
#        T = udt(time_scale[-1] + n*DT)
#        if T.month == time_scale[-1].month:
#            n += 1
#            continue
#        else:
#            idx.append(T.timestamp - std)
#            time_scale.append(T)
#            n = 0
#    tlabels = [T.strftime('%b%Y') for T in time_scale]
#    plt.xticks(idx, tlabels, rotation=45)
#    plt.xlim((0., end_day.timestamp-std))
#    plt.grid(axis='x')
#    plt.subplots_adjust(bottom=0.13)
#    plt.show()
#
#def STcorr(tid, DT, day, version='V1', STEP=5, study=autodet.cfg.substudy, db_path=autodet.cfg.dbpath):
#    import datetime as dt
#    tp_IDs, tp_names = templates_list('templates%s_%s_Z%i.txt' %(study, version, STEP))
#    T0 = autodet.db_h5py.read_template('template%i' %tid, db_path=db_path+'TEMPLATES_%s_Z%i/' %(version, STEP))
#    D = np.zeros(len(tp_IDs), dtype=np.float32)
#    COUNTS = np.array([[], []]).reshape(2,-1)
#    for i in xrange(D.size):
#        T = autodet.db_h5py.read_template('template%i' %tp_IDs[i], db_path=db_path+'TEMPLATES_%s_Z%i/' %(version, STEP))
#        D[i] = distance(T0.latitude, T0.longitude, T0.depth, T.latitude, T.longitude, T.depth)
#        catalog = autodet.db_h5py.read_catalog_multiplets('multiplets%i' %tp_IDs[i])
#        C = np.vstack((catalog['origin_times'], D[i]*np.ones(catalog['origin_times'].size)))
#        COUNTS = np.hstack( (COUNTS, C) )
#    #------------------------------------
#    day = udt(day)
#    start_day = day - dt.timedelta(days=5)
#    end_day = day + dt.timedelta(days=5)
#    Nsec = (end_day.timestamp - start_day.timestamp)
#    r = np.log2(Nsec/DT)
#    Nbinst = np.int32(2**np.ceil(r))
#    nd = 40
#    DD = np.min(np.abs(D[1:nd] - D[:nd-1]))
#    r = np.log2((D.max() - D.min())/DD)
#    Nbinsd = np.int32(2**np.ceil(r))
#    R = [[start_day.timestamp, end_day.timestamp], [D.min(), D.max()]]
#    B = np.int32(np.floor([Nbinst, Nbinsd]))
#    return COUNTS, B, R, D
#    H = np.histogram2d(COUNTS[0,:], COUNTS[1,:], bins=[Nbinst, Nbinsd], range=[[start_day.timestamp, end_day.timestamp], [D.min(), D.max()]])
#    return H
#
#def STcorr_V2(tid, DT, day, maxlag=3600., version='V1', STEP=5, study=autodet.cfg.substudy, db_path=autodet.cfg.dbpath):
#    import datetime as dt
#    from scipy.ndimage import gaussian_filter
#    tp_IDs, tp_names = templates_list('templates%s_%s_Z%i.txt' %(study, version, STEP))
#    tp_pos = np.loadtxt(autodet.cfg.base + 'projection%s_%s_Z%i.txt' %(study, version, STEP))
#    T0 = autodet.db_h5py.read_template('template%i' %tid, db_path=db_path+'TEMPLATES_%s_Z%i/' %(version, STEP))
#    D = np.zeros(len(tp_IDs), dtype=np.float32)
#    COUNTS = np.array([[], []]).reshape(2,-1)
#    for i in xrange(D.size):
#        #T = autodet.db_h5py.read_template('template%i' %tp_IDs[i], db_path=db_path+'TEMPLATES_%s_Z%i/' %(version, STEP))
#        #D[i] = distance(T0.latitude, T0.longitude, T0.depth, T.latitude, T.longitude, T.depth)
#        D[i] = np.float32(tp_pos[i])
#        catalog = autodet.db_h5py.read_catalog_multiplets('multiplets%i' %tp_IDs[i])
#        C = np.vstack((catalog['origin_times'], D[i]*np.ones(catalog['origin_times'].size)))
#        COUNTS = np.hstack( (COUNTS, C) )
#    #------------------------------------
#    day = udt(day)
#    start_day = day - dt.timedelta(days=5)
#    end_day = day + dt.timedelta(days=5)
#    Nsec = (end_day.timestamp - start_day.timestamp)
#    r = np.log2(Nsec/DT)
#    Nbinst = np.int32(2**np.ceil(r))
#    nd = 40
#    DD = np.min(np.abs(D[1:] - D[:-1]))
#    r = np.log2((D.max() - D.min())/DD)
#    Nbinsd = np.int32(2**np.ceil(r))
#    R = [[start_day.timestamp, end_day.timestamp], [D.min(), D.max()]]
#    B = np.int32(np.floor([Nbinst, Nbinsd]))
#    H = np.histogram2d(COUNTS[0,:], COUNTS[1,:], bins=B, range=R)
#    #-------------------------------------
#    DT = (R[0][1] - R[0][0])/B[0]
#    DD = (R[1][1] - R[1][0])/B[1]
#    print "%i bins in time (DT = %.2fsec) amd %i bins in space (DD = %.2fkm)" %(B[0], DT, B[1], DD)
#    #-------------------------------------
#    freqmaxT = 1./(3.*60.) # 1/5min
#    freqmaxD = 1./10. # 1./5km
#    if 1./freqmaxD < DD:
#        freqmaxD = 1./DD
#    sm_time = int(np.ceil(1./freqmaxT/DT))
#    #sm_sp = int(np.ceil(1./freqmaxD*DD))
#    sm_sp = 3
#    print "Smoothing parameters in time = %i, in space = %i" %(sm_time, sm_sp)
#    H_sm = gaussian_filter(H[0], sigma=(sm_time, sm_sp), order=0)
#    H_sm = H_sm.T.copy()
#    #-------------------------------------
#    ml = int(maxlag/DT)
#    t1 = dt.datetime.now()
#    XC = np.zeros((2*ml, B[1]), dtype=np.float32)
#    idx_tp = int( (D[tp_IDs.index(tid)] - D.min())/DD ) - 1
#    A = np.power(H_sm[idx_tp,ml:-ml], 2).sum()
#    for j in xrange(XC.shape[1]):
#        for i in xrange(XC.shape[0]):
#            h = H_sm[j, i:i-2*ml]
#            den = np.sqrt( np.power(h, 2).sum() * A )
#            if den != 0.:
#                XC[i,j] = np.sum(h * H_sm[idx_tp, ml:-ml]) / den
#    t2 = dt.datetime.now()
#    print "Cross-correlation computed in %.2fsec" %(t2-t1).total_seconds()
#    return COUNTS, B, R, D, H_sm, XC
#
#def STcorr_az(tid, DT, day, maxlag=3600., version='V1', STEP=5, study=autodet.cfg.substudy, db_path=autodet.cfg.dbpath):
#    import datetime as dt
#    from scipy.ndimage import gaussian_filter
#    tp_IDs, tp_names = templates_list('templates%s_%s_Z%i.txt' %(study, version, STEP))
#    tp_pos = np.loadtxt(autodet.cfg.base + 'projection%s_%s_Z%i.txt' %(study, version, STEP))
#    db_path_M = 'MULTIPLETS_%s_Z%i_MAD/' %(version, STEP)
#    T0 = autodet.db_h5py.read_template('template%i' %tid, db_path=db_path+'TEMPLATES_%s_Z%i/' %(version, STEP))
#    D = np.zeros(len(tp_IDs), dtype=np.float32)
#    Az = np.zeros(len(tp_IDs), dtype=np.float32)
#    COUNTS = []
#    day = udt(day)
#    start_day = day - dt.timedelta(days=5)
#    end_day = day + dt.timedelta(days=5)
#    Nbins = (end_day.timestamp - start_day.timestamp)/DT
#    for i in xrange(D.size):
#        T = autodet.db_h5py.read_template('template%i' %tp_IDs[i], db_path=db_path+'TEMPLATES_%s_Z%i/' %(version, STEP))
#        D[i] = distance(T0.latitude, T0.longitude, T0.depth, T.latitude, T.longitude, T.depth)
#        Az[i] = azimuth(T0.latitude, T0.longitude, T.latitude, T.longitude)
#        catalog = autodet.db_h5py.read_catalog_multiplets('multiplets%i' %tp_IDs[i], db_path_M=db_path_M)
#        C, times = np.histogram(catalog['origin_times'], bins=Nbins, range=(start_day.timestamp, end_day.timestamp))
#        print C.shape
#        COUNTS.append(C)
#    COUNTS = np.asarray(COUNTS)
#    #------------------------------------
#    print COUNTS.shape, D.size, len(tp_IDs)
#    Nsec = (end_day.timestamp - start_day.timestamp)
#    #-------------------------------------
#    ml = int(maxlag/DT)
#    t1 = dt.datetime.now()
#    XC = np.zeros((D.size, 2*ml), dtype=np.float32)
#    A = COUNTS[tp_IDs.index(tid),ml:-ml]
#    denA = np.power(A, 2).sum()
#    for i in xrange(XC.shape[0]):
#        for j in xrange(XC.shape[1]):
#            den = np.sqrt( np.power(COUNTS[i,j:j-2*ml], 2).sum() * denA )
#            if den != 0.:
#                XC[i,j] = np.sum( COUNTS[i,j:j-2*ml] * A) / den
#    t2 = dt.datetime.now()
#    print "Cross-correlation computed in %.2fsec" %(t2-t1).total_seconds()
#    return np.max(XC, axis=-1), Az, D
#
#def recurrence_times(tid, version='V2', STEP=4):
#    #----------------------------------------------------------------------
#    font = {'family' : 'serif', 'size' : 20}
#    plt.rc('font', **font)
#    plt.rcParams['pdf.fonttype'] = 42 #TrueType
#    #----------------------------------------------------------------------
#    db_path_M = 'MULTIPLETS_{}_STEP{:d}_RMS/'.format(version, STEP)
#    catalog   = autodet.db_h5py.read_catalog_multiplets('multiplets{:d}'.format(tid), db_path_M=db_path_M)
#    RT = catalog['origin_times'][1:] - catalog['origin_times'][:-1]
#    RT /= (3600.*24.) # in days
#    plt.figure('recurrence_times_template_{:d}'.format(tid), figsize=(15,15))
#    #RT_max = np.percentile(RT, 95.)
#    plt.hist(np.log10(RT), bins=100)
#    plt.xlabel('Recurrence Time (log[days])')
#    plt.ylabel('Number of Events')
#    plt.title('Distribution of the Recurrence Times for Template {:d}'.format(tid))
#    plt.show()
#
##========================================================================================================
#
#def event_count_number(origin_times, T1, T2, DT):
#    start_day = udt(T1)
#    end_day = udt(T2)
#    Nbins = int((end_day.timestamp - start_day.timestamp)/DT)
#    C, times = np.histogram(origin_times, bins=Nbins, range=(start_day.timestamp, end_day.timestamp))
#    return C, times
#
#
#def clean_catalog(OT, TTs):
#    import datetime as dt
#    Q = np.ones(OT.size, dtype=np.bool)
#    for i in xrange(OT.size):
#        D = udt(OT[i]) + TTs[i,:].min()
#        D0 = udt(D.strftime('%Y,%m,%d'))
#        D1 = udt(D + dt.timedelta(days=1))
#        D1 = udt(D1.strftime('%Y,%m,%d'))
#        if (D.timestamp - D0.timestamp) < 30. or (D1.timestamp - D.timestamp) < 30.:
#            # false detection
#            Q[i] = False
#    return Q
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
#def distance(lat_source,long_source,depth_source,lat_station,long_station,depth_station):
#    """
#    Hypocentral distance (a supprimer ?)
#    """
#    R = 6371.
#    rad = np.pi/180.
#    lat_station *= rad
#    long_station *= rad
#    lat_source *= rad
#    long_source *= rad
#    delta = [lat_station - lat_source,
#             long_station - long_source]
#
#    a = (np.sin(delta[0] / 2) ** 2
#         + np.sin(delta[1] / 2) ** 2
#         * np.cos(lat_source)
#         * np.cos(lat_station))
#    c = 2 * np.arctan2(np.sqrt(a), np.sqrt(1 - a))
#
#    dist_epi = R * c
#
#    return np.sqrt(dist_epi**2 + (depth_source - depth_station)**2)
#
#def azimuth(lat1, lon1, lat2, lon2):
#    R = 6371.
#    Dlat = (lat2 - lat1) * np.pi/180.
#    Dlon = (lon2 - lon1) * np.pi/180.
#    r = R * np.sin(lat1*np.pi/180.)
#    #az = np.arctan((R*Dlon)/(r*Dlat))
#    az = np.arctan2(R*Dlon, r*Dlat)
#    return az

