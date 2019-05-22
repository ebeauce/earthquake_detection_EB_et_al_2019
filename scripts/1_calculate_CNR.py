import automatic_detection as autodet
import numpy as np
import matplotlib.pyplot as plt
from time import time as give_time
from obspy.core import UTCDateTime as udt
from os.path import isfile

font = {'family' : 'serif', 'size' : 14}
plt.rc('font', **font)
plt.rcParams['pdf.fonttype'] = 42 #TrueType

net = autodet.dataset.Network('network.in')
net.read()

# if you have Nvidia GPUs:
device = 'gpu'
# else:
#device ='cpu'

print('The codes will run on {}'.format(device))
print('If this was not your intention, edit this script and comment the right line for the variable "device".')

# whether you use the P-wave moveouts to align the vertical trace or not
#method = 'S'
method = 'SP'

filename = 'subgrid_downsampled'

t1 = give_time()
if method == 'S':
    MV = autodet.moveouts.MV_object(filename, net, \
                                    relative=True, \
                                    remove_airquakes=True)
elif method == 'SP':
    MV = autodet.moveouts.MV_object(filename, net, \
                                    relativeSP=True, \
                                    remove_airquakes=True)
t2 = give_time()
print('{:.2f}sec to load the moveouts.'.format(t2-t1))

test_points = np.arange(MV.n_sources, dtype=np.int32) # create a vector with indexes for every potential seismic source
test_points = test_points[MV.idx_EQ] # remove the airquakes by removing some of the indexes

band = [1., 12.] # used to know where to get the data if folders with different frequency bands exist (not relevant for this example)
n_closest_stations = 20 # stacking is performed on the 20 stations closest to each grid point
envelopes  = True  # CNR is calculated with the envelopes of the waveforms
saturation = True  # tanh is used to cutoff the 95th percentile of the envelopes

dates = [udt('2013,03,17')] # from this example, you can see that it is easy to loop over several days and automize the processing...

print('Dates to process:', dates)

for date in dates:
    filename = 'detections_{}_'.format(date.strftime('%Y%m%d'))
    #if isfile(autodet.cfg.dbpath + filename + 'meta.h5'):
    #    # you can comment this condition, but is usually useful when you process
    #    # many days and have to re-run the processing and start from where
    #    # your last run stopped
    #    continue
    T1 = give_time()
    #------------------------------------------------------------
    t1 = give_time()
    data = autodet.data.ReadData(date.strftime('%Y,%m,%d'), band)
    t2 = give_time()
    print('{:.2f}sec to load the data.'.format(t2-t1))   
    #------------------------------------------------------------
    t1 = give_time()
    NR = autodet.template_search.calc_network_response(data, MV, method, device=device, n_closest_stations=n_closest_stations, envelopes=envelopes, test_points=test_points, saturation=saturation)
    t2 = give_time()
    print('{:.2f}sec to compute the beamformed network response.'.format(t2-t1))
    #------------------------------------------------------------
    detections = autodet.template_search.find_templates(data, NR, MV, closest=True)
    #------------------------------------------------------------
    autodet.db_h5py.write_detections(filename, detections)
    #------------------------------------------------------------
    T2 = give_time()
    print('Total time to process day {}: {:.2f}sec'.format(date.strftime('%Y,%m,%d'), T2-T1))
    print('====================================================\n')


# we provide a few plotting functions with which the user can play to visualize what was detected 

def plot_detection(idx):
    """
    Plots the potential template event associated with detection #idx
    """
    plt.figure('detection_{}'.format(idx))
    n_stations   = detections['waveforms'].shape[1]
    n_components = detections['waveforms'].shape[2]
    n_samples    = detections['waveforms'].shape[3]
    max_time     = autodet.cfg.template_len + detections['moveouts'][idx].max()
    plt.suptitle(r'Detection on {}, Event located at {:.2f}$^{{\mathrm{{o}}}}$ / {:.2f}$^{{\mathrm{{o}}}}$ / {:.2f}km'.format(udt(detections['origin_times'][idx]).strftime('%Y-%m-%d | %H:%M:%S'),\
                                                                                                                              detections['locations'][idx][0],\
                                                                                                                              detections['locations'][idx][1],\
                                                                                                                              detections['locations'][idx][2]))
    for s in range(n_stations):
        for c in range(n_components):
            plt.subplot(n_stations, n_components, s * n_components + c + 1)
            time = np.linspace(detections['moveouts'][idx, s, c], detections['moveouts'][idx, s, c] + autodet.cfg.template_len, detections['waveforms'].shape[-1])
            plt.plot(time, detections['waveforms'][idx, s, c, :], lw=0.75, label='{}.{}'.format(detections['stations'][idx, s].decode('utf-8'), \
                                                                                                detections['components'][c].decode('utf-8')))
            plt.xlim(0., max_time)
            plt.legend(loc='best', frameon=False, handlelength=0.1)
            if s != n_stations-1:
                plt.xticks([])
            if s == n_stations-1:
                plt.xlabel('Time (s)')
    plt.subplots_adjust(top=0.94,
            bottom=0.085,
            left=0.065,
            right=0.955,
            hspace=0.2,
            wspace=0.2)
    plt.show()

def plot_composite_network_response(composite):
    """
    Plots the composite network response, shows which peaks are detections and the time-varying threshold.
    """
    plt.figure('composite_network_response')
    ax1     = plt.subplot2grid((1, 4), (0,0))
    _, _, _ = plt.hist(composite, bins=100)
    mad     = np.median(np.abs(composite - np.median(composite)))
    threshold1 = np.median(composite) + autodet.cfg.ratio_threshold * mad
    plt.axvline(threshold1, lw=2, ls='--', color='k', label=r'Global $\mathrm{{median}}\ + {:d} \times \mathrm{{MAD}}$'.format(int(autodet.cfg.ratio_threshold)))
    plt.legend(loc='upper right', fancybox=True)
    plt.xlabel('Composite Network Response')
    plt.ylabel('Frequency')
    #--------------------------------------------------
    ax2     = plt.subplot2grid((1, 4), (0, 1), colspan=3)
    plt.plot(composite)
    n_detections = detections['origin_times'].size
    T0 = udt(udt(detections['origin_times'][0]).strftime('%Y,%m,%d')).timestamp
    for d in range(n_detections-1):
        idx = np.int32((detections['origin_times'][d] - T0) * autodet.cfg.sampling_rate)
        plt.plot(idx, detections['composite_network_response'][d], ls='', marker='o', color='C3')
    idx = np.int32((detections['origin_times'][n_detections-1] - T0) * autodet.cfg.sampling_rate)
    plt.plot(idx, detections['composite_network_response'][n_detections-1], ls='', marker='o', color='C3', label='Candidate Template Events')
    threshold2 = autodet.template_search.time_dependent_threshold(composite, np.int32(0.5 * 3600. * autodet.cfg.sampling_rate))
    plt.plot(threshold2, color='C5', ls='--', lw=2, label=r'Sliding $\mathrm{{median}}\ + {:d} \times \mathrm{{MAD}}$'.format(int(autodet.cfg.ratio_threshold)))
    xticks       = np.arange(0, composite.size+1, np.int32(2.*3600.*autodet.cfg.sampling_rate))
    plt.legend(loc='upper right', fancybox=True)
    xtick_labels = [udt(X/autodet.cfg.sampling_rate).strftime('%H:%M:%S') for X in xticks]
    plt.xticks(xticks, xtick_labels)
    plt.xlim(0, composite.size+1)
    plt.grid(axis='x')
    plt.xlabel('Time')
    plt.ylabel('Composite Network Response')
    #----------------------------
    plt.subplots_adjust(top=0.88,
            bottom=0.5,
            left=0.07,
            right=0.97,
            hspace=0.245,
            wspace=0.275)
    plt.show()
