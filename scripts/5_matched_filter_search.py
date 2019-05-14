import automatic_detection as autodet
import numpy as np
import matplotlib.pyplot as plt
import os
from time import time as give_time
import h5py as h5
from subprocess import Popen, PIPE
from obspy.core import UTCDateTime as udt

# change these paths accordingly
db_path_T = 'template_db_2/'
db_path_M = 'matched_filter_2/'
band = [1., 12.]

if not autodet.multiplet_search.fmf.GPU_LOADED:
    print('FMF does not work on this node... :(')
    sys.exit()

# read the database of template events
templates = autodet.db_h5py.read_template_list('database_index', db_path=autodet.cfg.dbpath+db_path_T, well_relocated_templates=False)
tids      = []
n_templates = len(templates)
for t in range(n_templates):
    tids.append(templates[t].metadata['template_idx'])
tids = np.int32(tids)

# for this simple example, n_part_templates=1 but this variable can be tuned
# in order to fit the templates on the GPU's memory
# (if n_part_templates=10 and if there are 1000 templates, then chunks of 100 templates will be processed at a time)
n_part_templates     = 1
n_templates_per_part = n_templates // n_part_templates + 1

#==========================================================
#      RESHAPE THE TEMPLATE INPUT SO THAT 
#      EACH TEMPLATE HAS AN ENTRY FOR EACH STATION
def make_templates_mat(templates, net):
    nd = len(templates)
    ns = net.n_stations()
    nc = net.n_components()
    templates_mat = np.zeros((nd, ns, nc, templates[0][0].data.size), dtype=np.float32)
    moveouts_mat = np.zeros((nd, ns, nc), dtype=np.int32)
    weights_mat = np.zeros((nd, ns, nc), dtype=np.float32)
    for d in range(nd):
        for s in range(ns):
            if net.stations[s] in templates[d].metadata['stations']:
                #ss = templates[d].metadata['stations'] == net.stations[s]
                ss = np.where(templates[d].metadata['stations'] == net.stations[s])[0]
                if ss.size == 0:
                    continue
                else:
                    ss = ss[0]
                if type(templates[d].metadata['s_moveouts'][ss]) is np.float32:
                    moveouts_mat[d,s,:2] = np.int32(templates[d].metadata['s_moveouts'][ss] * templates[d].metadata['sampling_rate'])  
                    moveouts_mat[d,s,-1] = np.int32(templates[d].metadata['p_moveouts'][ss] * templates[d].metadata['sampling_rate'])
                else:
                    print(type(templates[d].metadata['s_moveouts'][ss]))
                    moveouts_mat[d,s,:2] = templates[d].metadata['s_moveouts'][ss]  
                    moveouts_mat[d,s,-1] = templates[d].metadata['p_moveouts'][ss]
                weights_mat[d,s,:] = 1.
            else:
                continue
            for c in range(nc):
                templates_mat[d,s,c,:] = templates[d].select(station=net.stations[s], channel=templates[d].metadata['channels'][c])[0].data
        weights_mat[d,:] /= weights_mat[d,:].sum()
    return templates_mat, moveouts_mat, weights_mat
#==========================================================

net = autodet.dataset.Network('network.in')
net.read()

#dates = net.datelist()
dates = [udt('2013,03,17')]

t1 = give_time()
#if isfile(autodet.cfg.dbpath + db_path_T + 'matched_filter_input.h5'):
try:
    matched_filter_input = {}
    with h5.File(autodet.cfg.dbpath + db_path_T + 'matched_filter_input.h5', mode='r') as f:
        print('Reading the matched filter input from database...')
        for item in list(f.keys()):
            matched_filter_input[item] = f[item][()]
    templates_mat = matched_filter_input['templates_mat']
    moveouts_mat  = matched_filter_input['moveouts_mat']
    weights_mat   = matched_filter_input['weights_mat']
#else:
except:
    print('Formatting the matched filter input...')
    templates_mat, moveouts_mat, weights_mat = make_templates_mat(templates, net)
    with h5.File(autodet.cfg.dbpath + db_path_T + 'matched_filter_input.h5', mode='w') as f:
        f.create_dataset('templates_mat', data=templates_mat, compression='gzip')
        f.create_dataset('moveouts_mat',  data=moveouts_mat,  compression='gzip')
        f.create_dataset('weights_mat',   data=weights_mat,   compression='gzip')
t2 = give_time()
print('{:.2f}sec to load the input.'.format(t2-t1))

thr_type = 'RMS'
for date in dates:
    being_processed = autodet.cfg.dbpath + db_path_M + '{}_being_processed.h5'.format(date.strftime('%Y%m%d'))
    processed       = autodet.cfg.dbpath + db_path_M + '{}_processed.h5'.format(date.strftime('%Y%m%d'))
    # uncomment if you are processing lot's of data with many threads at the same time
    #if os.path.isfile(being_processed) or os.path.isfile(processed):
    #    continue
    with h5.File(being_processed, mode='w') as f:
        f.create_dataset('fake', data=np.zeros(0, dtype=np.float32))
    print('==================================================')
    print('==================================================')
    print('--------PROCESS DAY {}-------'.format(date.strftime('%Y,%m,%d')))
    data = autodet.data.ReadData(date.strftime('%Y,%m,%d'), band)
    #===========================
    list_metadata  = []
    list_waveforms = []
    CC             = []
    for i in range(n_part_templates):
        # loop over chunks of templates
        id1 = i*n_templates_per_part
        id2 = (i+1)*n_templates_per_part
        if id1 > n_templates:
            break
        if id2 > n_templates:
            id2 = n_templates
        metadata_, waveforms_, cc_ = autodet.multiplet_search.find_multiplets(templates_mat[id1:id2,:,:,:], 
                                                                              moveouts_mat[id1:id2,:,:], 
                                                                              data, 
                                                                              tids[id1:id2], 
                                                                              net, \
                                                                              threshold_type=thr_type, 
                                                                              device='gpu', 
                                                                              weights_mat=weights_mat[id1:id2,:,:])
        list_metadata.extend(metadata_)
        list_waveforms.extend(waveforms_)
        CC.extend(cc_)
    #===========================
    for t in range(n_templates):
        # save the output
        multi_name = 'multiplets{:d}_{}'.format(templates[t].metadata['template_idx'], date.strftime('%d%b%Y'))
        if list_metadata[t]['origin_times'].size > 0:
            autodet.db_h5py.write_multiplets(multi_name, list_metadata[t], list_waveforms[t]['waveforms'], db_path=autodet.cfg.dbpath + db_path_M)
            print('{}: {:d} multiplets added for Template {:d}'.format(date.strftime('%d%b%Y'), list_metadata[t]['origin_times'].size, templates[t].metadata['template_idx']))
    with h5.File(processed, mode='w') as f:
        f.create_dataset('fake', data=np.zeros(0, dtype=np.float32))
    os.system('rm ' + being_processed)
    del list_metadata, list_waveforms, CC, data
