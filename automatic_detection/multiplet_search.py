from . import common as cmn
from . import clib, data
import numpy as np
import h5py
import datetime as dt
import fast_matched_filter as fmf
from obspy.core import UTCDateTime as udt
from .config import cfg

from IPython.core.debugger import Tracer
debug_here = Tracer()

def find_multiplets(templates_mat, moveouts_mat, data, template_IDs, net, \
                    threshold_type='rms', weights_mat=None,
                    buf=True, device='gpu'):
    """
    Finds multiplets with different possible modes : if best_St is True, then the coherency coefficient
    used to trigger detections is calculated with the best stations. If best_Det is True, then the 
    best multiplets of the day are kept.
    """
    threshold_type = threshold_type.lower()

    nt,ns,nc,Nsamp = templates_mat.shape

    step = np.int32(cmn.to_samples(cfg.matched_filter_step, data['metadata']['sampling_rate']))

    n_stations     = data['waveforms'].shape[0]
    n_components   = data['waveforms'].shape[1]
    n_samples_data = data['waveforms'].shape[2]
    n_samples      = np.int32(cfg.multiplet_len * data['metadata']['sampling_rate'])

    buffer_extracted_events = cfg.buffer_extracted_events # select 10s before the detection

    if weights_mat is None:
        weights_mat = np.ones_like(moveouts_mat)
        for n in range(weights_mat.shape[0]):
            weights_mat[n,:] /= weights_mat[n,:].sum()

    CC_SUMS = []
    Nparts = 2
    L = ns // Nparts + 1
    for i in range(Nparts):
        # to be memory friendly, we subdivide the network into Nparts
        # and the resulting correlation coefficients are then manually stacked 
        # in a separate loop
        id1 = i*L
        id2 = (i+1)*L
        if id2 > ns:
            id2 = ns
        cc_sums = fmf.matched_filter(templates_mat[:,id1:id2,:,:],
                                      moveouts_mat[:,id1:id2],
                                      weights_mat[:,id1:id2,:],
                                      data['waveforms'][id1:id2,:,:],
                                      step,
                                      arch=device)
        CC_SUMS.append(cc_sums)
    cc_sums = CC_SUMS[0]
    for i in range(1,Nparts):
        # stack the correlation coefficients
        cc_sums += CC_SUMS[i]

    cc_sums[np.isnan(cc_sums)] = 0

    list_metadata  = []
    list_waveforms = []
    for i in range(nt):
        cc_sum = cc_sums[i,:]
        
        if threshold_type == 'rms':
            cc_sum -= np.mean(cc_sum)
            threshold = cfg.matched_filter_threshold * cmn.rms(cc_sum)
        elif threshold_type == 'mad':
            cc_sum -= np.median(cc_sum)
            threshold = cfg.matched_filter_threshold * cmn.mad(cc_sum)
        #------------------
        cc_idx = np.argwhere(cc_sum > threshold)
        detections = cc_idx * step

        if buf:
           # remove detections from buffer
           limit = np.int32(cfg.data_buffer * data['metadata']['sampling_rate'])
           idx = detections >= limit
           cc_idx = cc_idx[idx]
           detections = detections[idx]
           
           limit = np.int32((86400 + cfg.data_buffer) * data['metadata']['sampling_rate'])
           idx = detections < limit
           cc_idx = cc_idx[idx]
           detections = detections[idx]

        # only keep highest correlation coefficient for grouped detections
        #search_win = np.int32(cfg.template_len * cfg.sampling_rate / step)
        d_mv = moveouts_mat[i,:,0] - moveouts_mat[i,:,-1]
        search_win = min(np.int32(3. * cfg.template_len * cfg.sampling_rate / step), \
                         max(np.int32(1.*np.median(d_mv[d_mv != 0]) / step), \
                             np.int32(cfg.template_len * cfg.sampling_rate / step)))
        for j in range(cc_idx.size):
            idx = np.arange(max(0, cc_idx[j] - search_win // 2), min(cc_sum.size-1, cc_idx[j] + search_win // 2), dtype=np.int32)
            idx_to_update = np.where(cc_idx == cc_idx[j])[0]
            cc_idx[idx_to_update] = np.argmax(cc_sum[idx]) + idx[0]

        cc_idx = np.unique(cc_idx)
        detections = cc_idx * step

        # after this step, we can have detections closest than search_win / 2
        cc_idx = list(cc_idx)
        Nrm = 0
        for j in range(1, detections.size):
            if (cc_idx[j-Nrm]-cc_idx[j-Nrm-1]) < search_win // 2:
                if cc_sum[cc_idx[j-Nrm]] > cc_sum[cc_idx[j-Nrm-1]]:
                    cc_idx.remove(cc_idx[j-Nrm-1])
                else:
                    cc_idx.remove(cc_idx[j-Nrm])
                Nrm += 1
        cc_idx = np.asarray(cc_idx)
        detections = cc_idx * step

        n_multiplets = len(detections)
        #------------------------------------------------------
        metadata_events  = {}
        waveforms_events = {}
        origin_times             = np.zeros( n_multiplets, dtype=np.float64) 
        correlation_coefficients = np.zeros( n_multiplets, dtype=np.float32) 
        #maxima                   = np.zeros((n_multiplets, n_stations, n_components), dtype=np.float32)
        waveforms                = np.zeros((n_multiplets, n_stations, n_components, n_samples), dtype=np.float32) 
        idx_min = 0 # can't extract continuous data before index 0
        idx_max = n_samples_data # can't extract continuous data after the last sample of the day
        for d in range(n_multiplets):
            origin_time = udt(data['metadata']['date']) + detections[d] / cfg.sampling_rate
            origin_times[d] = origin_time.timestamp - buffer_extracted_events - cfg.data_buffer
            correlation_coefficients[d] = cc_sum[cc_idx[d]]
            #-----------------------------------------
            # take care of not selecting out-of-bound indexes:
            id1 = detections[d] - np.int32(buffer_extracted_events * cfg.sampling_rate)
            if id1 < idx_min:
                # will have to zero-pad the beginning of the extracted sequence
                dn_b = idx_min - id1
                id2  = np.int32(id1 + n_samples)
                id1  = np.int32(idx_min)
                #print('Have to zero-pad the beginning of the sequence with {:d} samples (id1 = {:d}, id2 = {:d}).'.format(dn_b, id1, id1 + n_samples))
            else:
                dn_b = 0
                id2 = id1 + n_samples
            if id2 > idx_max:
                # will have to zero-pad the end of the extracted sequence
                dn_e = id2 - idx_max
                id2  = np.int32(idx_max)
                #print('Have to zero-pad the end of the sequence with {:d} samples (id1 = {:d}, id2 = {:d}).'.format(dn_e, id1, id1 + n_samples))
            else:
                dn_e = 0
            waveforms[d, :, :, :] = np.concatenate( (np.zeros((n_stations, n_components, dn_b), dtype=np.float32), \
                                                     data['waveforms'][:,:,id1:id2], \
                                                     np.zeros((n_stations, n_components, dn_e), dtype=np.float32)), axis=-1 )
            #maxima[d, :, :]       = np.max(waveforms[d, :, :, :], axis=-1)
            #-----------------------------------------
        metadata_events.update({'template_id'                :   np.array([template_IDs[i]])})
        metadata_events.update({'stations'                   :   np.asarray(data['metadata']['stations']).astype('S')})
        metadata_events.update({'components'                 :   np.asarray(data['metadata']['components']).astype('S')})
        metadata_events.update({'origin_times'               :   origin_times})
        metadata_events.update({'correlation_coefficients'   :   correlation_coefficients})
        waveforms_events.update({'waveforms'                 :   waveforms})

        list_metadata.append(metadata_events)
        list_waveforms.append(waveforms_events)
    return list_metadata, list_waveforms, cc_sums

