import h5py as h5
import os
from .config import cfg
import numpy as np
from numpy import asarray as ar
from . import data, common
from obspy import UTCDateTime as udt
from obspy import Stream, Trace

#=================================================================================
#                FUNCTIONS FOR DETECTIONS
#=================================================================================

def write_detections(filename, detections, db_path=cfg.dbpath):
    """
    write_detection(filename, detections, db_path=cfg.dbpath)\n
    detections is a dictionary
    """
    f_meta = db_path + filename + 'meta.h5'
    with h5.File(f_meta, mode='w') as f:
        for item in detections.keys():
            if item == 'waveforms':
                # the waveforms are written in a separate file
                continue
            f.create_dataset(item, data=detections[item], compression='gzip')
    #-------------------------------------------------------------------
    f_wave = db_path + filename + 'wav.h5'
    with h5.File(f_wave, mode='w') as f:
        f.create_dataset('waveforms', data=detections['waveforms'], compression='gzip')

def read_detections(filename, attach_waveforms=True, db_path=cfg.dbpath):
    """
    read_detections(filename, attach_waveforms=True, db_path=cfg.dbpath)\n
    """
    detections = {}
    with h5.File(db_path + filename + 'meta.h5', mode='r') as f:
        for item in f.keys():
            detections.update({item : f[item][()]})
    if attach_waveforms:
        with h5.File(db_path + filename + 'wav.h5', mode='r') as f:
            detections.update({'waveforms' : f['waveforms'][()]})
    detections['stations']   = detections['stations'].astype('U')
    detections['components'] = detections['components'].astype('U')
    return detections

def read_template_database(filename, db_path=cfg.dbpath):
    """
    read_template_database(filename, db_path=cfg.template_db_path')\n
    """
    template_db = {}
    with h5.File(db_path + filename + '.h5', mode='r') as f:
        for item in list(f.keys()):
            template_db[item] = f[item][()]
    return template_db

#=================================================================================
#                       FUNCTIONS FOR TEMPLATES
#=================================================================================

def initialize_template(metadata):
    T = Stream()
    for st in metadata['stations']:
        for ch in metadata['channels']:
            T += Trace(data=np.zeros(np.int32(metadata['duration']*metadata['sampling_rate']), dtype=np.float32))
            T[-1].stats.station = st
            T[-1].stats.channel = ch
            T[-1].stats.sampling_rate = metadata['sampling_rate']
    T.metadata = metadata
    return T

def write_template(filename, template, waveforms, db_path=cfg.dbpath):
    f_meta = db_path+filename+'meta.h5'
    f_wave = db_path+filename+'wav.h5'
    with h5.File(f_meta, 'w') as fm:
        for key in list(template.metadata.keys()):
            if type(template.metadata[key]) == np.ndarray:
                fm.create_dataset(key, data = template.metadata[key], compression='gzip')
            else:
                fm.create_dataset(key, data = template.metadata[key])
    with h5.File(f_wave, 'w') as fw:
        fw.create_dataset('waveforms', data = waveforms, compression='gzip')

def read_template(filename, db_path=cfg.dbpath+'TEMPLATES_V1_STEP3/', attach_waveforms=False):
    f_meta = db_path+filename+'meta.h5'
    f_wave = db_path+filename+'wav.h5'
    with h5.File(f_meta, mode='r') as fm:
        metadata = {}
        for key in fm.keys():
            metadata.update({key:fm[key][()]})
    with h5.File(f_wave, mode='r') as fw:
        waveforms = fw['waveforms'][()]
    metadata['stations'] = metadata['stations'].astype('U')
    metadata['channels'] = metadata['channels'].astype('U')
    T = initialize_template(metadata)
    for s,st in enumerate(metadata['stations']):
        for c,ch in enumerate(metadata['channels']):
            T.select(station=st, channel=ch)[0].data = waveforms[s,c,:]
    if attach_waveforms:
        T.waveforms = waveforms
    return T

def read_template_list(database_index, db_path=cfg.dbpath+'template_db_1', attach_waveforms=True, well_relocated_templates=False):
    with h5.File(db_path + database_index + '.h5', mode='r') as f:
        if well_relocated_templates:
            template_indexes = np.intersect1d(f['well_relocated_template_indexes'][()], f['template_indexes'][()])
        else:
            template_indexes = f['template_indexes'][()]
    templates = []
    for tid in template_indexes:
        templates.append(read_template('template{:d}'.format(tid), db_path=db_path, attach_waveforms=attach_waveforms))
    return templates

def get_template_ids_list(database_index, db_path=cfg.dbpath+'template_db_1/', well_relocated_templates=False):
    with h5.File(db_path + database_index + '.h5', mode='r') as f:
        if well_relocated_templates:
            template_indexes = np.intersect1d(f['well_relocated_template_indexes'][()], f['template_indexes'][()])
        else:
            template_indexes = f['template_indexes'][()]
    return template_indexes

#=================================================================================
#                       FUNCTIONS FOR MULTIPLETS
#=================================================================================

def write_multiplets(filename, metadata, waveforms, db_path=cfg.dbpath):
    filename_meta = db_path + filename + 'meta.h5'
    filename_wave = db_path + filename + 'wav.h5'
    with h5.File(filename_meta, mode='w') as f:
        for item in metadata.keys():
            f.create_dataset(item, data=metadata[item], compression='gzip')
    with h5.File(filename_wave, mode='w') as f:
        f.create_dataset('waveforms', data=waveforms, compression='lzf')


def read_multiplet(filename, idx, return_tp=False, db_path=cfg.dbpath, db_path_T='template_db_1/', db_path_M='matched_filter_1/'):
    """
    read_multiplet(filename, idx, db_path=cfg.dbpath) \n
    """
    S = Stream()
    f_meta = db_path+db_path_M+filename+'meta.h5'
    fm = h5.File(f_meta, 'r')
    T = read_template('template{:d}'.format(fm['template_id'][0]), db_path=db_path+db_path_T)
    f_wave = db_path+db_path_M+filename+'wav.h5'
    fw = h5.File(f_wave, 'r')
    waveforms = fw['waveforms'][idx,:,:,:]
    fw.close()
    #---------------------------------
    stations   = fm['stations'][:].astype('U')
    components = fm['components'][:].astype('U')
    ns         = len(stations)
    nc         = len(components)
    #---------------------------------
    date = udt(fm['origin_times'][idx])
    for s in range(ns):
        for c in range(nc):
            S += Trace(data = waveforms[s,c,:])
            S[-1].stats['station'] = stations[s]
            S[-1].stats['channel'] = components[c]
            S[-1].stats['sampling_rate'] = cfg.sampling_rate
            S[-1].stats.starttime = date
    S.s_moveouts  = T.metadata['s_moveouts']
    S.p_moveouts  = T.metadata['p_moveouts']
    S.source_idx  = T.metadata['source_idx']
    S.template_ID = T.metadata['template_idx']
    S.latitude    = T.metadata['latitude']
    S.longitude   = T.metadata['longitude']
    S.depth       = T.metadata['depth']
    S.corr = fm['correlation_coefficients'][idx]
    S.stations   = stations.tolist()
    S.components = components.tolist()
    fm.close()
    if return_tp:
        return S, T
    else:
        return S

def write_catalog_multiplets(filename, catalog, db_path=cfg.dbpath, db_path_M='MULTIPLETS_V1_Z5_MAD/'):
    """
    write_meta_multiplets(filename, metadata, categories, db_path=cfg.dbpath) \n
    """
    fmeta = db_path+db_path_M+filename+'catalog.h5'
    with h5.File(fmeta, mode='w') as fm:
        for category in list(catalog.keys()):
            fm.create_dataset(category, data = catalog[category], compression='gzip')

def read_catalog_multiplets(filename, object_from_cat='', db_path=cfg.dbpath, db_path_M='MULTIPLETS_V1_Z5_MAD/'):
    """
    read_catalog_multiplets(filename, db_path=cfg.dbpath, db_path_M='MULTIPLETS_V1_Z5_MAD/') \n
    """
    fmeta = db_path+db_path_M+filename+'catalog.h5'
    catalog = {}
    with h5.File(fmeta, mode='r') as fm:
        if object_from_cat == '':
            for category in list(fm.keys()):
                catalog.update({category:fm[category][:]})
        else:
            catalog = fm[object_from_cat][:]
    return catalog

