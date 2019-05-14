import sys
sys.path.append('/nobackup1/ebeauce/automatic_detection/')
import automatic_detection as autodet
import numpy as np
import h5py as h5

db_path_T = 'template_db_1/'

input_database = autodet.cfg.template_db_path + 'template_event_database'

template_database = autodet.db_h5py.read_template_database(input_database, db_path='')

n_templates = template_database['waveforms'].shape[0]

for n in range(n_templates):
    metadata = {}
    metadata['longitude']     = template_database['locations'][n,0]
    metadata['latitude']      = template_database['locations'][n,1]
    metadata['depth']         = template_database['locations'][n,2]
    metadata['location']      = template_database['locations'][n,:]
    metadata['sampling_rate'] = autodet.cfg.sampling_rate
    metadata['stations']      = template_database['stations'][n,:]
    metadata['channels']      = template_database['components'][n,:]
    metadata['source_idx']    = template_database['source_index'][n]
    metadata['template_idx']  = np.int32(n)
    metadata['moveouts']      = template_database['moveouts'][n,:,:]
    metadata['s_moveouts']    = template_database['moveouts'][n,:,0]
    metadata['p_moveouts']    = template_database['moveouts'][n,:,2]
    metadata['duration']      = np.float32(autodet.cfg.template_len)
    #-----------------------------------------------------------
    T = autodet.db_h5py.initialize_template(metadata)
    T.waveforms = template_database['waveforms'][n,:,:,:]
    #-----------------------------------------------------------
    autodet.db_h5py.write_template('template{:d}'.format(n), T, T.waveforms, db_path=autodet.cfg.dbpath + db_path_T)

with h5.File(autodet.cfg.dbpath + db_path_T + 'database_index.h5', mode='w') as f:
    f.create_dataset('template_indexes', data=np.arange(n_templates), compression='gzip')

print('Added {:d} templates to {}'.format(n_templates, input_database))
