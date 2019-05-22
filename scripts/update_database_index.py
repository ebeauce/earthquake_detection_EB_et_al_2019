import automatic_detection as autodet
import h5py as h5
import numpy as np

# change these paths according to your needs
# for example, after the first match filter, you built 
# the database template_db_2 so you should have:
# db_path_T_1 = 'template_db_1/'
# db_path_T_2 = 'template_db_2/'

db_path_T_1 = 'template_db_1/'
db_path_T_2 = 'template_db_2/'

with h5.File(autodet.cfg.dbpath + db_path_T_1 + 'database_index.h5', mode='r') as f:
    template_indexes = f['template_indexes'][()]

templates_to_remove = []
with open(autodet.cfg.dbpath + db_path_T_2 + 'problem_relocalization.txt', 'r') as f:
    line = f.readline()[:-1]
    while len(line) != 0:
        templates_to_remove.append(np.int32(line[len('template'):]))
        line = f.readline()[:-1]

try:
    with open(autodet.cfg.dbpath + db_path_T_2 + 'redundant_templates.txt', 'r') as f:
        line = f.readline()[:-1]
        while len(line) != 0:
            print(line)
            templates_to_remove.append(np.int32(line[len('template'):]))
            line = f.readline()[:-1]
except:
    pass

templates_to_remove = np.unique(np.int32(templates_to_remove))

new_template_indexes = np.setdiff1d(template_indexes, templates_to_remove)
print('{:d} templates in the new database.'.format(new_template_indexes.size))

good_location_template_indexes = None
#--------------------------------------------------------------------------------------
#          SELECT TEMPLATES BASED ON THE QUALITY OF THEIR RELOCATION
# first run this script with localization_uncertainty_criterion = None and then
# set localization_uncertainty_criterion to some value
#localization_uncertainty_criterion = 15. # in kilometers
localization_uncertainty_criterion = None
if localization_uncertainty_criterion is not None:
    good_location_template_indexes = []
    templates = autodet.db_h5py.read_template_list('database_index', db_path=autodet.cfg.dbpath+db_path_T_2, attach_waveforms=False)
    for t in range(len(templates)):
        if templates[t].metadata['loc_uncertainty'] < localization_uncertainty_criterion:
            good_location_template_indexes.append(templates[t].metadata['template_idx'])
    good_location_template_indexes = np.setdiff1d(good_location_template_indexes, templates_to_remove)
#--------------------------------------------------------------------------------------

with h5.File(autodet.cfg.dbpath + db_path_T_2 + 'database_index.h5', mode='w') as f:
    f.create_dataset('template_indexes', data=new_template_indexes, compression='gzip')
    if localization_uncertainty_criterion is not None:
        f.create_dataset('well_relocated_template_indexes', data=np.int32(good_location_template_indexes), compression='gzip')
