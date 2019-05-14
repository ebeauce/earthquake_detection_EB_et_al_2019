import os
import automatic_detection as autodet

n_steps = 3
for i in range(n_steps):
    os.system('rm -r {}template_db_{:d}/'.format(autodet.cfg.dbpath, i+1))
    os.system('mkdir {}template_db_{:d}/'.format(autodet.cfg.dbpath, i+1))
    os.system('rm -r {}matched_filter_{:d}/'.format(autodet.cfg.dbpath, i+1))
    os.system('mkdir {}matched_filter_{:d}/'.format(autodet.cfg.dbpath, i+1))
