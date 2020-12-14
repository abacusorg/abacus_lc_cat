# alan

./build_mt.py --z_start 0.45 --z_stop 0.575 --sim_name AbacusSummit_base_c000_ph006 --merger_parent /mnt/gosling2/bigsims/merger --catalog_parent /mnt/gosling1/boryanah/light_cone_catalog/

./save_cat.py --z_start 0.45 --z_stop 0.575 --save_pos  --sim_name AbacusSummit_base_c000_ph006 --merger_parent /mnt/gosling2/bigsims/merger --compaso_parent /mnt/gosling2/bigsims/ --catalog_parent /mnt/gosling1/boryanah/light_cone_catalog/

./match_lc.py --z_lowest 0.45 --z_highest 0.575  --sim_name AbacusSummit_base_c000_ph006 --merger_parent /mnt/gosling2/bigsims/merger --light_cone_parent /mnt/gosling2/bigsims/ --catalog_parent /mnt/gosling1/boryanah/light_cone_catalog/


# NERSC

./build_mt.py --z_start 0.45 --z_stop 0.575 --sim_name AbacusSummit_highbase_c021_ph000 --merger_parent /global/project/projectdirs/desi/cosmosim/Abacus/merger --catalog_parent /global/cscratch1/sd/boryanah/light_cone_catalog/

./save_cat.py --z_start 0.45 --z_stop 0.575 --save_pos  --sim_name AbacusSummit_highbase_c021_ph000 --merger_parent /global/project/projectdirs/desi/cosmosim/Abacus/merger --compaso_parent /global/project/projectdirs/desi/cosmosim/Abacus --catalog_parent /global/cscratch1/sd/boryanah/light_cone_catalog/

./match_lc.py --z_lowest 0.45 --z_highest 0.575  --sim_name AbacusSummit_highbase_c021_ph000 --merger_parent /global/project/projectdirs/desi/cosmosim/Abacus/merger --light_cone_parent /global/project/projectdirs/desi/cosmosim/Abacus --catalog_parent /global/cscratch1/sd/boryanah/light_cone_catalog/
