from astropy.io import ascii
from pathlib import Path

cat_lc_dir = Path("/mnt/gosling1/boryanah/light_cone_catalog/AbacusSummit_highbase_c000_ph100/halos_light_cones/")
sim_name = "Abacus_highbase_c000_ph100"

table = ascii.read(cat_lc_dir / "tmp" / sim_name / ("Merger_next_z%4.3f_lc%d.%02d.ecsv"%(z_this,o,k)),format='ecsv')

print(table)
