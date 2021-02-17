# How to run:

run build: start with 0.1 and finish at 2.5

run save: save for those exact same redshifts as above

run match: start with 0.1 and finish at 2.5

run hod: select model and run for all available slices


Note: use python3.6 on alan (conflicts with packages)

# Outstanding questions:

in build: 

in save: use the clean CompaSO catalogs; save all fields in the catalogs; interpolation of the velocities (for merged halos) and masses (for all halos) if Sownak provides velocities and mass history (currently using average velocity between snapshots); marking halos as ineligible if Sownak provides merger trees;

in match: compare the new particle positions with what was in interp_lc for those halos -- kinda tested and looks fine; for the initial redshift (z=0.1), I get below interpolation but that is fine (pm 4 Mpc/h coming from Sownak's merger tree search, see build_mt.py)

in hod: save the interpolated redshift for the centrals in build and print it out in the final mock; broadcast and make into tables; bind to AbacusSummit HOD code

# Final checks:

Are we repeating halos and particles? Yes, for halos, cause of merger trees and for particles, probably cleaning and noinfo halos on the boundary; and particles in LightConeOrigin1 and LightConeOrigin2 overlap (see fix in match_lc where we separate by light cone origin)

Are there gaps between the shells? Yes cause of different number of halos at different redshifts; selecting no info halos from 0 to 1 as opposed to 1/2 and 3/2 fixes that, but not physical; there are also gaps in the particle data for particles close to the shell boundaries

# Sanity checks:

build_mt.py: checked that merger_next is output for the current superslab so don't need to open and append info; eligibility on the other hand does do that, but I think correctly.

build_mt.py: checked that the periodic wrapping for interpolated positions works and turned it off cause daniel said so (was my initial preference too). The offset_pos function does a different type of offset which is good cause it makes the light cones continuous, whereas the wrapping in solve_crossing shifts just the observer location and hence doesn't affect the continuity of light cones

