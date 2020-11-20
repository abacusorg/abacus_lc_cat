# How to run:

run build: start with 0.45 and finish at 3

run save: save for those exact same redshifts as above

run match: start with 0.45 and finish at 3

run hod: select model and run for all available slices


Note: use python3.6 on alan (conflicts with packages)

# Outstanding questions:

in build: incorporate main progenitor as well as progenitors; check condition for falling into light cone

in save: use the clean CompaSO catalogs; save all fields in the catalogs; interpolation of the velocities (currently using average velocity between snapshots)

in match: compare the new particle positions with what was in interp_lc for those halos; TEST PERCENTAGE OF MATCHES

in hod: save the interpolated redshift for the centrals in build and print it out in the final mock

# Final checks

Are we repeating halos and particles?

Are there gaps between the shells?

Are we interpolating correctly (I believe there is a minus mistake right now when we do the exceptional cases in build)? Test: a b c skip b and compare clustering