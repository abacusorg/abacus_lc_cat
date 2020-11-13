how to:

build: start with 0.45 and finish at 3
save: save for those exact same redshifts as above
match: start with 0.5 and finish at 2.5 (should be smaller than build I think)

# USE PYTHON3.6

questions:

build: main progenitor vs progenitors
build: untested and possibly we sometimes pick up the wrong objects because of the copying
build: currently you have to start unless you want to save the eligibility, pos_star_next, halo_ind_next, vel_star_next, number of copies and delta chi (maybe save to txt file the output)
build: BIG ISSUE WITH SOME OF THE INTERPOLATED VALUES BEING AWAY FROM STRIP (NOT RELATED TO PREV NEXT AND ELIGIBILTY)

save: use the clean catalogs
save: save all fields in the catalogs
save: interpolation of the velocities
save: I THINK BIG ISSUE WITH INDEXING IF WE HAVE HALO IND ABOVE HALO NUMBER


match: is this actually working? print and make plots
match: compare the new particle positions with what was in interp_lc for those halos
match: the position of the new dudes depends on which box we are looking at
match: BIG ISSUE WITH WHETHER WE ARE MATCHING EVERYONE

# OUTSTANDING BUG

currently:

build: problem with z = 0.8 cause of the copies, so 0.8 hasn't been saved
build, save: rename or delete old files


future:

create HODs for each redshift catalog: centrals uses halo vel/pos_interp; satellites should use the pos/vel of the pid_rv_lc tables
downsampling should happen through the ['redshift'] field

build: perhaps save the interpolated redshift in the halo files