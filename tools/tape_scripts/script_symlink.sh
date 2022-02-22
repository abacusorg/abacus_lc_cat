#!/bin/bash
file_loc="/global/cscratch1/sd/boryanah/data_hybrid/tape_data/"
link_loc="/global/project/projectdirs/desi/cosmosim/Abacus/"
f1="halo_rv_A"
f2="halo_pid_A"
f3="halo_info"
#f4="field_rv_A"
#f5="field_pid_A"
input="sim_names.txt"
input_z="redshifts.txt"
#sim_name="AbacusSummit_base_c113_ph000" #c101, c102, c104, c105, c112, c113
exec="ln -s"
output="symlink.sh"
rm "$output"

while IFS= read -r sim_name; do
    echo -n "mkdir ${file_loc}/${sim_name}/" >> "${output}"
    echo "" >> "$output"
    echo -n "mkdir ${file_loc}/${sim_name}/halos" >> "${output}"
    echo "" >> "$output"
    echo -n "cd ${file_loc}/${sim_name}/halos" >> "$output"
    echo "" >> "$output"
    while read -r zline; do
	echo -n "mkdir z${zline}; cd z${zline}" >> "${output}"
	echo "" >> "$output"
	echo -n "${exec}" "${link_loc}/${sim_name}/halos/z${zline}/${f1}"  >> "${output}"
	echo "" >> "$output"
	echo -n "${exec}" "${link_loc}/${sim_name}/halos/z${zline}/${f2}"  >> "${output}"
	echo "" >> "$output"
	echo -n "${exec}" "${link_loc}/${sim_name}/halos/z${zline}/${f3}"  >> "${output}"
	echo "" >> "$output"
	#echo -n "${exec}" "${link_loc}/${sim_name}/halos/z${zline}/${f4}"  >> "${output}"
	#echo "" >> "$output"
	#echo -n "${exec}" "${link_loc}/${sim_name}/halos/z${zline}/${f5}"  >> "${output}"
	#echo "" >> "$output"
	echo -n "cd .." >> "$output"
	echo "" >> "$output"
    done < "$input_z"
#    	echo -n "${exec}" "${link_loc}/cleaning/${sim_name}/ cleaning/"  >> "${output}"
#	echo "" >> "$output"
done < "$input"