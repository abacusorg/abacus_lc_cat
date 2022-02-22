#!/bin/bash
file_loc=""
input="sim_names.txt"
output="copy_B.sh"
rm "$output"

while IFS= read -r sim_name; do
    echo -n "mkdir ${sim_name}" >> "$output"
    echo "" >> "$output"
    echo -n "cd ${sim_name}" >> "$output"
    echo "" >> "$output"
    echo -n "htar -xvf /nersc/projects/desi/cosmosim/Abacus/$sim_name/Abacus_${sim_name}_halos.tar './halos/z*/{halo}_{rv,pid}_B'" >> "$output"
    echo "" >> "$output"
    echo -n "cd .." >> "$output"
    echo "" >> "$output"
done < "$input"
