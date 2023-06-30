#!/bin/bash

base_file="hpc.sh"

for i in $(seq 0.5 0.5 20.0)
do
    new_file="hpc-${i//.}.sh"

    cp "$base_file" "$new_file"
    chmod +x "$new_file"
done
