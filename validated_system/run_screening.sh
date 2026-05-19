#!/bin/bash

# Create the header for our results
echo "Ligand,Affinity(kcal/mol)" > screening_results.csv

for ligand in *.pdbqt; do
    # SKIP list: Don't dock the receptor, the reference, or previous output files
    if [[ "$ligand" == "receptor_A_only.pdbqt" || 
          "$ligand" == "reference_warfarin_chainA.pdbqt" || 
          "$ligand" == "replication_test.pdbqt" ||
          "$ligand" == *"_out.pdbqt" ]]; then
        continue
    fi

    echo "Processing $ligand..."

    # Run Vina (Vina 1.2.x uses standard output redirection instead of --log)
    vina --config vkor_config.txt \
         --ligand "$ligand" \
         --out "${ligand%.pdbqt}_out.pdbqt" > "${ligand%.pdbqt}.log"

    # Extract the affinity from the log file
    # We look for the first line that starts with '   1'
    score=$(grep -m 1 "^   1" "${ligand%.pdbqt}.log" | awk '{print $2}')

    echo "$ligand,$score" >> screening_results.csv
done

echo "-------------------------------"
echo "SCREENING COMPLETE"
column -t -s, screening_results.csv
