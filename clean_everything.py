# 1. Isolate ONLY Chain A Protein
with open("proteins/raw/6WV3.pdb", "r") as fin, open("receptor_A_only.pdb", "w") as fout:
    for line in fin:
        # Keep only ATOM records for Chain A
        if line.startswith("ATOM") and line[21] == "A":
            fout.write(line)

# 2. Double check if any SWF ligand accidentally became an 'ATOM'
with open("receptor_A_only.pdb", "r") as f:
    if "SWF" in f.read():
        print("ALERT: Ligand is still inside the receptor file!")
    else:
        print("Success: Receptor is clean of SWF.")
