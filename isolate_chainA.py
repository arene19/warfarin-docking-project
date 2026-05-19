with open("proteins/raw/6WV3.pdb", "r") as fin, open("reference_warfarin_chainA.pdb", "w") as fout:
    lines = []
    for line in fin:
        # PDB standard: column index 21 is the Chain ID
        if line.startswith("HETATM") and "SWF" in line and line[21] == "A":
            fout.write(line)
            lines.append(line)

if not lines:
    print("Error: Could not find SWF in Chain A!")
else:
    x = sum(float(l[30:38]) for l in lines) / len(lines)
    y = sum(float(l[38:46]) for l in lines) / len(lines)
    z = sum(float(l[46:54]) for l in lines) / len(lines)
    print(f"True Chain A Center -> X: {x:.2f} | Y: {y:.2f} | Z: {z:.2f}")
