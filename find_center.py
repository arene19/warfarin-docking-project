with open("reference_warfarin.pdb", "r") as f:
    lines = [l for l in f if l.startswith("HETATM") or l.startswith("ATOM")]
    if len(lines) == 0:
        print("Error: No atoms found!")
    else:
        x = sum(float(l[30:38]) for l in lines) / len(lines)
        y = sum(float(l[38:46]) for l in lines) / len(lines)
        z = sum(float(l[46:54]) for l in lines) / len(lines)
        print(f"True Grid Center -> X: {x:.2f} | Y: {y:.2f} | Z: {z:.2f}")

