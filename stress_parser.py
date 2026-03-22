import numpy as np

def parse_frd_stress(frd_file):

    stresses = []

    with open(frd_file, "r") as f:
        lines = f.readlines()

    reading = False

    for line in lines:

        if " -4  STRESS" in line:
            reading = True
            continue

        if reading:

            if line.startswith(" -3"):
                break

            parts = line.split()

            if len(parts) >= 7:
                try:
                    sxx = float(parts[1])
                    syy = float(parts[2])
                    szz = float(parts[3])
                    sxy = float(parts[4])
                    syz = float(parts[5])
                    sxz = float(parts[6])

                    vm = np.sqrt(
                        0.5 * (
                            (sxx - syy)**2 +
                            (syy - szz)**2 +
                            (szz - sxx)**2 +
                            6*(sxy**2 + syz**2 + sxz**2)
                        )
                    )

                    stresses.append(vm)

                except:
                    pass

    return np.mean(stresses)