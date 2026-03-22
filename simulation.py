import subprocess

def update_inp_file(params):
    # modify geometry/material parameters in .inp
    pass

def run_calculix():
    subprocess.run(["ccx", "your_model"])

def extract_avg_stress():
    # parse .dat or .frd
    return 123.0

def extract_mass():
    # compute from density * volume
    return 5.0

def objective_function(params):
    update_inp_file(params)
    run_calculix()
    stress = extract_avg_stress()
    mass = extract_mass()
    return stress / mass