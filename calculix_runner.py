import subprocess

def run_ccx(inp_path):
    name = inp_path.replace(".inp", "")
    subprocess.run(["ccx", name], check=True)

    