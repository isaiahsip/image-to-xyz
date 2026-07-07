import subprocess
import rowan
from stjames import Molecule
from pathlib import Path
from dotenv import load_dotenv
import os

load_dotenv()
rowan.api_key= os.environ["ROWAN_API_KEY"]

SCRIPT_DIR = Path(__file__).resolve().parent
file_path = SCRIPT_DIR / "points.xyz"

with open(file_path) as f:
    contents = f.read()

image = "rowanlogo.png"

result = subprocess.run(
    ["python3", "svg_image_to_3d_points.py", image, "--format", "xyz"],
    text=True
)

print(result.stdout)

molecule = Molecule.from_file(file_path, 0, 0) 

workflow=rowan.submit_basic_calculation_workflow(
    name=image,
    tasks=["optimize"],
    initial_molecule=molecule,
    is_draft=True
)

print("\ndone!")