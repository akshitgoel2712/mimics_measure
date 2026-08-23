Here is a clean, professional README.md draft tailored for your GitHub repository. It explains the purpose of the pipeline, how to format the input data, how to run the code, and what outputs to expect.

Aortic Dissection Centerline Geometric Pipeline
A fully automated, open-source Python computational pipeline designed to extract three-dimensional morphological and helical parameters from routine CT imaging in patients with Type B aortic dissection (TBAD).

This tool addresses the limitations of traditional 2D approximations and local polar projections by implementing an iterative longitudinal fiducial guideline method, successfully decoupling luminal twist from parent-vessel curvature in highly native, curved aortic geometries.

Features
Automated Topological Chaining: Reconstructs and orientates complex, disjointed centerline data exports.
Spatial Standardization: Normalizes anatomical coordinates to a standard origin (Left Subclavian Artery).
Advanced Morphometrics: Extracts global tortuosity, mean centerline curvature, and localized spatial eccentricity.
Volumetric Integration: Calculates precise true and false lumen volumes using 3D Frustum integration.
Robust Helical Extraction: Compares a standard local 2D polar projection method against a highly stable 3D fiducial guideline method.
Voxelation Artifact Handling: Dynamically scales a Savitzky-Golay filter to smooth high-frequency segmentation noise.

Prerequisites
This pipeline requires Python 3.8+.

You can install the required computational and visualization libraries using pip:
pip install numpy pandas scipy matplotlib

Data Preparation (Input Format)
The pipeline is designed to process centerline .txt files exported from medical image segmentation software (e.g., Materialise Mimics).
Your centerline files must contain the standard spatial vectors in their headers/columns: Px, Py, Pz (Position), Tx, Ty, Tz (Tangent), Nx, Ny, Nz (Normal), BNx, BNy, BNz (Binormal), Dfit (Diameter), and Area.

Naming Convention:
The script processes patients in "batches." For each patient/scan, you must provide exactly three text files in the same directory, sharing a common prefix (e.g., a patient ID) and ending with the specific lumen identifiers _Whole, _FL, and _TL.

Example of a valid directory structure:
Plaintext
/data_folder/
    ├── Patient001_Whole.txt
    ├── Patient001_FL.txt
    ├── Patient001_TL.txt
    ├── Patient002_Whole.txt
    ├── Patient002_FL.txt
    └── Patient002_TL.txt

Usage
Place your centerline .txt files into the same directory as the script (or modify the target_directory variable at the bottom of the script).

Run the script via your terminal or command prompt:
python centerline.py

Outputs
The script processes all complete patient batches sequentially and generates two primary outputs:

1. batch_output.csv: A compiled spreadsheet containing all extracted geometric, volumetric, and helical metrics for every processed patient. Key metrics include:
- Max/Mean Aortic Diameter
- Tortuosity Index & Mean Curvature
- Segmental Volumes & Lumen Eccentricity
- True/False Lumen Average Twist Rate, Peak Twist, Spiral Pitch, and Chirality (via both Fiducial and Polar methods).

2. Plots/ Directory: Automatically generates a 3D scatter plot (.png) for each patient, visualizing the whole aorta, the dissected lumens, and the calculated longitudinal fiducial guideline to ensure spatial accuracy.
