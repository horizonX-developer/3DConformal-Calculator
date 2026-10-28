# GeoConform 3D — Version 2 (Group 1)

A seven-parameter 3D similarity / Helmert transformation teaching tool.

## Public Streamlit app
Run `pip install -r requirements.txt` then `streamlit run app.py`.
Includes editable coordinate tables, 3D source/target/computed graph, residual chart, rotation matrix, equation viewer, and text/CSV exports.

## Teacher submission — online Python compiler
Copy **the entire contents** of `online_python_version.py` into a standard Python 3 online compiler and run. It uses **only the Python standard library** and never waits for input. Change `SCENARIO = 1` at the bottom to run the five built-in demonstrations.

## Mathematical conventions
`Target = T + k * R * Source`; active rotation `R = Rz @ Ry @ Rx`; translations in metres, rotations in arcseconds, scale correction in ppm. The engine is shared in logic with the standalone file. The solution is a least-squares 3D similarity fit, not a claim of survey-grade datum accuracy.

## Test
Run `python -m unittest -v test_geoconform3d.py`.

## Deploy
GitHub repository `horizonX-developer/3DConformal-Calculator`; main file path `GeoConform3D/app.py`. Upload the updated files inside the existing `GeoConform3D` folder, not into HorizonX.
