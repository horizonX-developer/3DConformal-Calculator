# GeoConform 3D — Version 3 (Group 1)

A seven-parameter 3D similarity / Helmert transformation teaching tool.

## Public Streamlit app
Run `pip install -r requirements.txt` then `streamlit run app.py`.
Includes editable coordinate tables, 3D source/target/computed graph, residual chart, rotation matrix, equation viewer, detailed numerical computation steps, and text/CSV exports.

## Teacher submission — online Python compiler
Copy **the entire contents** of `online_python_version.py` into a standard Python 3 online compiler and run. It uses **only the Python standard library** and never waits for input. Change `SCENARIO = 1` at the bottom to run the five built-in demonstrations.

## Mathematical conventions
`Target = T + k * R * Source`; active rotation `R = Rz @ Ry @ Rx`; translations in metres, rotations in arcseconds, scale correction in ppm. The engine is shared in logic with the standalone file. The solution is a least-squares 3D similarity fit, not a claim of survey-grade datum accuracy.

## Test
Run `py -m unittest -v test_geoconform3d.py` on Windows, or `python -m unittest -v test_geoconform3d.py` elsewhere.

## Deploy
GitHub repository `horizonX-developer/3DConformal-Calculator`; main file path `GeoConform3D/app.py`. Upload the updated files inside the existing `GeoConform3D` folder, not into HorizonX.

## Version 3 detailed solution
The Streamlit app now shows and exports numerical working for centroids, centered coordinates, cross-covariance, Horn matrix, quaternion, rotation, scale, translation, residuals, RMSE, and fully substituted additional-point calculations. Fully blank additional-point rows are ignored; partially filled rows raise a clear validation error.

## Version 3.2 — GROUP1 3DConformal Calculator

- **Least Squares Adjustment (default):** iterative nonlinear, unweighted coordinate adjustment using the exact rotation model `Rz @ Ry @ Rx`. Shows the design matrix `A`, identity weight matrix `P`, normal matrix `N = Aᵀ P A`, right-hand vector `U = Aᵀ P w`, iteration corrections, final parameters and residuals. Normal equations are solved with column preconditioning and partial-pivot elimination (not explicit matrix inversion).
- **Horn quaternion (comparison):** retains the original Version 3.1 solver and its detailed solution.
- Retains demonstration scenarios, editable controls, additional points, 3D plot and exports.
- The two methods both minimize squared XYZ residuals under equal weights and should agree on well-conditioned data. Neither proves professional field accuracy.
- No instructor lecture notes were available; the implementation follows a standard iterative nonlinear least-squares formulation, which may differ from a specific class's linearized small-angle convention.

Run `py -m unittest discover -v` to execute regression tests.
