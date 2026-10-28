# GeoConform 3D — Python Web Application

A Python-only Streamlit application for 7-parameter 3D conformal (Helmert) transformation using Horn's quaternion solution.

## Run in VS Code

1. Open this folder in VS Code.
2. Open Terminal and run `python -m pip install -r requirements.txt`.
3. Run `python -m streamlit run app.py`.
4. Open `http://localhost:8501` in your browser if it does not open automatically.

## Run the computation engine without Streamlit

`python geoconform3d.py`

The pure-Python engine can be pasted into online-python.com, but **the Streamlit web interface cannot run in an ordinary online Python compiler**.

## Features

- Five built-in test scenarios and editable user-supplied control points.
- Seven estimated parameters: Tx, Ty, Tz, Rx, Ry, Rz, scale ppm.
- Computed control-point coordinates, residuals, RMSE, sigma0.
- Additional point transformation and TXT/CSV exports.
- No interactive `input()` calls.

The built-in test data are synthetic. Always check the transformation convention, coordinate reference systems, units, point correspondences, and control geometry before real surveying use.

## Verification

Run `py -m unittest -v test_geoconform3d.py` in VS Code to run six regression tests.
The tests cover exact parameter recovery, noisy observations, extra-point transformation,
collinear rejection, invalid inputs, and an alternative transformation.

Scenario 2 intentionally adds observation errors. Its fitted parameters will differ
from the exact demonstration parameters, even though the residual RMSE is small.
Scenario 4 intentionally fails because collinear control points cannot establish a
unique 3D similarity transformation.

This is an educational equal-weight, unweighted fit. Real survey transformations
require independent check points, suitable datum definitions, and survey-quality validation.
