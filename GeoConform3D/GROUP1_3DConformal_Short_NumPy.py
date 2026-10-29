"""GROUP1 3DConformal Calculator - short classroom LSA version.
Requires NumPy. Run in VS Code or an online compiler with NumPy.
This is separate from the Streamlit website.
"""
import numpy as np

# Choose 1 (exact data) or 2 (data with small errors).
SCENARIO = 2
SHOW_MATRICES = False

# Five common control points (source XYZ, in metres).
source = np.array([
    [1000, 2000, 3000], [1500, 2500, 3200],
    [2000, 1800, 3500], [2500, 2300, 2800],
    [1800, 3000, 4000]
], dtype=float)

# Small errors added to the target points for Scenario 2.
errors = np.array([
    [0.003, -0.002, 0.001], [-0.002, 0.004, -0.003],
    [0.001, -0.003, 0.002], [0.004, 0.001, -0.002],
    [-0.003, 0.002, 0.004]
])


def rotation(angles):
    """Rotation order: Rz @ Ry @ Rx; angles are arcseconds."""
    x, y, z = np.deg2rad(np.array(angles) / 3600)
    cx, cy, cz = np.cos([x, y, z])
    sx, sy, sz = np.sin([x, y, z])
    rx = np.array([[1, 0, 0], [0, cx, -sx], [0, sx, cx]])
    ry = np.array([[cy, 0, sy], [0, 1, 0], [-sy, 0, cy]])
    rz = np.array([[cz, -sz, 0], [sz, cz, 0], [0, 0, 1]])
    return rz @ ry @ rx


def transform(points, params):
    """Parameters: Tx, Ty, Tz (m), Rx, Ry, Rz (arcsec), s (ppm)."""
    scale = 1 + params[6] * 1e-6
    return params[:3] + scale * (np.asarray(points) @ rotation(params[3:6]).T)


# Demo targets are generated from known parameters.
# Replace 'source' and 'target' with your own matched XYZ arrays if needed.
known = np.array([10, -5, 8, 1.5, -2, 0.8, 2.5], dtype=float)
target = transform(source, known)
if SCENARIO == 2:
    target = target + errors

if source.shape != target.shape or source.ndim != 2 or source.shape[1] != 3:
    raise ValueError('Source and target must contain matching XYZ points')
if len(source) < 3 or not np.isfinite(source).all() or not np.isfinite(target).all():
    raise ValueError('Provide at least three valid common points')

# Initial guesses: centroid translation, zero rotations and scale.
params = np.r_[target.mean(axis=0) - source.mean(axis=0), [0., 0., 0., 0.]]
step = np.array([0.001, 0.001, 0.001, 0.05, 0.05, 0.05, 0.1])
converged = False

for iteration in range(1, 31):
    predicted = transform(source, params)
    w = (target - predicted).ravel()  # observed minus calculated

    # Design matrix A: numerical partial derivatives for 7 parameters.
    columns = []
    for j in range(7):
        change = np.zeros(7)
        change[j] = step[j]
        plus = transform(source, params + change)
        minus = transform(source, params - change)
        columns.append(((plus - minus) / (2 * step[j])).ravel())
    A = np.column_stack(columns)

    # Equal observation weights: P = I. Scale columns for stability.
    sizes = np.linalg.norm(A, axis=0)
    if np.any(sizes < 1e-14) or np.linalg.matrix_rank(A / sizes) < 7:
        raise ValueError('Control points are poorly distributed or collinear')
    B = A / sizes
    N = B.T @ B                    # scaled normal matrix
    U = B.T @ w                    # scaled right-hand vector
    correction = np.linalg.solve(N, U) / sizes
    params = params + correction

    print(f'Iteration {iteration}: max correction = {np.max(np.abs(correction)):.3g}')
    if SHOW_MATRICES and iteration == 1:
        print('First iteration A:\n', A)
        print('First iteration scaled N:\n', N)
        print('First iteration scaled U:\n', U)
    if np.max(np.abs(correction)) < 1e-8:
        converged = True
        break

if not converged:
    raise ArithmeticError('Adjustment did not converge in 30 iterations')

residuals = target - transform(source, params)
rmse = np.sqrt(np.mean(residuals ** 2))
sigma0 = np.sqrt(np.sum(residuals ** 2) / (3 * len(source) - 7))

print('\nGROUP1 3DConformal Calculator - Least Squares')
print('Converged after', iteration, 'iterations')
for name, value, unit in zip(
    ['Tx', 'Ty', 'Tz', 'Rx', 'Ry', 'Rz', 'Scale'], params,
    ['m', 'm', 'm', 'arcsec', 'arcsec', 'arcsec', 'ppm']
):
    print(f'{name:5} = {value: .10f} {unit}')
print(f'RMSE   = {rmse:.10f} m')
print(f'Sigma0 = {sigma0:.10f} m')
print('Residuals (metres):\n', np.round(residuals, 9))
print('Additional point:', transform([[1750, 2200, 3300]], params)[0])
