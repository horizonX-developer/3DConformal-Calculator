"""Transparent numerical working for the Horn 7-parameter similarity fit."""
import math
from geoconform3d import matvec, largest_eigenvector


def computation_steps(source, target, result, ids=None, additional_points=None):
    """Return a reproducible text explanation using the actual input values."""
    n = len(source)
    ids = ids or [f"P{i+1:02d}" for i in range(n)]
    cx = tuple(math.fsum(p[j] for p in source)/n for j in range(3))
    cy = tuple(math.fsum(p[j] for p in target)/n for j in range(3))
    x = [tuple(p[j]-cx[j] for j in range(3)) for p in source]
    y = [tuple(p[j]-cy[j] for j in range(3)) for p in target]
    denom = math.fsum(t*t for p in x for t in p)
    b = [[math.fsum(x[k][i]*y[k][j] for k in range(n)) for j in range(3)] for i in range(3)]
    xx,xy,xz=b[0]; yx,yy,yz=b[1]; zx,zy,zz=b[2]
    h=[[xx+yy+zz, yz-zy, zx-xz, xy-yx],
       [yz-zy, xx-yy-zz, xy+yx, zx+xz],
       [zx-xz, xy+yx, -xx+yy-zz, yz+zy],
       [xy-yx, zx+xz, yz+zy, -xx-yy+zz]]
    q=largest_eigenvector(h)
    if q[0] < 0: q=[-v for v in q]
    r=result['rotation']
    numerator=math.fsum(y[k][j]*matvec(r,x[k])[j] for k in range(n) for j in range(3))
    k=result['scale']
    rc=matvec(r,cx)
    ss=math.fsum(t*t for p in result['residuals'] for t in p)
    f=lambda p: '('+', '.join(f'{z:.10f}' for z in p)+')'
    lines=['GEOCONFORM 3D — DETAILED COMPUTATION WORKING',
           'Model: Y = T + k R X, k = 1 + ppm × 10^-6',
           'Active column-vector rotation: R = Rz Ry Rx',
           'Coordinates: metres; angles: arcseconds',
           'Method: Horn quaternion least-squares similarity fit',
           '',f'STEP 1 — INPUT AND CONTROL POINTS (n = {n})']
    for pid,a,t in zip(ids,source,target):
        lines.append(f'{pid}: source={f(a)}  target={f(t)}')
    lines += ['', 'STEP 2 — CENTROIDS',
              'C_source = (ΣX/n, ΣY/n, ΣZ/n) = '+f(cx),
              'C_target = (ΣX_target/n, ΣY_target/n, ΣZ_target/n) = '+f(cy),
              '', 'STEP 3 — CENTERED COORDINATES',
              'x_i = X_i - C_source; y_i = Y_i - C_target']
    for pid,a,t in zip(ids,x,y): lines.append(f'{pid}: x={f(a)}  y={f(t)}')
    lines += ['', 'STEP 4 — CROSS-COVARIANCE MATRIX',
              'B_ij = Σ(x_ki × y_kj) (not divided by n)']
    lines += ['  '.join(f'{v: .10f}' for v in row) for row in b]
    lines += ['', 'STEP 5 — HORN 4×4 SYMMETRIC MATRIX',
              'Form H from B; solve H q = λ_max q, with ||q||=1.']
    lines += ['  '.join(f'{v: .10f}' for v in row) for row in h]
    lines += ['Largest-eigenvalue unit quaternion (w,x,y,z) = '+f(q),
              'Quaternion sign is arbitrary: q and -q give the same rotation.',
              '', 'STEP 6 — ROTATION MATRIX',
              'R = quaternion_to_rotation(q)']
    lines += ['  '.join(f'{v: .12f}' for v in row) for row in r]
    lines += ['Euler decomposition: R = Rz Ry Rx',
              'Rx = atan2(R32,R33) × 206264.806247 = %.9f arcsec'%result['angles'][0],
              'Ry = asin(-R31) × 206264.806247 = %.9f arcsec'%result['angles'][1],
              'Rz = atan2(R21,R11) × 206264.806247 = %.9f arcsec'%result['angles'][2],
              '', 'STEP 7 — SCALE',
              'D = Σ ||x_i||² = %.12f'%denom,
              'N = Σ y_i · (R x_i) = %.12f'%numerator,
              'k = N / D = %.12f'%k,
              'Scale correction = (k - 1) × 10^6 = %.9f ppm'%((k-1)*1e6),
              '', 'STEP 8 — TRANSLATION',
              'R C_source = '+f(rc),
              'T = C_target - k R C_source = '+f(result['translation']),
              '', 'STEP 9 — TRANSFORMED CONTROLS AND RESIDUALS',
              'Y_hat_i = T + k R X_i; v_i = Y_target_i - Y_hat_i']
    for pid,a,calc,res in zip(ids,source,result['calculated'],result['residuals']):
        lines += [f'{pid}: X={f(a)}',f'  Y_hat={f(calc)}',f'  residual={f(res)}']
    lines += ['', 'STEP 10 — FIT STATISTICS',
              'SSE = Σ(vx²+vy²+vz²) = %.15f m²'%ss,
              'Component RMSE = sqrt(SSE/(3n)) = sqrt(%.15f/%d) = %.12f m'%(ss,3*n,result['rmse']),
              'Degrees of freedom = 3n - 7 = %d'%(3*n-7),
              'Unit-weight sigma0 = sqrt(SSE/(3n-7)) = %.12f m'%result['sigma0'],
              'Near-zero values for exact synthetic controls reflect numerical precision, not field accuracy.',
              '', 'STEP 11 — ADDITIONAL POINTS',
              'For each additional point, apply Y_new = T + k R X_new.']
    if not additional_points:
        lines.append('No additional points were supplied.')
    for point_id, point in additional_points or []:
        rotated = matvec(r, point)
        scaled = tuple(k * v for v in rotated)
        computed = result['transform'](point)
        lines.extend([
            '', f'Point {point_id}:',
            'X_new = ' + f(point),
            'R X_new = ' + f(rotated),
            'k (R X_new) = ' + f(scaled),
            'T = ' + f(result['translation']),
            'X_prime = T + k (R X_new) = ' + f(computed),
        ])
        for i, axis in enumerate('XYZ'):
            lines.append(
                f"{axis}_prime = {result['translation'][i]:.10f} + "
                f"({k:.12f} × {rotated[i]:.10f}) = {computed[i]:.10f} m"
            )
    return '\n'.join(lines)
