"""Iterative unweighted nonlinear least-squares 7-parameter Helmert adjustment.

Unknowns: Tx,Ty,Tz (m), Rx,Ry,Rz (arcsec), scale correction (ppm).
Residual convention: observed target minus calculated target.
"""
import math
from geoconform3d import rotation_from_euler, matvec


def _predict(p, u):
    r = rotation_from_euler(*(math.radians(v / 3600) for v in u[3:6]))
    rp = matvec(r, p)
    k = 1 + u[6]*1e-6
    return tuple(u[i]+k*rp[i] for i in range(3))


def _solve(a, b):
    """Partial-pivot elimination; raises for rank-deficient systems."""
    n = len(b)
    m = [list(map(float, a[i])) + [float(b[i])] for i in range(n)]
    for k in range(n):
        pivot = max(range(k, n), key=lambda i: abs(m[i][k]))
        if abs(m[pivot][k]) < 1e-12:
            raise ValueError('Singular normal matrix: improve control-point geometry')
        m[k], m[pivot] = m[pivot], m[k]
        for i in range(k+1, n):
            t = m[i][k]/m[k][k]
            for j in range(k+1, n+1):
                m[i][j] -= t*m[k][j]
            m[i][k] = 0.
    x = [0.]*n
    for i in range(n-1, -1, -1):
        x[i] = (m[i][n]-math.fsum(m[i][j]*x[j] for j in range(i+1,n)))/m[i][i]
    return x


def estimate_lsa(source, target, max_iterations=30):
    if len(source) != len(target) or len(source) < 3:
        raise ValueError('At least three matched control points are required')
    try:
        source = [tuple(float(v) for v in p) for p in source]
        target = [tuple(float(v) for v in p) for p in target]
    except (ValueError, TypeError) as exc:
        raise ValueError('Coordinates must be numeric') from exc
    if any(len(p)!=3 or any(not math.isfinite(v) for v in p) for p in source+target):
        raise ValueError('Coordinates must be finite XYZ triples')
    n = len(source)
    cx = [math.fsum(p[i] for p in source)/n for i in range(3)]
    cy = [math.fsum(p[i] for p in target)/n for i in range(3)]
    u = [cy[i]-cx[i] for i in range(3)] + [0.,0.,0.,0.]
    # The column scales only precondition the normal equations; the model remains nonlinear.
    history = []
    steps = [1e-3]*3 + [0.05]*3 + [0.1]
    for iteration in range(max_iterations):
        predictions = [_predict(p,u) for p in source]
        w = [target[j][i]-predictions[j][i] for j in range(n) for i in range(3)]
        cols = []
        for k in range(7):
            if k < 3:
                cols.append([float(i == k) for _ in source for i in range(3)])
            else:
                up, um = u[:],u[:]
                up[k] += steps[k]; um[k] -= steps[k]
                pp = [_predict(p,up) for p in source]
                pm = [_predict(p,um) for p in source]
                cols.append([(pp[j][i]-pm[j][i])/(2*steps[k])
                             for j in range(n) for i in range(3)])
        norms = [math.sqrt(math.fsum(v*v for v in col)) for col in cols]
        if min(norms) < 1e-14:
            raise ValueError('Degenerate geometry: a parameter cannot be determined')
        ac = [[v/norms[k] for k,v in enumerate(row)] for row in zip(*cols)]
        normal = [[math.fsum(row[i]*row[j] for row in ac) for j in range(7)] for i in range(7)]
        rhs = [math.fsum(row[i]*wi for row,wi in zip(ac,w)) for i in range(7)]
        dz = _solve(normal,rhs)
        du = [dz[i]/norms[i] for i in range(7)]
        history.append({'iteration':iteration+1, 'parameters_before':u[:], 'correction':du[:],
                        'misclosures':w[:], 'normal_scaled':normal, 'rhs_scaled':rhs})
        u = [u[i]+du[i] for i in range(7)]
        if max(abs(du[i])/(1 if i<3 else 1) for i in range(7)) < 1e-8:
            break
    else:
        raise ArithmeticError('Least-squares adjustment did not converge')
    # Rebuild final design and unscaled normal matrix at adjusted parameters.
    predictions = [_predict(p,u) for p in source]
    residuals = [tuple(target[j][i]-predictions[j][i] for i in range(3)) for j in range(n)]
    sse = math.fsum(v*v for row in residuals for v in row)
    a = []
    for p in source:
        for i in range(3):
            row=[]
            for k in range(7):
                if k<3: row.append(float(k==i))
                else:
                    up,um=u[:],u[:]
                    up[k]+=steps[k];um[k]-=steps[k]
                    row.append((_predict(p,up)[i]-_predict(p,um)[i])/(2*steps[k]))
            a.append(row)
    N = [[math.fsum(row[i]*row[j] for row in a) for j in range(7)] for i in range(7)]
    w = [v for row in residuals for v in row]
    U = [math.fsum(row[i]*wi for row,wi in zip(a,w)) for i in range(7)]
    r = rotation_from_euler(*(math.radians(v/3600) for v in u[3:6]))
    return {'translation':tuple(u[:3]),'angles':tuple(u[3:6]),'scale':1+u[6]*1e-6,
            'rotation':r,'calculated':predictions,'residuals':residuals,
            'rmse':math.sqrt(sse/(3*n)), 'sigma0':math.sqrt(sse/(3*n-7)),
            'transform':lambda p: _predict(tuple(float(x) for x in p),u),
            'lsa':{'A':a,'P':'Identity (equal unit weights)','N':N,'U':U,
                   'iterations':history,'parameters':u,'SSE':sse,'dof':3*n-7}}


def lsa_steps(result):
    d = result['lsa']
    def matrix(name, rows):
        return name+'\n'+'\n'.join('  '+' '.join(f'{x: .10g}' for x in row) for row in rows)+'\n'
    out=['ITERATIVE NONLINEAR LEAST SQUARES ADJUSTMENT',
         'Model: target = T + (1 + ppm*1e-6) Rz Ry Rx source',
         'Unknowns: Tx, Ty, Tz [m]; Rx, Ry, Rz [arcsec]; scale [ppm]',
         'Residual = observed target - calculated target',
         'P = I (equal independent coordinate weights)',
         'A = partial derivatives of predicted coordinates w.r.t. seven unknowns',
         'w = observed - predicted; N = A^T P A; U = A^T P w',
         'Solve N delta = U; update parameters; repeat until convergence.',
         'Column scaling is used internally to stabilize the linear system.',
         '']
    for step in d['iterations']:
        out.append(f"ITERATION {step['iteration']}")
        out.append('Initial parameters: '+', '.join(f'{x:.10g}' for x in step['parameters_before']))
        out.append('Parameter corrections: '+', '.join(f'{x:.10g}' for x in step['correction']))
        out.append(matrix('Preconditioned normal matrix N*:',step['normal_scaled']))
        out.append('Preconditioned RHS U*: '+', '.join(f'{x:.10g}' for x in step['rhs_scaled']))
    out.append(matrix('FINAL DESIGN MATRIX A (3n x 7):',d['A']))
    out.append(matrix('FINAL NORMAL MATRIX N = A^T P A:',d['N']))
    out.append('FINAL U = A^T P w: '+', '.join(f'{x:.12g}' for x in d['U']))
    out.append('FINAL ADJUSTED PARAMETERS: '+', '.join(f'{x:.12g}' for x in d['parameters']))
    out.append(f"SSE = {d['SSE']:.12g} m²; DOF = {d['dof']}")
    out.append(f"RMSE = {result['rmse']:.12g} m; sigma0 = {result['sigma0']:.12g} m")
    return '\n'.join(out)
