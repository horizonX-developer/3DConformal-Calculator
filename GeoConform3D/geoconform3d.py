"""Pure-Python 7-parameter 3D conformal transformation (Horn quaternion)."""
import math


def matvec(a, p):
    return tuple(math.fsum(a[i][j] * p[j] for j in range(3)) for i in range(3))


def rotation_from_euler(rx, ry, rz):
    cx, sx = math.cos(rx), math.sin(rx)
    cy, sy = math.cos(ry), math.sin(ry)
    cz, sz = math.cos(rz), math.sin(rz)
    return [[cz*cy, cz*sy*sx-sz*cx, cz*sy*cx+sz*sx],
            [sz*cy, sz*sy*sx+cz*cx, sz*sy*cx-cz*sx],
            [-sy, cy*sx, cy*cx]]


def make_target(points, translations=(10., -5., 8.),
                angles_arcsec=(1.5, -2., .8), ppm=2.5):
    angles = [math.radians(a / 3600) for a in angles_arcsec]
    r = rotation_from_euler(*angles)
    scale = 1 + ppm * 1e-6
    return [tuple(translations[i] + scale * matvec(r, p)[i]
                  for i in range(3)) for p in points]


def largest_eigenvector(matrix, tolerance=1e-14, max_steps=200):
    a = [list(map(float, row)) for row in matrix]
    v = [[float(i == j) for j in range(4)] for i in range(4)]
    magnitude = max(1., max(abs(z) for row in a for z in row))
    for _ in range(max_steps):
        p, q = max(((i, j) for i in range(4) for j in range(i+1, 4)),
                   key=lambda ij: abs(a[ij[0]][ij[1]]))
        if abs(a[p][q]) <= tolerance * magnitude:
            idx = max(range(4), key=lambda i: a[i][i])
            vec = [v[i][idx] for i in range(4)]
            norm = math.sqrt(math.fsum(z*z for z in vec))
            return [z / norm for z in vec]
        tau = (a[q][q] - a[p][p]) / (2 * a[p][q])
        t = (1 if tau >= 0 else -1) / (abs(tau) + math.hypot(1, tau))
        c = 1 / math.sqrt(1 + t*t)
        s = t*c
        apq = a[p][q]
        a[p][p] -= t*apq
        a[q][q] += t*apq
        a[p][q] = a[q][p] = 0.
        for i in range(4):
            if i not in (p, q):
                aip, aiq = a[i][p], a[i][q]
                a[i][p] = a[p][i] = c*aip - s*aiq
                a[i][q] = a[q][i] = s*aip + c*aiq
            vip, viq = v[i][p], v[i][q]
            v[i][p] = c*vip - s*viq
            v[i][q] = s*vip + c*viq
    raise ArithmeticError('Jacobi eigenvalue solver did not converge')


def quaternion_matrix(q):
    w, x, y, z = q
    return [[1-2*(y*y+z*z), 2*(x*y-z*w), 2*(x*z+y*w)],
            [2*(x*y+z*w), 1-2*(x*x+z*z), 2*(y*z-x*w)],
            [2*(x*z-y*w), 2*(y*z+x*w), 1-2*(x*x+y*y)]]


def estimate(source, target):
    if len(source) != len(target) or len(source) < 3:
        raise ValueError('At least three matched source/target control points are required')
    try:
        source = [tuple(float(z) for z in p) for p in source]
        target = [tuple(float(z) for z in p) for p in target]
    except (TypeError, ValueError) as exc:
        raise ValueError('Control coordinates must be numeric XYZ triples') from exc
    if any(len(p) != 3 or not all(math.isfinite(z) for z in p)
           for p in source + target):
        raise ValueError('Each control point needs three finite XYZ values')
    n = len(source)
    cx = tuple(math.fsum(p[i] for p in source)/n for i in range(3))
    cy = tuple(math.fsum(p[i] for p in target)/n for i in range(3))
    x = [tuple(p[i]-cx[i] for i in range(3)) for p in source]
    y = [tuple(p[i]-cy[i] for i in range(3)) for p in target]
    denom = math.fsum(z*z for p in x for z in p)
    if denom <= 1e-24:
        raise ValueError('Source points coincide')
    best = max(x, key=lambda p: math.fsum(z*z for z in p))
    crossmax = max(math.fsum(z*z for z in (
        best[1]*p[2]-best[2]*p[1],
        best[2]*p[0]-best[0]*p[2],
        best[0]*p[1]-best[1]*p[0])) for p in x)
    if crossmax <= 1e-20*denom*denom:
        raise ValueError('Control points are collinear or nearly collinear')
    b = [[math.fsum(x[k][i]*y[k][j] for k in range(n))
          for j in range(3)] for i in range(3)]
    bxx, bxy, bxz = b[0]
    byx, byy, byz = b[1]
    bzx, bzy, bzz = b[2]
    tr = bxx+byy+bzz
    horn = [[tr, byz-bzy, bzx-bxz, bxy-byx],
            [byz-bzy, bxx-byy-bzz, bxy+byx, bzx+bxz],
            [bzx-bxz, bxy+byx, -bxx+byy-bzz, byz+bzy],
            [bxy-byx, bzx+bxz, byz+bzy, -bxx-byy+bzz]]
    q = largest_eigenvector(horn)
    r = quaternion_matrix(q)
    rotated = [matvec(r, p) for p in x]
    scale = math.fsum(y[k][i]*rotated[k][i]
                      for k in range(n) for i in range(3))/denom
    if scale <= 0:
        raise ValueError('Estimated scale must be positive')
    rc = matvec(r, cx)
    shift = tuple(cy[i]-scale*rc[i] for i in range(3))

    def transform(p):
        try:
            p = tuple(float(z) for z in p)
        except (TypeError, ValueError) as exc:
            raise ValueError('Invalid new point') from exc
        if len(p) != 3 or not all(math.isfinite(z) for z in p):
            raise ValueError('Invalid new point')
        rp = matvec(r, p)
        return tuple(shift[i]+scale*rp[i] for i in range(3))

    calculated = [transform(p) for p in source]
    residuals = [tuple(target[k][i]-calculated[k][i]
                       for i in range(3)) for k in range(n)]
    ss = math.fsum(z*z for p in residuals for z in p)
    rmse = math.sqrt(ss/(3*n))
    sigma0 = math.sqrt(ss/(3*n-7))
    ry = math.asin(max(-1., min(1., -r[2][0])))
    if abs(math.cos(ry)) > 1e-12:
        rx = math.atan2(r[2][1], r[2][2])
        rz = math.atan2(r[1][0], r[0][0])
    else:
        rx = 0.
        rz = math.atan2(-r[0][1], r[1][1])
    angles = tuple(math.degrees(a)*3600 for a in (rx, ry, rz))
    return {'translation': shift, 'scale': scale, 'angles': angles,
            'calculated': calculated, 'residuals': residuals,
            'rmse': rmse, 'sigma0': sigma0, 'transform': transform,
            'rotation': r}


BASE_SOURCE = [(1000,2000,3000), (1500,2500,3200),
               (2000,1800,3500), (2500,2300,2800),
               (1800,3000,4000)]
ERRORS = [(0.003,-0.002,0.001), (-0.002,0.004,-0.003),
          (0.001,-0.003,0.002), (0.004,0.001,-0.002),
          (-0.003,0.002,0.004)]


def scenario_data(scenario):
    src = BASE_SOURCE[:]
    target = make_target(src)
    new = [(1750,2200,3300)]
    if scenario == 2:
        target = [tuple(a+b for a,b in zip(p,e))
                  for p,e in zip(target, ERRORS)]
    elif scenario == 3:
        new = [(1750,2200,3300), (2100,2400,3600),
               (1600,2700,3100)]
    elif scenario == 4:
        src = [(1000,2000,3000), (1100,2100,3100),
               (1200,2200,3200)]
        target = make_target(src)
    elif scenario not in (1,5):
        raise ValueError('Unknown scenario')
    return src, target, new


if __name__ == '__main__':
    source, target, new = scenario_data(2)
    result = estimate(source, target)
    print('GeoConform 3D - 7-parameter Helmert transformation')
    print('Translation:', result['translation'])
    print('Rotation (arcsec):', result['angles'])
    print('Scale (ppm):', (result['scale']-1)*1e6)
    print('RMSE (m):', result['rmse'])
    for p in new:
        print('New point:', result['transform'](p))
