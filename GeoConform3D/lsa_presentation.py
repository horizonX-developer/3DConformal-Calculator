"""Readable classroom presentation of the existing LSA computation (no solver changes)."""
import math

PARAMETERS = ['Tx (m)', 'Ty (m)', 'Tz (m)', 'Rx (arcsec)', 'Ry (arcsec)', 'Rz (arcsec)', 'Scale (ppm)']
SHORT = ['Tx', 'Ty', 'Tz', 'Rx', 'Ry', 'Rz', 'Scale']


def fmt(x):
    x = float(x)
    if abs(x) < 5e-11:
        return '≈ 0' if x else '0'
    if abs(x) < 1e-5 or abs(x) >= 1e8:
        return f'{x:.5e}'
    return f'{x:,.8f}'.rstrip('0').rstrip('.')


def matrix_table(st, matrix, columns, row_labels=None):
    data = []
    for i, row in enumerate(matrix):
        entry = {'Row': row_labels[i] if row_labels else str(i + 1)}
        entry.update({col: fmt(v) for col, v in zip(columns, row)})
        data.append(entry)
    st.dataframe(data, hide_index=True, use_container_width=True)


def display(st, result, ids, source, target, additional_inputs):
    """Show 12 explanatory steps from actual output of estimate_lsa."""
    d = result['lsa']
    st.caption('The following tables use actual computed values. Rounded values are for display only; calculations retain full precision. The sign convention is observed target minus computed target.')
    with st.expander('Step 1 — Given control-point coordinates', expanded=False):
        st.write('Each source XYZ point must correspond to the same physical point in the target system. Units: metres.')
        matrix_table(st, [list(s)+list(t) for s,t in zip(source,target)], ['Source X','Source Y','Source Z','Target X','Target Y','Target Z'], ids)
    with st.expander('Step 2 — Initial estimates of the seven unknowns', expanded=False):
        st.write('Initial translations are the differences between target and source centroids; initial rotations and scale correction are zero.')
        matrix_table(st, [[v] for v in d['iterations'][0]['parameters_before']], ['Initial value'], PARAMETERS)
    with st.expander('Step 3 — Observation equations and misclosures', expanded=False):
        st.latex(r'\mathbf{Y}_{calc}=\mathbf{T}+(1+s\,10^{-6})\,R_zR_yR_x\mathbf{X}')
        st.latex(r'\mathbf{w}=\mathbf{Y}_{observed}-\mathbf{Y}_{calculated}')
        st.write('For each control point, the X, Y and Z coordinate equations contribute three rows to the adjustment. The model is nonlinear, so we repeat the linearized solution until corrections become negligible.')
        first_w = d['iterations'][0]['misclosures']
        matrix_table(st, [first_w[3*i:3*i+3] for i in range(len(ids))], ['wX (m)','wY (m)','wZ (m)'], ids)
    with st.expander('Step 4 — Design matrix A', expanded=False):
        st.latex(r'A_{ij}=\frac{\partial Y_{calc,i}}{\partial u_j}')
        st.write('The seven columns correspond to translations, rotations and scale correction. The displayed A is evaluated at the final adjusted parameters; the program recomputes it during iterations. Rotation and scale derivatives are evaluated numerically.')
        labels = [f'{pid} {axis}' for pid in ids for axis in 'XYZ']
        matrix_table(st, d['A'], SHORT, labels)
    with st.expander('Step 5 — Weight matrix P', expanded=False):
        st.latex(r'P=I_{3n}')
        st.write(f'Here n = {len(ids)} control points, so P is a {3*len(ids)} × {3*len(ids)} identity matrix. All coordinate components receive equal unit weight. This is an assumption of the example, not a claim that field observations always have equal precision.')
    with st.expander('Step 6 — Normal matrix N = AᵀPA', expanded=False):
        st.latex(r'N=A^TPA')
        st.write('The final 7 × 7 normal matrix is shown below. Values may differ greatly in magnitude because translation, angle and scale parameters use different units.')
        matrix_table(st, d['N'], SHORT, SHORT)
    with st.expander('Step 7 — Right-hand vector U = AᵀPw', expanded=False):
        st.latex(r'U=A^TPw')
        st.write('This vector uses the final residual misclosures. Because the adjustment has converged, its components should be near zero. The larger U values used to compute earlier corrections appear in the iteration details.')
        matrix_table(st, [[x] for x in d['U']], ['U (final)'], SHORT)
    with st.expander('Step 8 — Solve for parameter corrections', expanded=False):
        st.latex(r'N\,\Delta=U\qquad u_{new}=u_{old}+\Delta')
        st.write('For numerical stability, the solver scales the columns of A internally, solves the equivalent preconditioned normal equations, and converts corrections back into the original parameter units. It does not explicitly calculate the inverse of N.')
        first = d['iterations'][0]
        matrix_table(st, [[p,c,p+c] for p,c in zip(first['parameters_before'],first['correction'])], ['Initial estimate','Correction Δ','Updated estimate'], PARAMETERS)
    with st.expander('Step 9 — Iteration summary', expanded=True):
        st.write('Each iteration refines the seven unknowns. Select an iteration below to inspect its actual corrections and preconditioned normal equations.')
        summary = []
        for h in d['iterations']:
            summary.append({'Iteration':h['iteration'], 'Max |correction| (mixed units)':fmt(max(abs(x) for x in h['correction'])), 'Tx after (m)':fmt(h['parameters_before'][0]+h['correction'][0]), 'Ty after (m)':fmt(h['parameters_before'][1]+h['correction'][1]), 'Tz after (m)':fmt(h['parameters_before'][2]+h['correction'][2])})
        st.dataframe(summary, hide_index=True, use_container_width=True)
        st.caption('The maximum correction mixes parameter units, so it is only a compact convergence indicator—not a physical distance or accuracy measure.')
        selected = st.selectbox('Inspect iteration', [h['iteration'] for h in d['iterations']], key='lsa_iteration_view')
        h = d['iterations'][selected-1]
        matrix_table(st, [[x] for x in h['correction']], ['Parameter correction Δ'], PARAMETERS)
        with st.expander('Advanced: preconditioned N* and U* for this iteration', expanded=False):
            st.write('These scaled matrices are used by the numerical solver. They are equivalent to the unscaled normal equations but are not the same displayed values as N and U.')
            matrix_table(st, h['normal_scaled'], SHORT, SHORT)
            matrix_table(st, [[x] for x in h['rhs_scaled']], ['U*'], SHORT)
    with st.expander('Step 10 — Final adjusted seven parameters', expanded=True):
        matrix_table(st, [[x] for x in d['parameters']], ['Adjusted value'], PARAMETERS)
        st.write('The displayed scale parameter is the correction in ppm; the multiplicative scale factor is k = 1 + scale(ppm) × 10⁻⁶.')
    with st.expander('Step 11 — Residuals and adjustment statistics', expanded=False):
        st.latex(r'v=Y_{observed}-Y_{calculated}')
        st.latex(r'RMSE=\sqrt{\frac{\sum v^2}{3n}}\qquad\hat\sigma_0=\sqrt{\frac{\sum v^2}{3n-7}}')
        matrix_table(st, result['residuals'], ['vX (m)','vY (m)','vZ (m)'], ids)
        st.write(f"Sum of squared residuals (SSE): {fmt(d['SSE'])} m²; degrees of freedom: {d['dof']}; component RMSE: {fmt(result['rmse'])} m; unit-weight sigma-zero: {fmt(result['sigma0'])} m.")
    with st.expander('Step 12 — Transform additional points', expanded=False):
        st.latex(r'\mathbf{X}_{target}=\mathbf{T}+k(R\mathbf{X}_{source})')
        if not additional_inputs:
            st.write('No additional points were entered.')
        for pid, xyz in additional_inputs:
            rotated = [math.fsum(result['rotation'][i][j]*xyz[j] for j in range(3)) for i in range(3)]
            scaled = [result['scale']*x for x in rotated]
            transformed = result['transform'](xyz)
            st.write(f'**Point {pid}**')
            matrix_table(st, [xyz,rotated,scaled,result['translation'],transformed], ['X','Y','Z'], ['Input XYZ','R × XYZ','k × (R × XYZ)','Translation T','Final target XYZ'])
            for axis, t, sc, val in zip('XYZ',result['translation'],scaled,transformed):
                st.code(f"{axis}' = {fmt(t)} + ({fmt(sc)}) = {fmt(val)} m", language=None)
