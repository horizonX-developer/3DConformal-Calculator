"""GeoConform 3D Streamlit web interface (Python only)."""
import csv
import io
import math
import plotly.graph_objects as go
import streamlit as st
from geoconform3d import estimate, scenario_data, BASE_SOURCE, make_target
from computation_steps import computation_steps
from least_squares import estimate_lsa, lsa_steps
from lsa_presentation import display as display_lsa_steps

st.set_page_config(page_title='GROUP1 3DConformal Calculator (Partida Assignment rani HAHAHAAAYS) ', page_icon='🌐', layout='wide')
st.title('GROUP1 3DConformal Calculator (Partida Assignment rani HAHAHAAAYS)')
st.caption('GROUP 1 | Interactive surveying computation laboratory')
st.caption('3D Conformal Coordinate Transformation | 7-Parameter Helmert | Python')
st.markdown('**Model:** Target = Translation + Scale × Rotation × Source')
st.caption('Active rotation: R = Rz @ Ry @ Rx; source and target must use compatible 3D Cartesian coordinates.')

SCENARIOS = {
    '1 · Reference-system transformation': 1,
    '2 · Simulated measurement errors': 2,
    '3 · Additional survey points': 3,
    '4 · Collinear control points (error test)': 4,
    '5 · Verify all seven parameters': 5,
    'Custom · Enter your own coordinates': 0,
}
method = st.radio('Computation method', ['Least Squares Adjustment (classroom method)', 'Horn quaternion (comparison)'], horizontal=True)
choice = st.selectbox('Choose computation scenario', list(SCENARIOS), index=1)
if SCENARIOS[choice] == 2:
    st.info('Scenario 2 adds synthetic millimetre-level errors. Estimated parameters are not expected to equal the exact reference values.')
elif SCENARIOS[choice] == 4:
    st.warning('Scenario 4 intentionally uses collinear points. A geometry error is the correct result.')
elif SCENARIOS[choice] in (1, 3, 5):
    st.info('Exact synthetic reference: Tx=10 m, Ty=-5 m, Tz=8 m, Rx=1.5 arcsec, Ry=-2 arcsec, Rz=0.8 arcsec, scale=2.5 ppm.')
scenario = SCENARIOS[choice]
if scenario:
    source, target, new_points = scenario_data(scenario)
else:
    source = BASE_SOURCE[:]
    target = make_target(source)
    new_points = [(1750, 2200, 3300)]


def point_rows(source, target):
    return [dict(Point=f'P{i:02d}', Source_X=float(a[0]),
                 Source_Y=float(a[1]), Source_Z=float(a[2]),
                 Target_X=float(b[0]), Target_Y=float(b[1]),
                 Target_Z=float(b[2]))
            for i, (a,b) in enumerate(zip(source,target), 1)]

st.subheader('1. Matched control-point coordinates')
st.info('All coordinates are in metres. Each row must pair the same physical point in both systems.')
rows = st.data_editor(point_rows(source, target), num_rows='dynamic',
                      hide_index=True, use_container_width=True,
                      key=f'controls_{scenario}',
                      column_config={'Point': st.column_config.TextColumn('Point ID')})

st.subheader('2. Additional points to transform')
new_rows = st.data_editor(
    [dict(Point=f'N{i:02d}', X=float(p[0]), Y=float(p[1]), Z=float(p[2]))
     for i,p in enumerate(new_points, 1)],
    num_rows='dynamic', hide_index=True, use_container_width=True,
    key=f'new_{scenario}')


def parse_controls(data):
    if len(data) < 3:
        raise ValueError('Enter at least three common control points.')
    ids = [str(row['Point']).strip() for row in data]
    if any(not p for p in ids) or len(set(ids)) != len(ids):
        raise ValueError('Each control point needs a unique, nonempty Point ID.')
    src = [tuple(float(row[k]) for k in ('Source_X','Source_Y','Source_Z'))
           for row in data]
    tgt = [tuple(float(row[k]) for k in ('Target_X','Target_Y','Target_Z'))
           for row in data]
    return ids, src, tgt


def csv_text(headers, rows):
    buffer = io.StringIO()
    writer = csv.writer(buffer)
    writer.writerow(headers)
    writer.writerows(rows)
    return buffer.getvalue()

if st.button('Compute 3D Conformal Transformation', type='primary',
             use_container_width=True):
    try:
        ids, src, tgt = parse_controls(rows)
        result = estimate_lsa(src, tgt) if method.startswith('Least Squares') else estimate(src, tgt)
        st.success('Transformation computed successfully.')
        if scenario == 2:
            st.caption('The fitted parameters differ from the reference because simulated observation errors were added.')
        st.subheader('3. Estimated seven parameters')
        names = ['Tx (m)', 'Ty (m)', 'Tz (m)',
                 'Rx (arcsec)', 'Ry (arcsec)', 'Rz (arcsec)',
                 'Scale correction (ppm)']
        values = [*result['translation'], *result['angles'],
                  (result['scale']-1)*1e6]
        parameter_rows = [{'Parameter': name, 'Value': f'{value:.9f}'}
                          for name, value in zip(names, values)]
        st.dataframe(parameter_rows, hide_index=True, use_container_width=True)
        col1, col2, col3 = st.columns(3)
        col1.metric('Control points', len(src))
        col2.metric('Component RMSE', f"{result['rmse']:.6f} m")
        col3.metric('Unit-weight σ₀', f"{result['sigma0']:.6f} m")

        st.subheader('4. Transformation equation and rotation matrix')
        st.latex(r'\mathbf{X}_{target}=\mathbf{T}+(1+s\cdot10^{-6})\mathbf{R}\mathbf{X}_{source}')
        st.caption('R = Rz × Ry × Rx (active, column-vector convention). The matrix below is the estimated full rotation matrix.')
        st.dataframe([{'Row': f'R{i+1}', **{f'Column {j+1}': f'{result["rotation"][i][j]:.12f}' for j in range(3)}} for i in range(3)], hide_index=True, use_container_width=True)
        st.subheader('5. Computed control points and residuals')
        control_results = []
        for point_id, computed, residual in zip(
                ids, result['calculated'], result['residuals']):
            control_results.append({
                'Point': point_id,
                'Computed_X': computed[0], 'Computed_Y': computed[1],
                'Computed_Z': computed[2],
                'Residual_X': residual[0], 'Residual_Y': residual[1],
                'Residual_Z': residual[2]})
        st.dataframe(control_results, hide_index=True,
                     use_container_width=True)
        st.caption('Residual = target coordinate − computed coordinate.')

        st.subheader('6. Residual diagnostics')
        residual_norms = [math.sqrt(sum(v*v for v in r)) for r in result['residuals']]
        fig_res = go.Figure()
        for axis, idx in [('X',0),('Y',1),('Z',2)]:
            fig_res.add_bar(name=f'{axis} residual (mm)', x=ids, y=[r[idx]*1000 for r in result['residuals']])
        fig_res.update_layout(barmode='group', xaxis_title='Control point', yaxis_title='Residual (mm)', title='Component residuals by point')
        st.plotly_chart(fig_res, use_container_width=True)
        st.dataframe([{'Point': pid, 'Residual magnitude (mm)': round(norm*1000, 4)} for pid,norm in zip(ids,residual_norms)], hide_index=True, use_container_width=True)
        st.caption('Large residuals may indicate mismatched points or observation errors; low residuals alone do not establish absolute accuracy.')

        st.subheader('7. Interactive 3D coordinate comparison')
        fig3d = go.Figure()
        def add_xyz(points, label, color):
            fig3d.add_trace(go.Scatter3d(x=[p[0] for p in points], y=[p[1] for p in points], z=[p[2] for p in points], mode='markers+text', text=ids, textposition='top center', name=label, marker=dict(size=5, color=color)))
        add_xyz(src, 'Source XYZ', '#2582cc')
        add_xyz(tgt, 'Target XYZ', '#e89938')
        add_xyz(result['calculated'], 'Computed XYZ', '#39a27e')
        fig3d.update_layout(scene=dict(xaxis_title='X (m)', yaxis_title='Y (m)', zaxis_title='Z (m)', aspectmode='data'), margin=dict(l=0,r=0,b=0,t=25), height=520)
        st.plotly_chart(fig3d, use_container_width=True)
        st.caption('Rotate or zoom the graph. For small datum shifts relative to large XYZ values, points may appear nearly coincident.')

        st.subheader('8. Transformed additional points')
        transformed_rows = []
        additional_inputs = []
        for row in new_rows:
            point_id = str(row.get('Point') or '').strip()
            values_new = [row.get(k) for k in ('X','Y','Z')]
            blank = lambda v: v is None or (isinstance(v, str) and not v.strip()) or (isinstance(v, float) and math.isnan(v))
            if not point_id and all(blank(v) for v in values_new):
                continue  # Ignore a completely empty extra row.
            if not point_id or any(blank(v) for v in values_new):
                raise ValueError('Complete the Point ID and all XYZ values, or leave the entire extra row blank.')
            xyz = tuple(float(v) for v in values_new)
            transformed = result['transform'](xyz)
            additional_inputs.append((point_id, xyz))
            transformed_rows.append({
                'Point': row['Point'],
                'X_prime': transformed[0],
                'Y_prime': transformed[1],
                'Z_prime': transformed[2]})
        st.dataframe(transformed_rows, hide_index=True,
                     use_container_width=True)

        detailed_working = (lsa_steps(result) if 'lsa' in result else computation_steps(src, tgt, result, ids, additional_inputs))
        if 'lsa' in result and additional_inputs:
            detailed_working += '\n\nADDITIONAL POINT TRANSFORMATIONS\n'
            for pid, xyz in additional_inputs:
                rotated = tuple(sum(result['rotation'][i][j]*xyz[j] for j in range(3)) for i in range(3))
                scaled = tuple(result['scale']*v for v in rotated)
                final = result['transform'](xyz)
                detailed_working += (f'\n{pid}: source XYZ = {xyz}\n'
                                     f'  R × XYZ = {rotated}\n'
                                     f'  k × (R × XYZ) = {scaled}\n'
                                     f'  + T {result["translation"]} = target XYZ {final}\n')
        st.subheader('9. Step-by-step numerical solution')
        st.caption('The working below uses the actual coordinates and fitted parameters. LSA displays A, P, N, U, and parameter corrections; Horn displays its quaternion procedure.')
        if 'lsa' in result:
            display_lsa_steps(st, result, ids, src, tgt, additional_inputs)
            with st.expander('Advanced: original complete numerical log (raw text)', expanded=False):
                st.code(detailed_working, language=None)
        else:
            with st.expander('Show all computation steps', expanded=True):
                st.code(detailed_working, language=None)
        st.download_button('Download detailed solution (.txt)', detailed_working,
                           'GROUP1_Detailed_Solution.txt', mime='text/plain')

        report = [
            'GROUP1 3DCONFORMAL CALCULATOR | 7-PARAMETER HELMERT TRANSFORMATION',
            'Active rotation convention: R = Rz @ Ry @ Rx',
            'Method: '+method,
            'Coordinates in metres; angles in arcseconds',
            '', 'ESTIMATED PARAMETERS',
            *[f'{name}: {value:.9f}' for name,value in zip(names,values)],
            '', f"Component RMSE: {result['rmse']:.9f} m",
            f"Unit-weight sigma0: {result['sigma0']:.9f} m",
            '', 'CONTROL POINTS (computed XYZ; target - computed residuals)',
        ]
        for row in control_results:
            report.append(', '.join(f'{k}={v:.9f}' if isinstance(v,float)
                                    else f'{k}={v}' for k,v in row.items()))
        report += ['', 'ADDITIONAL POINTS']
        for row in transformed_rows:
            report.append(', '.join(f'{k}={v:.9f}' if isinstance(v,float)
                                    else f'{k}={v}' for k,v in row.items()))
        report += ['', 'Demonstration scenarios use synthetic coordinates.'
                   if scenario else 'User-supplied coordinates; verify datum and units.']
        report += ['', 'ROTATION MATRIX']
        report.extend(' '.join(f'{v:.12f}' for v in row) for row in result['rotation'])
        report += ['', 'RESIDUAL MAGNITUDES (mm)']
        report.extend(f'{pid}: {norm*1000:.6f}' for pid,norm in zip(ids,residual_norms))
        report += ['', '', detailed_working]
        st.subheader('10. Export results')
        left, right = st.columns(2)
        left.download_button('Download computation report (.txt)',
                             '\n'.join(report), 'GROUP1_Report.txt',
                             mime='text/plain', use_container_width=True)
        right.download_button('Download transformed points (.csv)',
                              csv_text(['Point','X_prime','Y_prime','Z_prime'],
                                       [[r[k] for k in
                                         ('Point','X_prime','Y_prime','Z_prime')]
                                        for r in transformed_rows]),
                              'GROUP1_Transformed.csv',
                              mime='text/csv', use_container_width=True)
        if scenario == 4:
            st.warning('This scenario should normally be rejected: check the control geometry.')
    except (ValueError, ArithmeticError, TypeError, KeyError, OverflowError) as exc:
        st.error(f'Cannot compute transformation: {exc}')

st.divider()
st.caption('Educational computation tool. The example coordinates are synthetic, not measured survey control. '
           'A low residual does not by itself establish geodetic datum compatibility or field accuracy.')
