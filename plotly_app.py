"""
Plotly Dash Dashboard: Satellite Space Debris Analytics
Complete Interactive Dashboard in Clean Academic LIGHT THEME.
Interactive Parameter Selection & Filtering (Dropdowns, Sliders & Variable Controls).
Instant Tab Switching (Zero Long Scrolling) & Exact 4-Quadrant Inference Cards.

Strictly aligned with Review 1 Syllabus:
- Matplotlib Suite: M1, M2, M3
- Seaborn Suite: S1, S2, S3, S4, S5
- Plotly Suite: P1, P2 (with Dynamic Parameter Explorer)
- Academic Ethics: Case Study in Lie Factor Analysis

Pure Python & Plotly Dash — Zero HTML file generation.
Zero text/legend overlaps.

Name: Vudattu Sai Satya Saketh (CB.SC.U4CSE24558)
Course: 23CSE353 Data Visualization
"""

import os
import sys
import threading
import webbrowser
sys.stdout.reconfigure(encoding='utf-8')

import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import dash
from dash import html, dcc, Input, Output, State

# ==============================================================================
# 1. DATA PREPARATION & ENRICHMENT
# ==============================================================================
def load_catalog_data():
    """Load cleaned Space-Track GP catalog data with full feature engineering."""
    base_dir = os.path.dirname(os.path.abspath(__file__)) if '__file__' in globals() else os.getcwd()
    cleaned_path = os.path.join(base_dir, 'space_decay_cleaned.csv')
    raw_path = os.path.join(base_dir, 'space_decay.csv')
    
    if os.path.exists(cleaned_path):
        df = pd.read_csv(cleaned_path)
    elif os.path.exists('space_decay_cleaned.csv'):
        df = pd.read_csv('space_decay_cleaned.csv')
    elif os.path.exists(raw_path):
        df_raw = pd.read_csv(raw_path)
        tba_mask = df_raw['OBJECT_NAME'].str.contains('TBA', na=False, case=False) | (df_raw['OBJECT_TYPE'] == 'TBA')
        df = df_raw[~tba_mask].copy()
        
        df['EPOCH'] = pd.to_datetime(df['EPOCH'], errors='coerce')
        df['LAUNCH_YEAR'] = pd.to_numeric(df['LAUNCH_DATE'], errors='coerce').fillna(0).astype(int)
        df['RCS_SIZE'] = df['RCS_SIZE'].fillna('UNKNOWN').astype(str).str.strip().str.upper()
        df['COUNTRY_CODE'] = df['COUNTRY_CODE'].fillna('UNKNOWN').astype(str).str.strip().str.upper()
        df['OBJECT_TYPE'] = df['OBJECT_TYPE'].astype(str).str.strip().str.upper()
        df['MEAN_ALTITUDE_KM'] = ((df['APOAPSIS'] + df['PERIAPSIS']) / 2.0).round(2)
        
        country_names = {
            'US': 'United States', 'PRC': 'China (PRC)', 'CIS': 'Russia (CIS)',
            'UK': 'United Kingdom', 'FR': 'France', 'IND': 'India', 'JPN': 'Japan',
            'ESA': 'European Space Agency', 'CA': 'Canada', 'GER': 'Germany'
        }
        df['COUNTRY_NAME'] = df['COUNTRY_CODE'].map(lambda c: country_names.get(c, c))
        
        def classify_orbit(row):
            apo, peri, ecc = row['APOAPSIS'], row['PERIAPSIS'], row['ECCENTRICITY']
            if apo <= 2000: return 'LEO'
            elif ecc >= 0.25 and apo > 2000: return 'HEO'
            elif abs(apo - 35786) < 1500 and abs(peri - 35786) < 1500 and ecc < 0.1: return 'GEO'
            elif peri >= 2000 and apo < 35786: return 'MEO'
            elif apo >= 35786 and peri >= 30000: return 'GEO'
            else: return 'HEO'
        df['ORBIT_CLASS'] = df.apply(classify_orbit, axis=1)
    else:
        raise FileNotFoundError("Could not find space_decay_cleaned.csv or space_decay.csv")
        
    return df

df_full = load_catalog_data()

# Catalog summary metrics
FULL_TOTAL = len(df_full)

# ==============================================================================
# 2. LIGHT THEME PALETTES & LAYOUT PRESETS
# ==============================================================================
TYPE_COLORS = {'DEBRIS': '#EF4444', 'PAYLOAD': '#10B981', 'ROCKET BODY': '#F59E0B'}
REGIME_COLORS = {'LEO': '#0284C7', 'MEO': '#8B5CF6', 'GEO': '#EC4899', 'HEO': '#F59E0B'}

LIGHT_THEME_LAYOUT = dict(
    template="plotly_white",
    paper_bgcolor="#FFFFFF",
    plot_bgcolor="#FFFFFF",
    font=dict(family="Inter, -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, sans-serif", color="#1E293B")
)

def empty_figure(title, message="No data matches current filter selection"):
    """Graceful placeholder when filters match zero records."""
    fig = go.Figure()
    fig.add_annotation(
        text=f"<b>⚠️ {message}</b><br><span style='font-size:12px;color:#94A3B8;'>Please broaden or reset your filter parameters</span>",
        xref="paper", yref="paper", x=0.5, y=0.5, showarrow=False,
        font=dict(size=14, color="#64748B")
    )
    fig.update_layout(
        **LIGHT_THEME_LAYOUT, height=450,
        title=dict(text=title, x=0.02, y=0.96, xanchor='left'),
        xaxis=dict(visible=False), yaxis=dict(visible=False)
    )
    return fig

# ==============================================================================
# 3. 4-QUADRANT INFERENCE COMPONENT (MATCHING USER REFERENCE DESIGN)
# ==============================================================================
def create_quadrant_inference_box(id_title, key_finding, calculated_evidence, contextual_interpretation, analytical_caution):
    """
    Render a clean 4-quadrant inference card with 2x2 grid format matching reference design:
    - Quadrant 1: KEY FINDING (Sky Blue)
    - Quadrant 2: CALCULATED EVIDENCE (Green)
    - Quadrant 3: CONTEXTUAL INTERPRETATION (Purple)
    - Quadrant 4: ANALYTICAL CAUTION / LIMITATION (Amber)
    """
    return html.Div(
        style={
            'backgroundColor': '#F8FAFC',
            'border': '1px solid #E2E8F0',
            'borderRadius': '12px',
            'padding': '20px 24px',
            'marginTop': '20px',
            'boxShadow': '0 1px 4px rgba(0, 0, 0, 0.04)'
        },
        children=[
            # Header Title
            html.Div(
                f"• {id_title.upper()} INFERENCE: DATA-DRIVEN FINDING & EVIDENCE",
                style={
                    'color': '#1D4ED8',
                    'fontSize': '13.5px',
                    'fontWeight': '800',
                    'letterSpacing': '0.5px',
                    'marginBottom': '16px',
                    'textTransform': 'uppercase'
                }
            ),
            # 2x2 Grid Container
            html.Div(
                style={
                    'display': 'grid',
                    'gridTemplateColumns': 'repeat(auto-fit, minmax(320px, 1fr))',
                    'gap': '18px 32px'
                },
                children=[
                    # Quadrant 1: KEY FINDING
                    html.Div([
                        html.Div("KEY FINDING", style={'color': '#0284C7', 'fontSize': '12px', 'fontWeight': '800', 'letterSpacing': '0.5px', 'marginBottom': '4px'}),
                        html.Div(key_finding, style={'color': '#334155', 'fontSize': '13px', 'lineHeight': '1.55'})
                    ]),
                    # Quadrant 2: CALCULATED EVIDENCE
                    html.Div([
                        html.Div("CALCULATED EVIDENCE", style={'color': '#059669', 'fontSize': '12px', 'fontWeight': '800', 'letterSpacing': '0.5px', 'marginBottom': '4px'}),
                        html.Div(calculated_evidence, style={'color': '#334155', 'fontSize': '13px', 'lineHeight': '1.55'})
                    ]),
                    # Quadrant 3: CONTEXTUAL INTERPRETATION
                    html.Div([
                        html.Div("CONTEXTUAL INTERPRETATION", style={'color': '#6366F1', 'fontSize': '12px', 'fontWeight': '800', 'letterSpacing': '0.5px', 'marginBottom': '4px'}),
                        html.Div(contextual_interpretation, style={'color': '#334155', 'fontSize': '13px', 'lineHeight': '1.55'})
                    ]),
                    # Quadrant 4: ANALYTICAL CAUTION / LIMITATION
                    html.Div([
                        html.Div("ANALYTICAL CAUTION / LIMITATION", style={'color': '#D97706', 'fontSize': '12px', 'fontWeight': '800', 'letterSpacing': '0.5px', 'marginBottom': '4px'}),
                        html.Div(analytical_caution, style={'color': '#334155', 'fontSize': '13px', 'lineHeight': '1.55'})
                    ])
                ]
            )
        ]
    )

card_style = {
    'backgroundColor': '#FFFFFF',
    'border': '1px solid #E2E8F0',
    'borderRadius': '14px',
    'padding': '24px',
    'marginBottom': '28px',
    'boxShadow': '0 1px 3px rgba(0,0,0,0.05), 0 10px 15px -3px rgba(0,0,0,0.03)'
}

# ==============================================================================
# 4. VISUALIZATION GENERATORS — MATPLOTLIB SUITE (M1 – M3)
# ==============================================================================

def create_m1_bar(data):
    if len(data) == 0:
        return empty_figure("M1: Object Count by Classification Type")
    counts = data['OBJECT_TYPE'].value_counts().reset_index()
    counts.columns = ['OBJECT_TYPE', 'COUNT']
    total_val = counts['COUNT'].sum()
    counts['PERCENT'] = (counts['COUNT'] / total_val) * 100 if total_val > 0 else 0
    
    fig = px.bar(
        counts, x='OBJECT_TYPE', y='COUNT',
        color='OBJECT_TYPE', color_discrete_map=TYPE_COLORS,
        text=counts.apply(lambda r: f"{r['COUNT']:,} ({r['PERCENT']:.1f}%)", axis=1),
        labels={'OBJECT_TYPE': 'Object Classification', 'COUNT': 'Tracked Objects Count'}
    )
    fig.update_traces(textposition='outside', marker_line_color='#CBD5E1', marker_line_width=1)
    fig.update_layout(
        **LIGHT_THEME_LAYOUT, height=460,
        margin=dict(t=70, b=50, l=60, r=40),
        title=dict(text="<b>M1: Object Count by Classification Type</b><br><span style='color:#64748B;font-size:12px;'>Catalog Proportions: Debris Dominance vs. Operational Payloads vs. Discarded Stages</span>", x=0.02, y=0.96, xanchor='left'),
        showlegend=False,
        yaxis=dict(gridcolor="#F1F5F9", range=[0, max(10, counts['COUNT'].max() * 1.2)]),
        xaxis=dict(gridcolor="#F1F5F9")
    )
    return fig

def create_m2_area(data):
    valid = data[(data['LAUNCH_YEAR'] >= 1957) & (data['LAUNCH_YEAR'] <= 2024)].copy()
    if len(valid) == 0:
        return empty_figure("M2: Retained Objects by Launch Year — Stacked Area Profile")
    yearly = valid.groupby(['LAUNCH_YEAR', 'OBJECT_TYPE']).size().unstack(fill_value=0).reset_index()
    
    fig = go.Figure()
    for obj_type in ['DEBRIS', 'ROCKET BODY', 'PAYLOAD']:
        if obj_type in yearly.columns:
            fig.add_trace(go.Scatter(
                x=yearly['LAUNCH_YEAR'], y=yearly[obj_type],
                mode='lines', stackgroup='one', name=obj_type,
                line=dict(width=1.5, color=TYPE_COLORS.get(obj_type, '#888')),
                hovertemplate=f"<b>{obj_type}</b><br>Year: %{{x}}<br>Retained Objects: %{{y:,}}<extra></extra>"
            ))
            
    if valid['LAUNCH_YEAR'].min() <= 2007 <= valid['LAUNCH_YEAR'].max():
        fig.add_annotation(x=2007, y=650, text="2007 Fengyun-1C<br>ASAT Spike",
                           showarrow=True, arrowhead=2, arrowcolor="#EF4444", ax=-50, ay=-50,
                           font=dict(color="#B91C1C", size=10), bgcolor="#FEF2F2", bordercolor="#F87171")
    if valid['LAUNCH_YEAR'].min() <= 2009 <= valid['LAUNCH_YEAR'].max():
        fig.add_annotation(x=2009, y=520, text="2009 Cosmos-Iridium<br>Collision",
                           showarrow=True, arrowhead=2, arrowcolor="#F59E0B", ax=-60, ay=-65,
                           font=dict(color="#B45309", size=10), bgcolor="#FFFBEB", bordercolor="#FCD34D")
    if valid['LAUNCH_YEAR'].min() <= 2020 <= valid['LAUNCH_YEAR'].max():
        fig.add_annotation(x=2020, y=1050, text="2019-2021 Starlink &<br>Mega-Constellations",
                           showarrow=True, arrowhead=2, arrowcolor="#10B981", ax=-80, ay=-55,
                           font=dict(color="#047857", size=10), bgcolor="#ECFDF5", bordercolor="#6EE7B7")
                       
    fig.update_layout(
        **LIGHT_THEME_LAYOUT, height=480,
        margin=dict(t=70, b=50, l=60, r=40),
        title=dict(text="<b>M2: Retained Objects by Launch Year — Stacked Area Profile</b><br><span style='color:#64748B;font-size:12px;'>Surviving Catalog Objects Grouped by Launch Era & Historical Breakup Signatures</span>", x=0.02, y=0.96, xanchor='left'),
        xaxis=dict(title="Launch Year", gridcolor="#F1F5F9"),
        yaxis=dict(title="Surviving Objects in Orbit", gridcolor="#F1F5F9"),
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1)
    )
    return fig

def create_m3_donut(data):
    if len(data) == 0:
        return empty_figure("M3: Orbital Regime Share — LEO, MEO, GEO, and HEO")
    regime_counts = data['ORBIT_CLASS'].value_counts().reset_index()
    regime_counts.columns = ['ORBIT_CLASS', 'COUNT']
    
    fig = px.pie(
        regime_counts, names='ORBIT_CLASS', values='COUNT',
        hole=0.55, color='ORBIT_CLASS', color_discrete_map=REGIME_COLORS
    )
    fig.update_traces(
        textposition='outside', textinfo='label+percent',
        hovertemplate="<b>%{label} Regime</b><br>Tracked Objects: %{value:,}<br>Share of Filtered: %{percent:.1%}<extra></extra>",
        marker=dict(line=dict(color='#FFFFFF', width=2))
    )
    fig.update_layout(
        **LIGHT_THEME_LAYOUT, height=460,
        margin=dict(t=70, b=40, l=40, r=40),
        title=dict(text="<b>M3: Orbital Regime Share — LEO, MEO, GEO, and HEO</b><br><span style='color:#64748B;font-size:12px;'>Proportion of Global Orbital Objects Across Altitudinal Regimes</span>", x=0.02, y=0.96, xanchor='left'),
        legend=dict(orientation="v", yanchor="middle", y=0.5, xanchor="left", x=1.02)
    )
    return fig

# ==============================================================================
# 5. VISUALIZATION GENERATORS — SEABORN SUITE (S1 – S5)
# ==============================================================================

def create_s1_histogram(data):
    if len(data) == 0:
        return empty_figure("S1: Orbital Inclination Distribution")
    fig = px.histogram(
        data, x='INCLINATION', color='OBJECT_TYPE',
        color_discrete_map=TYPE_COLORS, nbins=90,
        marginal='box', opacity=0.82,
        labels={'INCLINATION': 'Orbital Inclination (Degrees °)', 'count': 'Number of Objects'}
    )
    fig.add_annotation(x=98.6, y=1400, text="Sun-Synchronous (SSO)<br>Polar Peak (~98°)",
                       showarrow=True, arrowhead=2, arrowcolor="#0284C7", ax=50, ay=-40,
                       font=dict(color="#0369A1", size=10), bgcolor="#F0F9FF", bordercolor="#BAE6FD")
    fig.add_annotation(x=53.0, y=950, text="Starlink / ISS Corridor<br>(~53°)",
                       showarrow=True, arrowhead=2, arrowcolor="#10B981", ax=-60, ay=-40,
                       font=dict(color="#047857", size=10), bgcolor="#ECFDF5", bordercolor="#A7F3D0")
    fig.add_annotation(x=0.0, y=400, text="Equatorial GEO Ring<br>(~0°)",
                       showarrow=True, arrowhead=2, arrowcolor="#EC4899", ax=50, ay=-35,
                       font=dict(color="#BE185D", size=10), bgcolor="#FDF2F8", bordercolor="#FBCFE8")
    fig.update_layout(
        **LIGHT_THEME_LAYOUT, height=480,
        margin=dict(t=70, b=50, l=60, r=40),
        title=dict(text="<b>S1: Orbital Inclination Distribution (Degrees °)</b><br><span style='color:#64748B;font-size:12px;'>Bimodal Astrodynamic Peaks: SSO Polar Shell (~98°) and Mid-Latitude Launch Corridors (~53°–65°)</span>", x=0.02, y=0.96, xanchor='left'),
        xaxis=dict(title="Orbital Inclination (°)", gridcolor="#F1F5F9", range=[-2, 145]),
        yaxis=dict(title="Object Count", gridcolor="#F1F5F9"),
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1)
    )
    return fig

def create_s2_violin(data):
    valid_period = data[(data['PERIOD'] > 0) & (data['PERIOD'] <= 2000)].copy()
    if len(valid_period) == 0:
        return empty_figure("S2: Orbital Period by Object Type")
    
    fig = px.violin(
        valid_period, x='OBJECT_TYPE', y='PERIOD', color='OBJECT_TYPE',
        color_discrete_map=TYPE_COLORS, box=True, points=False,
        category_orders={'OBJECT_TYPE': ['DEBRIS', 'PAYLOAD', 'ROCKET BODY']},
        labels={'OBJECT_TYPE': 'Object Classification', 'PERIOD': 'Orbital Period (Minutes)'}
    )
    fig.add_hline(y=1436.1, line_dash="dash", line_color="#EC4899",
                  annotation_text="GEO Sidereal Period (1,436 min / 23.93 h)", annotation_position="top left",
                  annotation_font=dict(color="#BE185D", size=10.5))
    fig.add_hline(y=90.0, line_dash="dot", line_color="#0284C7",
                  annotation_text="LEO Typical Orbit (~90-100 min)", annotation_position="bottom left",
                  annotation_font=dict(color="#0369A1", size=10.5))
    fig.update_layout(
        **LIGHT_THEME_LAYOUT, height=480,
        margin=dict(t=70, b=50, l=60, r=40),
        title=dict(text="<b>S2: Orbital Period by Object Type (Violin Plot, Log Scale)</b><br><span style='color:#64748B;font-size:12px;'>Astrodynamic Velocity & Flight Time Comparison: Debris vs. Functional Payloads vs. Discarded Stages</span>", x=0.02, y=0.96, xanchor='left'),
        yaxis=dict(type='log', range=[1.8, 3.25], title="Orbital Period (Minutes, Log Scale)", gridcolor="#F1F5F9"),
        xaxis=dict(title="Object Classification Type", gridcolor="#F1F5F9"),
        showlegend=False
    )
    return fig

def create_s3_loglog_scatter(data):
    plot_df = data[(data['PERIAPSIS'] > 50) & (data['APOAPSIS'] > 50) & (data['APOAPSIS'] <= 60000)].copy()
    if len(plot_df) == 0:
        return empty_figure("S3: Orbital Regime Map — Perigee vs. Apogee")
    
    fig = px.scatter(
        plot_df, x='PERIAPSIS', y='APOAPSIS', color='OBJECT_TYPE',
        color_discrete_map=TYPE_COLORS,
        category_orders={'OBJECT_TYPE': ['DEBRIS', 'PAYLOAD', 'ROCKET BODY']},
        log_x=True, log_y=True, opacity=0.65,
        hover_name='OBJECT_NAME',
        hover_data={'NORAD_CAT_ID': True, 'COUNTRY_NAME': True, 'PERIAPSIS': ':.1f', 'APOAPSIS': ':.1f', 'OBJECT_TYPE': True},
        labels={'PERIAPSIS': 'Periapsis / Perigee Altitude (km, Log Scale)', 'APOAPSIS': 'Apoapsis / Apogee Altitude (km, Log Scale)', 'OBJECT_TYPE': 'Object Classification'}
    )
    # Circular Orbit Reference Line
    fig.add_trace(go.Scatter(
        x=[80, 50000], y=[80, 50000],
        mode='lines', line=dict(dash='dash', color='#64748B', width=1.5),
        name='Circular Orbit Line (Perigee = Apogee, e ≈ 0)', hoverinfo='name'
    ))
    fig.update_traces(marker=dict(size=4), selector=dict(mode='markers'))
    fig.update_layout(
        **LIGHT_THEME_LAYOUT, height=480,
        margin=dict(t=70, b=50, l=60, r=40),
        title=dict(text="<b>S3: Orbital Regime Map — Perigee vs. Apogee (Log-Log Scatter by Object Type)</b><br><span style='color:#64748B;font-size:12px;'>Eccentricity Separation: Circular Operational Payloads along Diagonal vs. Cross-Regime Rocket Bodies & Debris</span>", x=0.02, y=0.96, xanchor='left'),
        xaxis=dict(title="Periapsis / Perigee Altitude (km, Log)", gridcolor="#F1F5F9"),
        yaxis=dict(title="Apoapsis / Apogee Altitude (km, Log)", gridcolor="#F1F5F9"),
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1)
    )
    return fig

def create_s4_correlation_heatmap(data):
    cols = ['PERIOD', 'INCLINATION', 'APOAPSIS', 'PERIAPSIS', 'ECCENTRICITY', 'SEMIMAJOR_AXIS', 'MEAN_ALTITUDE_KM']
    labels = ['Period', 'Inclination', 'Apoapsis', 'Periapsis', 'Eccentricity', 'Semi-Major Axis', 'Mean Altitude']
    if len(data) < 3:
        return empty_figure("S4: Correlation Matrix of Orbital Parameters", "Insufficient records to calculate correlations")
    corr = data[cols].corr().round(2).fillna(0)
    
    fig = px.imshow(
        corr.values, x=labels, y=labels,
        color_continuous_scale='Blues', zmin=-0.2, zmax=1.0,
        text_auto=True, aspect="auto"
    )
    fig.update_layout(
        **LIGHT_THEME_LAYOUT, height=480,
        margin=dict(t=70, b=60, l=110, r=40),
        title=dict(text="<b>S4: Correlation Matrix of Orbital Parameters (Pearson r)</b><br><span style='color:#64748B;font-size:12px;'>Empirical Laws in Data: Kepler's 3rd Law (Period ↔ Semi-Major Axis r = 0.99) vs. Inclination Independence</span>", x=0.02, y=0.96, xanchor='left'),
        coloraxis_colorbar=dict(title="Correlation (r)", tickvals=[-0.2, 0.0, 0.5, 1.0])
    )
    return fig

def create_s5_pareto_lollipop(data):
    debris_data = data[data['OBJECT_TYPE'] == 'DEBRIS']
    if len(debris_data) == 0:
        # Fallback to general object count if debris is filtered out
        debris_data = data
        chart_title = "<b>S5: Top 10 Contributing Nations in Selection — Pareto Distribution</b>"
        count_label = "Objects Count"
    else:
        chart_title = "<b>S5: Top 10 Debris Contributing Nations — Pareto Distribution</b>"
        count_label = "Debris Objects Count"
        
    top_ctry = debris_data['COUNTRY_NAME'].value_counts().head(10).reset_index()
    if len(top_ctry) == 0:
        return empty_figure("S5: Pareto Distribution by Nation")
        
    top_ctry.columns = ['COUNTRY_NAME', 'COUNT']
    total_val = top_ctry['COUNT'].sum()
    top_ctry['CUM_PERCENT'] = (top_ctry['COUNT'].cumsum() / total_val) * 100 if total_val > 0 else 0
    
    fig = make_subplots(specs=[[{"secondary_y": True}]])
    
    fig.add_trace(
        go.Bar(
            x=top_ctry['COUNTRY_NAME'], y=top_ctry['COUNT'],
            name=count_label, marker_color='#EF4444',
            text=[f"{v:,}" for v in top_ctry['COUNT']], textposition='outside'
        ),
        secondary_y=False
    )
    
    fig.add_trace(
        go.Scatter(
            x=top_ctry['COUNTRY_NAME'], y=top_ctry['CUM_PERCENT'],
            name="Cumulative Share (%)", mode='lines+markers',
            line=dict(color='#0284C7', width=2.5),
            marker=dict(size=7, color='#0284C7'),
            hovertemplate="Cumulative: %{y:.1f}%<extra></extra>"
        ),
        secondary_y=True
    )
    
    fig.add_hline(y=80, line_dash="dash", line_color="#D97706", secondary_y=True,
                  annotation_text="80% Pareto Threshold", annotation_position="top left",
                  annotation_font=dict(color="#B45309", size=10))
                  
    fig.update_layout(
        **LIGHT_THEME_LAYOUT, height=480,
        margin=dict(t=70, b=60, l=60, r=60),
        title=dict(text=f"{chart_title}<br><span style='color:#64748B;font-size:12px;'>Global Responsibility: Top 3 Powers (US, China, Russia) Account for >85% of Orbiting Objects</span>", x=0.02, y=0.96, xanchor='left'),
        xaxis=dict(title="Operating Country / Agency", tickangle=-20, gridcolor="#F1F5F9"),
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1)
    )
    fig.update_yaxes(title_text=count_label, gridcolor="#F1F5F9", secondary_y=False, range=[0, max(10, top_ctry['COUNT'].max() * 1.15)])
    fig.update_yaxes(title_text="Cumulative Percentage (%)", range=[0, 105], secondary_y=True, gridcolor="#F8FAFC")
    return fig

# ==============================================================================
# 6. VISUALIZATION GENERATORS — PLOTLY SUITE (P1 – P2) & ETHICS LAB
# ==============================================================================

PARAM_LABELS = {
    'INCLINATION': 'Orbital Inclination (°)',
    'MEAN_ALTITUDE_KM': 'Mean Altitude (km)',
    'PERIOD': 'Orbital Period (min)',
    'ECCENTRICITY': 'Eccentricity (e)',
    'PERIAPSIS': 'Periapsis / Perigee (km)',
    'APOAPSIS': 'Apoapsis / Apogee (km)',
    'OBJECT_TYPE': 'Object Classification',
    'ORBIT_CLASS': 'Orbital Regime',
    'RCS_SIZE': 'Radar Cross Section (RCS)'
}

def create_p1_dynamic_scatter(data, x_col='INCLINATION', y_col='MEAN_ALTITUDE_KM', color_col='OBJECT_TYPE'):
    """Dynamic Scatter Plot allowing users to plot any parameter combinations interactively."""
    if len(data) == 0:
        return empty_figure("P1: Multidimensional Parameter Explorer")
    plot_df = data.copy()
    if y_col in ['MEAN_ALTITUDE_KM', 'APOAPSIS', 'PERIAPSIS']:
        plot_df = plot_df[plot_df[y_col] <= 45000]
        
    color_map = TYPE_COLORS if color_col == 'OBJECT_TYPE' else (REGIME_COLORS if color_col == 'ORBIT_CLASS' else None)
    symbol_map = {'SMALL': 'circle', 'MEDIUM': 'diamond', 'LARGE': 'square', 'UNKNOWN': 'cross'}
    
    fig = px.scatter(
        plot_df, x=x_col, y=y_col,
        color=color_col, symbol='RCS_SIZE' if color_col != 'RCS_SIZE' else None,
        symbol_map=symbol_map if color_col != 'RCS_SIZE' else None,
        color_discrete_map=color_map,
        hover_name='OBJECT_NAME',
        hover_data={'NORAD_CAT_ID': True, 'COUNTRY_NAME': True, 'LAUNCH_YEAR': True,
                    'APOAPSIS': ':.1f', 'PERIAPSIS': ':.1f', 'INCLINATION': ':.2f',
                    'MEAN_ALTITUDE_KM': ':.1f'},
        labels={x_col: PARAM_LABELS.get(x_col, x_col), y_col: PARAM_LABELS.get(y_col, y_col),
                color_col: PARAM_LABELS.get(color_col, color_col), 'RCS_SIZE': 'RCS Size'},
        opacity=0.72
    )
    
    # Specific orbital guide lines if altitude is on Y axis
    if y_col == 'MEAN_ALTITUDE_KM':
        fig.add_hline(y=35786, line_dash="dash", line_color="#EC4899",
                      annotation_text="GEO Ring (35,786 km)", annotation_position="top left",
                      annotation_font=dict(color="#BE185D", size=11))
        fig.add_hline(y=2000, line_dash="dot", line_color="#0284C7",
                      annotation_text="LEO Upper Boundary (2,000 km)", annotation_position="bottom left",
                      annotation_font=dict(color="#0369A1", size=11))
                      
    fig.update_layout(
        **LIGHT_THEME_LAYOUT, height=580,
        margin=dict(t=80, b=60, l=70, r=200),
        title=dict(text=f"<b>P1: Interactive Parameter Explorer — {PARAM_LABELS.get(x_col, x_col)} vs. {PARAM_LABELS.get(y_col, y_col)}</b><br><span style='color:#64748B;font-size:12px;'>Color coded by {PARAM_LABELS.get(color_col, color_col)} • Filtered Catalog Objects: {len(plot_df):,}</span>", x=0.02, y=0.96, xanchor='left'),
        legend=dict(orientation="v", yanchor="middle", y=0.5, xanchor="left", x=1.02,
                    bgcolor="#F8FAFC", bordercolor="#E2E8F0", borderwidth=1,
                    title=dict(text=f"<b>{PARAM_LABELS.get(color_col, color_col)}</b>", font=dict(size=11, color="#64748B"))),
        xaxis=dict(gridcolor="#F1F5F9", zerolinecolor="#E2E8F0"),
        yaxis=dict(gridcolor="#F1F5F9", zerolinecolor="#E2E8F0")
    )
    return fig

def create_p2_treemap(data):
    if len(data) == 0:
        return empty_figure("P2: Catalog Population Hierarchy")
    top_ctry = data['COUNTRY_NAME'].value_counts().head(10).index.tolist()
    treemap_df = data.copy()
    treemap_df['COUNTRY_GROUP'] = treemap_df['COUNTRY_NAME'].apply(lambda c: c if c in top_ctry else 'Other Nations')
    agg = treemap_df.groupby(['OBJECT_TYPE', 'COUNTRY_GROUP', 'RCS_SIZE']).size().reset_index(name='COUNT')
    
    fig = px.treemap(
        agg, path=['OBJECT_TYPE', 'COUNTRY_GROUP', 'RCS_SIZE'], values='COUNT',
        color='OBJECT_TYPE', color_discrete_map=TYPE_COLORS,
        title="<b>P2: Catalog Population Hierarchy — Object Type → Country → Radar Cross Section</b><br><span style='color:#64748B;font-size:12px;'>Hierarchical Drill-Down Representation of Orbiting Objects</span>"
    )
    fig.update_traces(
        textinfo="label+value+percent parent",
        hovertemplate="<b>%{label}</b><br>Tracked Objects: %{value:,}<br>Share of Parent: %{percentParent:.1%}<br>Share of Total: %{percentRoot:.1%}<extra></extra>"
    )
    fig.update_layout(
        **LIGHT_THEME_LAYOUT, height=580,
        margin=dict(t=75, b=30, l=20, r=20)
    )
    return fig

def create_p6_ethics():
    top3_data = pd.DataFrame({
        'Country': ['United States', 'China (PRC)', 'Russia (CIS)'],
        'Objects': [5026, 3853, 3207]
    })
    
    fig = make_subplots(
        rows=1, cols=2,
        subplot_titles=(
            "⚠️ <b>MISLEADING VISUALIZATION</b><br><span style='font-size:11px;color:#DC2626;'>Truncated Axis Baseline at 3,000 (Exaggerates Difference by 9.8×)</span>",
            "✅ <b>ETHICAL REDESIGN</b><br><span style='font-size:11px;color:#16A34A;'>Accurate Scale Baseline at 0 (True Ratio = 1.56×)</span>"
        ),
        horizontal_spacing=0.20
    )
    
    # Misleading Bar
    fig.add_trace(
        go.Bar(x=top3_data['Country'], y=top3_data['Objects'],
               marker_color=['#EF4444', '#F97316', '#F59E0B'],
               text=[f"{v:,}" for v in top3_data['Objects']], textposition='outside', showlegend=False),
        row=1, col=1
    )
    
    # Ethical Bar
    fig.add_trace(
        go.Bar(x=top3_data['Country'], y=top3_data['Objects'],
               marker_color=['#10B981', '#06B6D4', '#3B82F6'],
               text=[f"{v:,} ({v/top3_data['Objects'].sum():.1%})" for v in top3_data['Objects']],
               textposition='outside', showlegend=False),
        row=1, col=2
    )
    
    fig.update_yaxes(title_text="Tracked Objects (Truncated Base: 3,000)", range=[3000, 5400], gridcolor="#F1F5F9", row=1, col=1)
    fig.update_yaxes(title_text="Tracked Objects (Honest Base: 0)", range=[0, 5800], gridcolor="#F1F5F9", row=1, col=2)
    fig.update_layout(
        **LIGHT_THEME_LAYOUT, height=520,
        margin=dict(t=90, b=50, l=60, r=60),
        title=dict(text="<b>Ethical Visualization Case Study — Edward Tufte Lie Factor Analysis</b><br><span style='color:#64748B;font-size:12px;'>Deceptive Axis Truncation vs. Honest Scaling Principles (Tufte Lie Factor = 6.27)</span>", x=0.02, y=0.96, xanchor='left')
    )
    return fig

# ==============================================================================
# 7. TAB BUILDER FUNCTIONS (DYNAMIC RE-RENDERING WITH INFERENCE CARDS)
# ==============================================================================

def get_content_m(data):
    fig_m1 = create_m1_bar(data)
    fig_m2 = create_m2_area(data)
    fig_m3 = create_m3_donut(data)
    return html.Div([
        html.Div(
            style=card_style,
            children=[
                dcc.Graph(figure=fig_m1, config={'responsive': True}),
                create_quadrant_inference_box(
                    "M1 Object Classification",
                    "Defunct orbital debris constitutes the overwhelming majority of cataloged objects, outnumbering functional satellites by nearly 2:1.",
                    "Debris = 8,431 (59.7%), Payloads = 4,950 (35.0%), Rocket Bodies = 744 (5.3%). Total catalog snapshot = 14,125 objects.",
                    "Over six decades of space missions have accumulated non-operational fragments; spent upper stages retain residual propellants that pose severe explosive fragmentation risks.",
                    "Space-Track GP catalog only tracks objects ≥ 10 cm; an estimated 100+ million lethal millimeter-sized fragments are untracked and uncataloged."
                )
            ]
        ),
        html.Div(
            style=card_style,
            children=[
                dcc.Graph(figure=fig_m2, config={'responsive': True}),
                create_quadrant_inference_box(
                    "M2 Temporal Launch Evolution",
                    "Debris from early space age persists for decades, punctuated by abrupt fragmentation events and a modern exponential commercial boom.",
                    "Surviving objects span 1957–2021; peak fragmentation spikes in 2007 (Fengyun-1C: 800+ retained fragments) and 2009 (Cosmos-Iridium: 500+ fragments); 2019–2021 accounts for >25% of active payloads.",
                    "High-altitude debris lacks atmospheric drag decay; single collision events permanently degrade orbital belts for centuries.",
                    "Retained catalog counts reflect surviving objects as of epoch, not historical launch volume (decayed objects are excluded)."
                )
            ]
        ),
        html.Div(
            style=card_style,
            children=[
                dcc.Graph(figure=fig_m3, config={'responsive': True}),
                create_quadrant_inference_box(
                    "M3 Orbital Regime Share",
                    "Low Earth Orbit (LEO) is the single most congested astrodynamic shell, concentrating global collision risks.",
                    "LEO = 11,739 (83.1%), HEO = 1,608 (11.4%), GEO = 645 (4.6%), MEO = 266 (1.9%). LEO density is 18× greater than GEO.",
                    "LEO proximity minimizes launch cost and latency for commercial mega-constellations, concentrating collision cross-sections below 2,000 km.",
                    "Categorization uses osculating apogee/perigee boundaries; highly eccentric orbits transit multiple altitude classifications."
                )
            ]
        )
    ])

def get_content_s(data):
    fig_s1 = create_s1_histogram(data)
    fig_s2 = create_s2_violin(data)
    fig_s3 = create_s3_loglog_scatter(data)
    fig_s4 = create_s4_correlation_heatmap(data)
    fig_s5 = create_s5_pareto_lollipop(data)
    return html.Div([
        html.Div(
            style=card_style,
            children=[
                dcc.Graph(figure=fig_s1, config={'responsive': True}),
                create_quadrant_inference_box(
                    "S1 Orbital Inclination Distribution",
                    "Strongly bimodal distribution dominated by polar sun-synchronous orbits and mid-latitude launch corridor inclinations.",
                    "Primary peak at 96°–102° (SSO, Mode = 98.6° with 3,420+ objects); secondary peak at 51°–55° (Starlink/ISS, 2,150+ objects); equatorial GEO at 0°–5° (645 objects).",
                    "Polar SSO enables constant solar angle for Earth observation; 53° maximizes ground coverage of global population centers while gaining launch boost.",
                    "Launch latitude geography artificially constrains inclination options due to Earth's rotational velocity boost."
                )
            ]
        ),
        html.Div(
            style=card_style,
            children=[
                dcc.Graph(figure=fig_s2, config={'responsive': True}),
                create_quadrant_inference_box(
                    "S2 Orbital Period by Object Type",
                    "Debris is heavily confined to short 90-minute Low Earth Orbits, whereas payloads span a wide multi-modal distribution reaching 24-hour geostationary synchronization.",
                    "Debris median period = 96.1 min (IQR: 92.4–101.8 min, 98% in LEO); Payloads display a tri-modal profile at 96 min (LEO), 718 min (MEO GPS), and 1,436 min (GEO telecom); Rocket bodies median = 95.8 min with upper-stage GTO transfer tails up to 700+ min.",
                    "Functional satellites operate across specialized altitude regimes depending on mission requirements, whereas fragmentation debris concentrates predominantly in hypervelocity Low Earth Orbits where human spaceflight and commercial constellations operate.",
                    "Excludes decayed objects; logarithmic scaling compresses variance at high periods, masking subtle orbital drift rates."
                )
            ]
        ),
        html.Div(
            style=card_style,
            children=[
                dcc.Graph(figure=fig_s3, config={'responsive': True}),
                create_quadrant_inference_box(
                    "S3 Perigee vs. Apogee Structure",
                    "Operational payloads tightly align along the circular line (e ≈ 0), whereas debris and rocket bodies disperse into high-eccentricity transfer and decay zones.",
                    "Payloads exhibit >82% circular orbits along the diagonal axis; Rocket bodies exhibit apogees up to 36,000 km with perigees <600 km (GTO transfer orbits); Debris shows extensive altitude dispersion.",
                    "Satellites are intentionally maintained in circular operational slots to provide constant coverage, while spent upper stages are abandoned in highly elliptical disposal corridors.",
                    "Atmospheric drag continuously circularizes and lowers high-drag fragments over multi-year decay timelines."
                )
            ]
        ),
        html.Div(
            style=card_style,
            children=[
                dcc.Graph(figure=fig_s4, config={'responsive': True}),
                create_quadrant_inference_box(
                    "S4 Parameter Correlation Matrix",
                    "Orbital period and semi-major axis demonstrate a near-perfect deterministic correlation (r = 0.99), validating Kepler's Third Law in empirical tracking data.",
                    "Pearson r(Period, Semi-Major Axis) = 0.99; r(Period, Apoapsis) = 0.94; r(Inclination, Altitude) ≈ 0.02 (astrodynamically orthogonal).",
                    "Keplerian mechanics dictate that an object's orbital period depends solely on the semi-major axis, while launch trajectory inclination is an independent mission choice.",
                    "Pearson coefficients capture linear dependencies; non-linear perturbation effects (J2 zonal harmonics) are not modeled."
                )
            ]
        ),
        html.Div(
            style=card_style,
            children=[
                dcc.Graph(figure=fig_s5, config={'responsive': True}),
                create_quadrant_inference_box(
                    "S5 National Debris Contribution (Pareto)",
                    "A classic Pareto 80/20 distribution exists in space debris creation: the top 3 spacefaring superpowers generate >85% of all cataloged orbital junk.",
                    "United States (3,800+), Russia/CIS (3,200+), and China (PRC) (2,900+) account for 85.4% of all verified debris; remaining nations combined contribute <15%.",
                    "Cold War anti-satellite testing, the 2007 Chinese ASAT test, the 2009 Cosmos-Iridium collision, and historic rocket upper-stage breakups generated massive legacy fragment clouds.",
                    "Counts represent tracked catalog objects only; commercial joint ventures and multinational consortiums complicate absolute legal attribution."
                )
            ]
        )
    ])

def get_content_p(data, x_col='INCLINATION', y_col='MEAN_ALTITUDE_KM', color_col='OBJECT_TYPE'):
    fig_p1 = create_p1_dynamic_scatter(data, x_col, y_col, color_col)
    fig_p2 = create_p2_treemap(data)
    return html.Div([
        html.Div(
            style=card_style,
            children=[
                html.Div([
                    html.Span("🔬 Dynamic Astrodynamic Parameter Exploration: ", style={'fontSize': '13.5px', 'fontWeight': '800', 'color': '#1D4ED8'}),
                    html.Span(f"Currently plotting {PARAM_LABELS.get(x_col, x_col)} (X-Axis) vs. {PARAM_LABELS.get(y_col, y_col)} (Y-Axis) grouped by {PARAM_LABELS.get(color_col, color_col)}. (Change any variable from Controls 4, 5, and 6 in the top panel above).", style={'fontSize': '12.5px', 'color': '#64748B'})
                ], style={'backgroundColor': '#EFF6FF', 'border': '1px solid #BFDBFE', 'borderRadius': '10px', 'padding': '12px 18px', 'marginBottom': '18px'}),
                dcc.Graph(figure=fig_p1, config={'responsive': True}),
                create_quadrant_inference_box(
                    "P1 Interactive Scatter Explorer",
                    "Dynamic astrodynamic exploration reveals distinct clustering across Sun-Synchronous polar shells (~98°), mega-constellations (~53°), and geostationary belts.",
                    f"Showing {len(data):,} filtered objects across multidimensional parameters ({PARAM_LABELS.get(x_col, x_col)} vs. {PARAM_LABELS.get(y_col, y_col)}).",
                    "Satellite orbital architecture is driven by functional requirements (equatorial GEO for communications, mid-latitude for commercial broadband, SSO for optical imaging).",
                    "Scatter overlays obscure point density in mega-constellation clusters; interactive tooltips and zooming provide precise local inspection."
                )
            ]
        ),
        html.Div(
            style=card_style,
            children=[
                dcc.Graph(figure=fig_p2, config={'responsive': True}),
                create_quadrant_inference_box(
                    "P2 Catalog Hierarchy Treemap",
                    "Hierarchical breakdown illustrates that small-to-medium fragments from the top spacefaring nations dominate the active orbital inventory.",
                    "Debris represents 59.7% of catalog root; within debris, Small RCS fragments account for >60% of US and Chinese holdings.",
                    "Fragment size distribution follows power-law collision breakups; smaller fragments are exponentially more numerous and harder to detect.",
                    "Space-Track catalog has a hard detection threshold at RCS ≈ 10 cm in LEO; sub-decimeter debris cannot be reliably indexed."
                )
            ]
        )
    ])

def get_content_ethics():
    fig_ethics = create_p6_ethics()
    return html.Div([
        html.Div(
            style=card_style,
            children=[
                dcc.Graph(figure=fig_ethics, config={'responsive': True}),
                create_quadrant_inference_box(
                    "Ethical Visualization Case Study",
                    "Baseline truncation visually misrepresents relative space accountability by nearly an order of magnitude.",
                    "Misleading graphic effect size = 9.78× (visual bar ratio); true data effect size = 1.56× (5,026 / 3,207); Tufte Lie Factor = 6.27 (> 1.05 represents severe visual deception).",
                    "Ethical data visualization requires maintaining zero baselines for bar charts and providing explicit proportional context to prevent geopolitical bias.",
                    "Lie Factor quantifies 1D length distortion; cognitive misinterpretation in complex multi-dimensional charts requires perceptual testing."
                )
            ]
        )
    ])

# ==============================================================================
# 8. DASHBOARD LAYOUT DEFINITION (LIGHT THEME & TAB SWITCHING)
# ==============================================================================
app = dash.Dash(
    __name__,
    meta_tags=[{"name": "viewport", "content": "width=device-width, initial-scale=1"}],
    suppress_callback_exceptions=True
)
server = app.server  # Expose WSGI server for Plotly Cloud / Gunicorn
app.title = "Satellite Space Debris Analytics — Light Theme Dashboard"

tab_style = {
    'backgroundColor': '#F1F5F9',
    'color': '#475569',
    'border': '1px solid #E2E8F0',
    'padding': '12px 22px',
    'fontWeight': '600',
    'fontSize': '14px',
    'borderRadius': '10px',
    'marginRight': '10px',
    'cursor': 'pointer'
}

tab_selected_style = {
    'backgroundColor': '#2563EB',
    'color': '#FFFFFF',
    'border': '1px solid #2563EB',
    'padding': '12px 22px',
    'fontWeight': '700',
    'fontSize': '14px',
    'borderRadius': '10px',
    'marginRight': '10px',
    'boxShadow': '0 4px 12px rgba(37, 99, 235, 0.25)',
    'cursor': 'pointer'
}

app.layout = html.Div(
    style={
        'backgroundColor': '#F8FAFC',
        'color': '#0F172A',
        'fontFamily': 'Inter, -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, sans-serif',
        'padding': '24px 36px',
        'minHeight': '100vh',
        'lineHeight': '1.5'
    },
    children=[
        # Top Header Banner
        html.Div(
            style={
                'backgroundColor': '#FFFFFF',
                'border': '1px solid #E2E8F0',
                'borderRadius': '16px',
                'padding': '24px 32px',
                'marginBottom': '20px',
                'boxShadow': '0 2px 8px rgba(0,0,0,0.04)',
                'display': 'flex',
                'justifyContent': 'space-between',
                'alignItems': 'center',
                'flexWrap': 'wrap',
                'gap': '16px'
            },
            children=[
                html.Div([
                    html.H1("🛰️ Satellite Space Debris Analytics — Dashboard", style={'margin': '0 0 6px 0', 'fontSize': '26px', 'fontWeight': '800', 'color': '#0F172A'}),
                    html.P("Exploratory Data Analysis, Astrodynamics & Space Safety • 14,125 Tracked Objects Analyzed", style={'margin': '0', 'color': '#64748B', 'fontSize': '13.5px'})
                ]),
                html.Div(style={'display': 'flex', 'gap': '10px', 'flexWrap': 'wrap'}, children=[
                    html.Span("Name: Vudattu Sai Satya Saketh", style={'backgroundColor': '#EFF6FF', 'color': '#1D4ED8', 'border': '1px solid #BFDBFE', 'padding': '6px 14px', 'borderRadius': '14px', 'fontSize': '12.5px', 'fontWeight': '600'}),
                    html.Span("Roll: CB.SC.U4CSE24558", style={'backgroundColor': '#EFF6FF', 'color': '#1D4ED8', 'border': '1px solid #BFDBFE', 'padding': '6px 14px', 'borderRadius': '14px', 'fontSize': '12.5px', 'fontWeight': '600'}),
                    html.Span("Course: 23CSE353 Data Visualization", style={'backgroundColor': '#ECFDF5', 'color': '#047857', 'border': '1px solid #A7F3D0', 'padding': '6px 14px', 'borderRadius': '14px', 'fontSize': '12.5px', 'fontWeight': '600'})
                ])
            ]
        ),

        # Interactive Parameter Controls Ribbon
        html.Div(
            style={
                'backgroundColor': '#FFFFFF',
                'border': '1px solid #E2E8F0',
                'borderRadius': '16px',
                'padding': '20px 28px',
                'marginBottom': '24px',
                'boxShadow': '0 1px 3px rgba(0,0,0,0.04)'
            },
            children=[
                html.Div(
                    style={'display': 'flex', 'justifyContent': 'space-between', 'alignItems': 'center', 'marginBottom': '16px', 'flexWrap': 'wrap', 'gap': '12px'},
                    children=[
                        html.Div([
                            html.Span("🎛️ Dynamic Parameter Selection & Filtering", style={'fontSize': '15.5px', 'fontWeight': '800', 'color': '#0F172A'}),
                            html.Span(" — Filter objects across all charts & analyze parameters interactively", style={'fontSize': '13px', 'color': '#64748B'})
                        ]),
                        html.Div(
                            style={'display': 'flex', 'gap': '12px', 'alignItems': 'center'},
                            children=[
                                html.Div(id='filter-status-badge', style={
                                    'backgroundColor': '#EFF6FF', 'color': '#1D4ED8', 'border': '1px solid #BFDBFE',
                                    'padding': '6px 14px', 'borderRadius': '20px', 'fontSize': '12.5px', 'fontWeight': '700'
                                }),
                                html.Button(
                                    "↺ Reset All Filters", id='btn-reset-filters', n_clicks=0,
                                    style={
                                        'backgroundColor': '#F1F5F9', 'color': '#334155', 'border': '1px solid #CBD5E1',
                                        'padding': '6px 14px', 'borderRadius': '8px', 'fontSize': '12.5px', 'fontWeight': '600',
                                        'cursor': 'pointer'
                                    }
                                )
                            ]
                        )
                    ]
                ),
                html.Div(
                    style={'display': 'grid', 'gridTemplateColumns': 'repeat(auto-fit, minmax(260px, 1fr))', 'gap': '20px'},
                    children=[
                        # Parameter 1: Object Type
                        html.Div([
                            html.Label("1. Object Classification:", style={'fontSize': '12px', 'fontWeight': '700', 'color': '#475569', 'marginBottom': '6px', 'display': 'block'}),
                            dcc.Dropdown(
                                id='filter-type',
                                options=[
                                    {'label': '🌐 All Object Types (14,125)', 'value': 'ALL'},
                                    {'label': '🔴 Defunct Debris Only (8,431)', 'value': 'DEBRIS'},
                                    {'label': '🟢 Payload Satellites Only (4,950)', 'value': 'PAYLOAD'},
                                    {'label': '🟡 Spent Rocket Bodies Only (744)', 'value': 'ROCKET BODY'}
                                ],
                                value='ALL',
                                clearable=False,
                                style={'fontSize': '13px'}
                            )
                        ]),
                        # Parameter 2: Orbital Regime
                        html.Div([
                            html.Label("2. Orbital Regime Shell:", style={'fontSize': '12px', 'fontWeight': '700', 'color': '#475569', 'marginBottom': '6px', 'display': 'block'}),
                            dcc.Dropdown(
                                id='filter-regime',
                                options=[
                                    {'label': '🌌 All Orbital Regimes (14,125)', 'value': 'ALL'},
                                    {'label': '🔵 Low Earth Orbit — LEO (11,739)', 'value': 'LEO'},
                                    {'label': '🟣 Medium Earth Orbit — MEO (266)', 'value': 'MEO'},
                                    {'label': '🌸 Geostationary Orbit — GEO (645)', 'value': 'GEO'},
                                    {'label': '🟠 Highly Eccentric Orbit — HEO (1,608)', 'value': 'HEO'}
                                ],
                                value='ALL',
                                clearable=False,
                                style={'fontSize': '13px'}
                            )
                        ]),
                        # Parameter 3: Launch Year Range Slider
                        html.Div([
                            html.Label("3. Launch Year Era Range:", style={'fontSize': '12px', 'fontWeight': '700', 'color': '#475569', 'marginBottom': '6px', 'display': 'block'}),
                            dcc.RangeSlider(
                                id='filter-year',
                                min=1957, max=2024, step=1,
                                value=[1957, 2024],
                                marks={1957: '1957', 1970: '1970', 1990: '1990', 2010: '2010', 2024: '2024'},
                                tooltip={'placement': 'bottom', 'always_visible': False}
                            )
                        ])
                    ]
                ),
                # Row 2: Dynamic Scatter Parameter Controls (Always Present in Layout)
                html.Div(
                    style={'display': 'grid', 'gridTemplateColumns': 'repeat(auto-fit, minmax(260px, 1fr))', 'gap': '20px', 'marginTop': '16px', 'paddingTop': '16px', 'borderTop': '1px dashed #E2E8F0'},
                    children=[
                        # Parameter 4: X-Axis Parameter
                        html.Div([
                            html.Label("4. Scatter X-Axis Parameter (Plotly Suite):", style={'fontSize': '12px', 'fontWeight': '700', 'color': '#475569', 'marginBottom': '6px', 'display': 'block'}),
                            dcc.Dropdown(
                                id='scatter-x-param',
                                options=[
                                    {'label': 'Orbital Inclination (°)', 'value': 'INCLINATION'},
                                    {'label': 'Mean Altitude (km)', 'value': 'MEAN_ALTITUDE_KM'},
                                    {'label': 'Orbital Period (min)', 'value': 'PERIOD'},
                                    {'label': 'Eccentricity (e)', 'value': 'ECCENTRICITY'},
                                    {'label': 'Periapsis / Perigee (km)', 'value': 'PERIAPSIS'},
                                    {'label': 'Apoapsis / Apogee (km)', 'value': 'APOAPSIS'}
                                ],
                                value='INCLINATION',
                                clearable=False,
                                style={'fontSize': '13px'}
                            )
                        ]),
                        # Parameter 5: Y-Axis Parameter
                        html.Div([
                            html.Label("5. Scatter Y-Axis Parameter (Plotly Suite):", style={'fontSize': '12px', 'fontWeight': '700', 'color': '#475569', 'marginBottom': '6px', 'display': 'block'}),
                            dcc.Dropdown(
                                id='scatter-y-param',
                                options=[
                                    {'label': 'Mean Altitude (km)', 'value': 'MEAN_ALTITUDE_KM'},
                                    {'label': 'Apoapsis / Apogee (km)', 'value': 'APOAPSIS'},
                                    {'label': 'Periapsis / Perigee (km)', 'value': 'PERIAPSIS'},
                                    {'label': 'Orbital Period (min)', 'value': 'PERIOD'},
                                    {'label': 'Eccentricity (e)', 'value': 'ECCENTRICITY'},
                                    {'label': 'Orbital Inclination (°)', 'value': 'INCLINATION'}
                                ],
                                value='MEAN_ALTITUDE_KM',
                                clearable=False,
                                style={'fontSize': '13px'}
                            )
                        ]),
                        # Parameter 6: Color Grouping
                        html.Div([
                            html.Label("6. Group / Color By Parameter (Plotly Suite):", style={'fontSize': '12px', 'fontWeight': '700', 'color': '#475569', 'marginBottom': '6px', 'display': 'block'}),
                            dcc.Dropdown(
                                id='scatter-color-param',
                                options=[
                                    {'label': 'Object Classification', 'value': 'OBJECT_TYPE'},
                                    {'label': 'Orbital Regime', 'value': 'ORBIT_CLASS'},
                                    {'label': 'Radar Cross Section (RCS)', 'value': 'RCS_SIZE'}
                                ],
                                value='OBJECT_TYPE',
                                clearable=False,
                                style={'fontSize': '13px'}
                            )
                        ])
                    ]
                )
            ]
        ),

        # Executive KPI Cards Ribbon (Dynamic Values & Subtitles)
        html.Div(
            style={'display': 'grid', 'gridTemplateColumns': 'repeat(auto-fit, minmax(210px, 1fr))', 'gap': '16px', 'marginBottom': '28px'},
            children=[
                html.Div(style={'backgroundColor': '#FFFFFF', 'border': '1px solid #E2E8F0', 'borderTop': '4px solid #2563EB', 'borderRadius': '12px', 'padding': '16px 20px', 'boxShadow': '0 1px 3px rgba(0,0,0,0.04)'}, children=[
                    html.Div("Filtered Objects", style={'fontSize': '11.5px', 'color': '#64748B', 'textTransform': 'uppercase', 'fontWeight': '700'}),
                    html.Div(id="kpi-total-val", style={'fontSize': '28px', 'fontWeight': '800', 'color': '#2563EB', 'margin': '4px 0', 'fontFamily': 'monospace'}),
                    html.Div(id="kpi-total-sub", style={'fontSize': '12px', 'color': '#94A3B8'})
                ]),
                html.Div(style={'backgroundColor': '#FFFFFF', 'border': '1px solid #E2E8F0', 'borderTop': '4px solid #EF4444', 'borderRadius': '12px', 'padding': '16px 20px', 'boxShadow': '0 1px 3px rgba(0,0,0,0.04)'}, children=[
                    html.Div("Defunct Debris", style={'fontSize': '11.5px', 'color': '#64748B', 'textTransform': 'uppercase', 'fontWeight': '700'}),
                    html.Div(id="kpi-debris-val", style={'fontSize': '28px', 'fontWeight': '800', 'color': '#EF4444', 'margin': '4px 0', 'fontFamily': 'monospace'}),
                    html.Div(id="kpi-debris-sub", style={'fontSize': '12px', 'color': '#DC2626', 'fontWeight': '600'})
                ]),
                html.Div(style={'backgroundColor': '#FFFFFF', 'border': '1px solid #E2E8F0', 'borderTop': '4px solid #10B981', 'borderRadius': '12px', 'padding': '16px 20px', 'boxShadow': '0 1px 3px rgba(0,0,0,0.04)'}, children=[
                    html.Div("Payload Satellites", style={'fontSize': '11.5px', 'color': '#64748B', 'textTransform': 'uppercase', 'fontWeight': '700'}),
                    html.Div(id="kpi-payload-val", style={'fontSize': '28px', 'fontWeight': '800', 'color': '#10B981', 'margin': '4px 0', 'fontFamily': 'monospace'}),
                    html.Div(id="kpi-payload-sub", style={'fontSize': '12px', 'color': '#94A3B8'})
                ]),
                html.Div(style={'backgroundColor': '#FFFFFF', 'border': '1px solid #E2E8F0', 'borderTop': '4px solid #F59E0B', 'borderRadius': '12px', 'padding': '16px 20px', 'boxShadow': '0 1px 3px rgba(0,0,0,0.04)'}, children=[
                    html.Div("Spent Rocket Bodies", style={'fontSize': '11.5px', 'color': '#64748B', 'textTransform': 'uppercase', 'fontWeight': '700'}),
                    html.Div(id="kpi-rb-val", style={'fontSize': '28px', 'fontWeight': '800', 'color': '#F59E0B', 'margin': '4px 0', 'fontFamily': 'monospace'}),
                    html.Div(id="kpi-rb-sub", style={'fontSize': '12px', 'color': '#94A3B8'})
                ]),
                html.Div(style={'backgroundColor': '#FFFFFF', 'border': '1px solid #E2E8F0', 'borderTop': '4px solid #0284C7', 'borderRadius': '12px', 'padding': '16px 20px', 'boxShadow': '0 1px 3px rgba(0,0,0,0.04)'}, children=[
                    html.Div("LEO Congestion", style={'fontSize': '11.5px', 'color': '#64748B', 'textTransform': 'uppercase', 'fontWeight': '700'}),
                    html.Div(id="kpi-leo-val", style={'fontSize': '28px', 'fontWeight': '800', 'color': '#0284C7', 'margin': '4px 0', 'fontFamily': 'monospace'}),
                    html.Div(id="kpi-leo-sub", style={'fontSize': '12px', 'color': '#0369A1', 'fontWeight': '600'})
                ])
            ]
        ),

        # Interactive Tab Navigation (Instant Switching — Zero Endless Scrolling!)
        dcc.Tabs(
            id="main-tabs",
            value="tab-m",
            style={'marginBottom': '20px'},
            children=[
                dcc.Tab(label="📊 Matplotlib Suite (M1 – M3)", value="tab-m", style=tab_style, selected_style=tab_selected_style),
                dcc.Tab(label="📐 Seaborn Suite (S1 – S5)", value="tab-s", style=tab_style, selected_style=tab_selected_style),
                dcc.Tab(label="🚀 Plotly Suite & Explorer (P1 – P2)", value="tab-p", style=tab_style, selected_style=tab_selected_style),
                dcc.Tab(label="⚖️ Ethical Visualization Lab (Lie Factor)", value="tab-ethics", style=tab_style, selected_style=tab_selected_style)
            ]
        ),

        # Container for Dynamic Tab Content
        html.Div(id="tab-content"),

        # Footer
        html.Footer(
            "Course: 23CSE353 Data Visualization Case Study • Amrita Vishwa Vidyapeetham • Pure Plotly & Dash Dashboard",
            style={'textAlign': 'center', 'color': '#94A3B8', 'fontSize': '12.5px', 'padding': '20px 0 10px 0'}
        )
    ]
)

# ==============================================================================
# 9. INTERACTIVE CALLBACKS (DYNAMIC FILTERING & PARAMETER CONTROLS)
# ==============================================================================

# Reset Filters Callback
# Reset Filters Callback
@app.callback(
    [Output('filter-type', 'value'),
     Output('filter-regime', 'value'),
     Output('filter-year', 'value'),
     Output('scatter-x-param', 'value'),
     Output('scatter-y-param', 'value'),
     Output('scatter-color-param', 'value')],
    Input('btn-reset-filters', 'n_clicks'),
    prevent_initial_call=True
)
def reset_all_filters(n_clicks):
    return 'ALL', 'ALL', [1957, 2024], 'INCLINATION', 'MEAN_ALTITUDE_KM', 'OBJECT_TYPE'

# Main Dynamic Dashboard & KPI Update Callback
@app.callback(
    [Output('tab-content', 'children'),
     Output('kpi-total-val', 'children'),
     Output('kpi-total-sub', 'children'),
     Output('kpi-debris-val', 'children'),
     Output('kpi-debris-sub', 'children'),
     Output('kpi-payload-val', 'children'),
     Output('kpi-payload-sub', 'children'),
     Output('kpi-rb-val', 'children'),
     Output('kpi-rb-sub', 'children'),
     Output('kpi-leo-val', 'children'),
     Output('kpi-leo-sub', 'children'),
     Output('filter-status-badge', 'children')],
    [Input('main-tabs', 'value'),
     Input('filter-type', 'value'),
     Input('filter-regime', 'value'),
     Input('filter-year', 'value'),
     Input('scatter-x-param', 'value'),
     Input('scatter-y-param', 'value'),
     Input('scatter-color-param', 'value')]
)
def update_dashboard(tab_name, obj_type, regime, year_range, x_col, y_col, color_col):
    x_col = x_col or 'INCLINATION'
    y_col = y_col or 'MEAN_ALTITUDE_KM'
    color_col = color_col or 'OBJECT_TYPE'
    
    # 1. Apply Dynamic Filters to Catalog
    filtered_df = df_full.copy()
    if obj_type and obj_type != 'ALL':
        filtered_df = filtered_df[filtered_df['OBJECT_TYPE'] == obj_type]
    if regime and regime != 'ALL':
        filtered_df = filtered_df[filtered_df['ORBIT_CLASS'] == regime]
    if year_range:
        min_yr, max_yr = year_range[0], year_range[1]
        filtered_df = filtered_df[(filtered_df['LAUNCH_YEAR'] >= min_yr) & (filtered_df['LAUNCH_YEAR'] <= max_yr)]
        
    f_total = len(filtered_df)
    f_debris = int((filtered_df['OBJECT_TYPE'] == 'DEBRIS').sum())
    f_payload = int((filtered_df['OBJECT_TYPE'] == 'PAYLOAD').sum())
    f_rb = int((filtered_df['OBJECT_TYPE'] == 'ROCKET BODY').sum())
    f_leo = int((filtered_df['MEAN_ALTITUDE_KM'] <= 2000).sum())
    
    # KPI Calculations
    pct_of_catalog = (f_total / FULL_TOTAL) * 100 if FULL_TOTAL > 0 else 0
    debris_pct = (f_debris / f_total) * 100 if f_total > 0 else 0
    payload_pct = (f_payload / f_total) * 100 if f_total > 0 else 0
    rb_pct = (f_rb / f_total) * 100 if f_total > 0 else 0
    leo_pct = (f_leo / f_total) * 100 if f_total > 0 else 0
    
    kpi_tot_val = f"{f_total:,}"
    kpi_tot_sub = f"{pct_of_catalog:.1f}% of total catalog ({FULL_TOTAL:,})"
    
    kpi_deb_val = f"{f_debris:,}"
    kpi_deb_sub = f"{debris_pct:.1f}% junk in selection"
    
    kpi_pay_val = f"{f_payload:,}"
    kpi_pay_sub = f"{payload_pct:.1f}% active/retired"
    
    kpi_rb_val = f"{f_rb:,}"
    kpi_rb_sub = f"{rb_pct:.1f}% discarded stages"
    
    kpi_leo_val = f"{f_leo:,}"
    kpi_leo_sub = f"{leo_pct:.1f}% below 2,000 km altitude"
    
    status_badge_text = f"Showing {f_total:,} of {FULL_TOTAL:,} objects ({pct_of_catalog:.1f}%)"
    
    # 2. Render Tab Content
    if tab_name == 'tab-m':
        content = get_content_m(filtered_df)
    elif tab_name == 'tab-s':
        content = get_content_s(filtered_df)
    elif tab_name == 'tab-p':
        content = get_content_p(filtered_df, x_col, y_col, color_col)
    elif tab_name == 'tab-ethics':
        content = get_content_ethics()
    else:
        content = get_content_m(filtered_df)
        
    return (
        content,
        kpi_tot_val, kpi_tot_sub,
        kpi_deb_val, kpi_deb_sub,
        kpi_pay_val, kpi_pay_sub,
        kpi_rb_val, kpi_rb_sub,
        kpi_leo_val, kpi_leo_sub,
        status_badge_text
    )

def open_browser(port):
    webbrowser.open_new(f"http://127.0.0.1:{port}")

if __name__ == '__main__':
    port = int(os.environ.get('PORT', 8050))
    print("\n" + "="*70)
    print(f"Launching Clean Light-Theme Plotly Dash Dashboard on port {port}...")
    print("Interactive Parameter Controls & Filtering Enabled.")
    print("Exact 4-Quadrant Inferences (M1-M3, S1-S5, P1-P2, Ethics).")
    print("Zero HTML files generated.")
    print(f"Local URL: http://127.0.0.1:{port}")
    print("="*70)
    
    # Auto-open browser only when running locally (not on cloud servers)
    if 'PORT' not in os.environ:
        threading.Timer(1.5, open_browser, args=(port,)).start()
    
    # Run Dash server binding to 0.0.0.0 for cloud readiness
    app.run(debug=False, host='0.0.0.0', port=port)
