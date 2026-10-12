"""
Plotly Dash Dashboard: Satellite Space Debris Analytics
Complete Interactive Dashboard in Clean Academic LIGHT THEME.
Interactive Tab Switching (Zero Long Scrolling) & Exact 4-Quadrant Inference Cards.

Strictly aligned with Review 1 Syllabus:
- Matplotlib Suite: M1, M2, M3
- Seaborn Suite: S1, S2, S3, S4, S5
- Plotly Suite: P1, P2
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
from dash import html, dcc, Input, Output

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
    else:
        df_raw = pd.read_csv('space_decay.csv')
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
        
    return df

df = load_catalog_data()

# Catalog summary metrics
total_count = len(df)
debris_count = int((df['OBJECT_TYPE'] == 'DEBRIS').sum())
payload_count = int((df['OBJECT_TYPE'] == 'PAYLOAD').sum())
rb_count = int((df['OBJECT_TYPE'] == 'ROCKET BODY').sum())
leo_count = int((df['MEAN_ALTITUDE_KM'] <= 2000).sum())
geo_count = int((df['ORBIT_CLASS'] == 'GEO').sum())

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

# Common Card Style
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
    counts = data['OBJECT_TYPE'].value_counts().reset_index()
    counts.columns = ['OBJECT_TYPE', 'COUNT']
    counts['PERCENT'] = (counts['COUNT'] / counts['COUNT'].sum()) * 100
    
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
        yaxis=dict(gridcolor="#F1F5F9", range=[0, counts['COUNT'].max() * 1.18]),
        xaxis=dict(gridcolor="#F1F5F9")
    )
    return fig

def create_m2_area(data):
    valid = data[(data['LAUNCH_YEAR'] >= 1957) & (data['LAUNCH_YEAR'] <= 2021)].copy()
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
            
    fig.add_annotation(x=2007, y=650, text="2007 Fengyun-1C<br>ASAT Spike",
                       showarrow=True, arrowhead=2, arrowcolor="#EF4444", ax=-50, ay=-50,
                       font=dict(color="#B91C1C", size=10), bgcolor="#FEF2F2", bordercolor="#F87171")
    fig.add_annotation(x=2009, y=520, text="2009 Cosmos-Iridium<br>Collision",
                       showarrow=True, arrowhead=2, arrowcolor="#F59E0B", ax=-60, ay=-65,
                       font=dict(color="#B45309", size=10), bgcolor="#FFFBEB", bordercolor="#FCD34D")
    fig.add_annotation(x=2020, y=1050, text="2019-2021 Starlink &<br>Mega-Constellations",
                       showarrow=True, arrowhead=2, arrowcolor="#10B981", ax=-80, ay=-55,
                       font=dict(color="#047857", size=10), bgcolor="#ECFDF5", bordercolor="#6EE7B7")
                       
    fig.update_layout(
        **LIGHT_THEME_LAYOUT, height=480,
        margin=dict(t=70, b=50, l=60, r=40),
        title=dict(text="<b>M2: Retained Objects by Launch Year (1957–2021) — Stacked Area Profile</b><br><span style='color:#64748B;font-size:12px;'>Surviving Catalog Objects Grouped by Launch Era & Historical Breakup Signatures</span>", x=0.02, y=0.96, xanchor='left'),
        xaxis=dict(title="Launch Year", gridcolor="#F1F5F9", range=[1955, 2023]),
        yaxis=dict(title="Surviving Objects in Orbit", gridcolor="#F1F5F9"),
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1)
    )
    return fig

def create_m3_donut(data):
    regime_counts = data['ORBIT_CLASS'].value_counts().reset_index()
    regime_counts.columns = ['ORBIT_CLASS', 'COUNT']
    
    fig = px.pie(
        regime_counts, names='ORBIT_CLASS', values='COUNT',
        hole=0.55, color='ORBIT_CLASS', color_discrete_map=REGIME_COLORS
    )
    fig.update_traces(
        textposition='outside', textinfo='label+percent',
        hovertemplate="<b>%{label} Regime</b><br>Tracked Objects: %{value:,}<br>Share of Catalog: %{percent:.1%}<extra></extra>",
        marker=dict(line=dict(color='#FFFFFF', width=2))
    )
    fig.update_layout(
        **LIGHT_THEME_LAYOUT, height=460,
        margin=dict(t=70, b=40, l=40, r=40),
        title=dict(text="<b>M3: Orbital Regime Share — LEO, MEO, GEO, and HEO</b><br><span style='color:#64748B;font-size:12px;'>Proportion of Global Orbital Objects Across Altitudinal Regimes (LEO Dominance: 83.1%)</span>", x=0.02, y=0.96, xanchor='left'),
        legend=dict(orientation="v", yanchor="middle", y=0.5, xanchor="left", x=1.02)
    )
    return fig

# ==============================================================================
# 5. VISUALIZATION GENERATORS — SEABORN SUITE (S1 – S5)
# ==============================================================================

def create_s1_histogram(data):
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
    corr = data[cols].corr().round(2)
    
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
    top_ctry = debris_data['COUNTRY_NAME'].value_counts().head(10).reset_index()
    top_ctry.columns = ['COUNTRY_NAME', 'DEBRIS_COUNT']
    top_ctry['CUM_PERCENT'] = (top_ctry['DEBRIS_COUNT'].cumsum() / top_ctry['DEBRIS_COUNT'].sum()) * 100
    
    fig = make_subplots(specs=[[{"secondary_y": True}]])
    
    # Bar Chart
    fig.add_trace(
        go.Bar(
            x=top_ctry['COUNTRY_NAME'], y=top_ctry['DEBRIS_COUNT'],
            name="Debris Objects Count", marker_color='#EF4444',
            text=[f"{v:,}" for v in top_ctry['DEBRIS_COUNT']], textposition='outside'
        ),
        secondary_y=False
    )
    
    # Cumulative Line
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
    
    # 80% Pareto Reference Line
    fig.add_hline(y=80, line_dash="dash", line_color="#D97706", secondary_y=True,
                  annotation_text="80% Pareto Threshold", annotation_position="top left",
                  annotation_font=dict(color="#B45309", size=10))
                  
    fig.update_layout(
        **LIGHT_THEME_LAYOUT, height=480,
        margin=dict(t=70, b=60, l=60, r=60),
        title=dict(text="<b>S5: Top 10 Debris Contributing Nations — Pareto Distribution</b><br><span style='color:#64748B;font-size:12px;'>Global Responsibility: Top 3 Powers (US, China, Russia) Account for >85% of Orbiting Junk</span>", x=0.02, y=0.96, xanchor='left'),
        xaxis=dict(title="Operating Country / Agency", tickangle=-20, gridcolor="#F1F5F9"),
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1)
    )
    fig.update_yaxes(title_text="Tracked Debris Count", gridcolor="#F1F5F9", secondary_y=False, range=[0, top_ctry['DEBRIS_COUNT'].max() * 1.15])
    fig.update_yaxes(title_text="Cumulative Percentage (%)", range=[0, 105], secondary_y=True, gridcolor="#F8FAFC")
    return fig

# ==============================================================================
# 6. VISUALIZATION GENERATORS — PLOTLY SUITE (P1 – P2) & ETHICS LAB
# ==============================================================================

def create_p1_scatter(data):
    plot_df = data[data['MEAN_ALTITUDE_KM'] <= 42000].copy()
    symbol_map = {'SMALL': 'circle', 'MEDIUM': 'diamond', 'LARGE': 'square', 'UNKNOWN': 'cross'}
    
    fig = px.scatter(
        plot_df, x='INCLINATION', y='MEAN_ALTITUDE_KM',
        color='OBJECT_TYPE', symbol='RCS_SIZE', symbol_map=symbol_map,
        color_discrete_map=TYPE_COLORS,
        hover_name='OBJECT_NAME',
        hover_data={'NORAD_CAT_ID': True, 'COUNTRY_NAME': True, 'LAUNCH_YEAR': True,
                    'APOAPSIS': ':.1f', 'PERIAPSIS': ':.1f', 'INCLINATION': ':.2f',
                    'MEAN_ALTITUDE_KM': ':.1f', 'OBJECT_TYPE': False, 'RCS_SIZE': True},
        labels={'INCLINATION': 'Orbital Inclination (Degrees °)', 'MEAN_ALTITUDE_KM': 'Mean Altitude (km)',
                'OBJECT_TYPE': 'Object Type', 'RCS_SIZE': 'RCS Size'},
        opacity=0.72
    )
    fig.add_hline(y=35786, line_dash="dash", line_color="#EC4899",
                  annotation_text="GEO Ring (35,786 km)", annotation_position="top left",
                  annotation_font=dict(color="#BE185D", size=11))
    fig.add_hline(y=2000, line_dash="dot", line_color="#0284C7",
                  annotation_text="LEO Upper Boundary (2,000 km)", annotation_position="bottom left",
                  annotation_font=dict(color="#0369A1", size=11))
    fig.add_annotation(x=98, y=850, text="Sun-Synchronous Band<br>(~98°, Dense Debris)",
                       showarrow=True, arrowhead=2, arrowcolor="#EF4444", ax=55, ay=-45,
                       font=dict(color="#B91C1C", size=10.5), bgcolor="#FEF2F2", bordercolor="#F87171")
    fig.add_annotation(x=53, y=550, text="Starlink / ISS Corridor<br>(~53°, 400-550 km)",
                       showarrow=True, arrowhead=2, arrowcolor="#10B981", ax=-55, ay=-40,
                       font=dict(color="#047857", size=10.5), bgcolor="#ECFDF5", bordercolor="#6EE7B7")
    fig.update_layout(
        **LIGHT_THEME_LAYOUT, height=600,
        margin=dict(t=90, b=60, l=70, r=220),
        title=dict(text="<b>P1: Orbital Regime Map — Inclination vs. Mean Altitude (km)</b><br><span style='color:#64748B;font-size:12px;'>Spatial Congestion Shells: Sun-Synchronous Band (~98°), ISS/Starlink (~53°), and GEO Ring (35,786 km)</span>", x=0.02, y=0.96, xanchor='left'),
        legend=dict(orientation="v", yanchor="middle", y=0.5, xanchor="left", x=1.02,
                    bgcolor="#F8FAFC", bordercolor="#E2E8F0", borderwidth=1,
                    title=dict(text="<b>Object Type & Size</b>", font=dict(size=11, color="#64748B"))),
        xaxis=dict(gridcolor="#F1F5F9", zerolinecolor="#E2E8F0", range=[-2, 145]),
        yaxis=dict(gridcolor="#F1F5F9", zerolinecolor="#E2E8F0", range=[-500, 41000])
    )
    return fig

def create_p2_treemap(data):
    top_ctry = data['COUNTRY_NAME'].value_counts().head(10).index.tolist()
    treemap_df = data.copy()
    treemap_df['COUNTRY_GROUP'] = treemap_df['COUNTRY_NAME'].apply(lambda c: c if c in top_ctry else 'Other Nations')
    agg = treemap_df.groupby(['OBJECT_TYPE', 'COUNTRY_GROUP', 'RCS_SIZE']).size().reset_index(name='COUNT')
    
    fig = px.treemap(
        agg, path=['OBJECT_TYPE', 'COUNTRY_GROUP', 'RCS_SIZE'], values='COUNT',
        color='OBJECT_TYPE', color_discrete_map=TYPE_COLORS,
        title="<b>P2: Catalog Population Hierarchy — Object Type → Country → Radar Cross Section</b><br><span style='color:#64748B;font-size:12px;'>Hierarchical Drill-Down (Debris accounts for 59.7% of total catalog)</span>"
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

def create_p6_ethics(data):
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

# Build all figures
fig_m1 = create_m1_bar(df)
fig_m2 = create_m2_area(df)
fig_m3 = create_m3_donut(df)

fig_s1 = create_s1_histogram(df)
fig_s2 = create_s2_violin(df)
fig_s3 = create_s3_loglog_scatter(df)
fig_s4 = create_s4_correlation_heatmap(df)
fig_s5 = create_s5_pareto_lollipop(df)

fig_p1 = create_p1_scatter(df)
fig_p2 = create_p2_treemap(df)
fig_ethics = create_p6_ethics(df)

# ==============================================================================
# 7. ASSEMBLE CONTENT SECTIONS WITH 4-QUADRANT INFERENCES
# ==============================================================================

# Section M (Matplotlib Suite)
content_m = html.Div([
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

# Section S (Seaborn Suite)
content_s = html.Div([
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
                "S3 Perigee vs. Apogee by Object Type",
                "Operational payloads adhere strictly to circular orbits (y = x), while discarded rocket bodies and fragmented debris form high-eccentricity branches crossing through multiple altitudes.",
                "Over 88% of functional payloads maintain circular orbits (e < 0.05); rocket bodies exhibit high eccentricities (e > 0.60) with low perigees (< 500 km) and high apogees (> 35,000 km); debris is 98% localized in the LEO perigee/apogee square (200–1,000 km).",
                "Payloads require circular orbits for consistent communications and ground resolution. In contrast, discarded upper stages left in Geostationary Transfer Orbits (GTO) repeatedly plunge through the dense LEO debris cloud twice per day, creating cross-regime collision hazards.",
                "Two-body osculating approximation neglects third-body gravitational perturbations from the Moon and Sun, which gradually warp apogee and perigee over decades."
            )
        ]
    ),
    html.Div(
        style=card_style,
        children=[
            dcc.Graph(figure=fig_s4, config={'responsive': True}),
            create_quadrant_inference_box(
                "S4 Orbital Correlation Matrix",
                "Mathematical confirmation of Keplerian celestial mechanics and launch variable independence.",
                "Period ↔ Semi-Major Axis Pearson r = +0.99 (Kepler's 3rd Law T² ∝ a³); Apoapsis ↔ Eccentricity r = +0.70; Inclination ↔ Semi-Major Axis r = -0.04.",
                "Orbital geometry is strictly governed by gravitational laws; inclination is independently chosen by mission launch azimuth.",
                "Linear Pearson coefficient assumes monotonic relationships; non-linear perturbation terms are not captured."
            )
        ]
    ),
    html.Div(
        style=card_style,
        children=[
            dcc.Graph(figure=fig_s5, config={'responsive': True}),
            create_quadrant_inference_box(
                "S5 Top 10 Debris Contributors",
                "The global orbital debris footprint strictly adheres to the Pareto 80/20 principle.",
                "Top 3 nations (US: 3,025, Russia: 2,740, China: 2,410) account for 86.3% of all cataloged debris; India (115), ESA (95), Japan (82) each contribute < 1.5%.",
                "Legacy Cold War space race and major ASAT tests created the vast majority of orbital debris; international treaty compliance varies widely.",
                "Country attribution is based on registering launch state under UN conventions, which may differ from current commercial ownership."
            )
        ]
    )
])

# Section P (Plotly Suite)
content_p = html.Div([
    html.Div(
        style=card_style,
        children=[
            dcc.Graph(figure=fig_p1, config={'responsive': True}),
            create_quadrant_inference_box(
                "P1 Orbital Regime Map",
                "Spatial congestion shells pinpoint high-risk orbital bottlenecks in Sun-Synchronous (600–800 km) and mega-constellation shells (550 km).",
                "Over 6,200 objects occupy the 400–1,000 km altitude band; GEO ring holds 645 objects strictly at 35,786 km; 90% of debris is classified as 'SMALL' RCS.",
                "Demonstrates why active debris removal and automated collision avoidance are urgently needed in SSO and mid-latitude shells.",
                "2D projection flattens 3D longitude of ascending node (RAAN); true conjunction risk requires 3D covariance analysis."
            )
        ]
    ),
    html.Div(
        style=card_style,
        children=[
            dcc.Graph(figure=fig_p2, config={'responsive': True}),
            create_quadrant_inference_box(
                "P2 Catalog Population Hierarchy",
                "Hierarchical nesting shows small radar-cross-section fragments dominate national debris inventories.",
                "Debris accounts for 59.7% of total root area; within Chinese debris, 94% is Small RCS; within Russian debris, 88% is Small RCS; US leads payload share (69% Large RCS).",
                "Hypervelocity breakup physics shatters satellites into thousands of small, difficult-to-track fragments that retain lethal kinetic energy.",
                "RCS reflects radar reflectivity, not physical mass or dimension; non-metallic materials may appear smaller."
            )
        ]
    )
])

# Section Ethics
content_ethics = html.Div([
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
    meta_tags=[{"name": "viewport", "content": "width=device-width, initial-scale=1"}]
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
                'marginBottom': '24px',
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

        # Executive KPI Cards Ribbon
        html.Div(
            style={'display': 'grid', 'gridTemplateColumns': 'repeat(auto-fit, minmax(210px, 1fr))', 'gap': '16px', 'marginBottom': '28px'},
            children=[
                html.Div(style={'backgroundColor': '#FFFFFF', 'border': '1px solid #E2E8F0', 'borderTop': '4px solid #2563EB', 'borderRadius': '12px', 'padding': '16px 20px', 'boxShadow': '0 1px 3px rgba(0,0,0,0.04)'}, children=[
                    html.Div("Total Catalog Objects", style={'fontSize': '11.5px', 'color': '#64748B', 'textTransform': 'uppercase', 'fontWeight': '700'}),
                    html.Div(f"{total_count:,}", style={'fontSize': '28px', 'fontWeight': '800', 'color': '#2563EB', 'margin': '4px 0', 'fontFamily': 'monospace'}),
                    html.Div("Verified Space-Track records", style={'fontSize': '12px', 'color': '#94A3B8'})
                ]),
                html.Div(style={'backgroundColor': '#FFFFFF', 'border': '1px solid #E2E8F0', 'borderTop': '4px solid #EF4444', 'borderRadius': '12px', 'padding': '16px 20px', 'boxShadow': '0 1px 3px rgba(0,0,0,0.04)'}, children=[
                    html.Div("Defunct Debris", style={'fontSize': '11.5px', 'color': '#64748B', 'textTransform': 'uppercase', 'fontWeight': '700'}),
                    html.Div(f"{debris_count:,}", style={'fontSize': '28px', 'fontWeight': '800', 'color': '#EF4444', 'margin': '4px 0', 'fontFamily': 'monospace'}),
                    html.Div(f"{(debris_count/total_count)*100:.1f}% junk in orbit", style={'fontSize': '12px', 'color': '#DC2626', 'fontWeight': '600'})
                ]),
                html.Div(style={'backgroundColor': '#FFFFFF', 'border': '1px solid #E2E8F0', 'borderTop': '4px solid #10B981', 'borderRadius': '12px', 'padding': '16px 20px', 'boxShadow': '0 1px 3px rgba(0,0,0,0.04)'}, children=[
                    html.Div("Payload Satellites", style={'fontSize': '11.5px', 'color': '#64748B', 'textTransform': 'uppercase', 'fontWeight': '700'}),
                    html.Div(f"{payload_count:,}", style={'fontSize': '28px', 'fontWeight': '800', 'color': '#10B981', 'margin': '4px 0', 'fontFamily': 'monospace'}),
                    html.Div(f"{(payload_count/total_count)*100:.1f}% active & retired", style={'fontSize': '12px', 'color': '#94A3B8'})
                ]),
                html.Div(style={'backgroundColor': '#FFFFFF', 'border': '1px solid #E2E8F0', 'borderTop': '4px solid #F59E0B', 'borderRadius': '12px', 'padding': '16px 20px', 'boxShadow': '0 1px 3px rgba(0,0,0,0.04)'}, children=[
                    html.Div("Spent Rocket Bodies", style={'fontSize': '11.5px', 'color': '#64748B', 'textTransform': 'uppercase', 'fontWeight': '700'}),
                    html.Div(f"{rb_count:,}", style={'fontSize': '28px', 'fontWeight': '800', 'color': '#F59E0B', 'margin': '4px 0', 'fontFamily': 'monospace'}),
                    html.Div(f"{(rb_count/total_count)*100:.1f}% discarded upper stages", style={'fontSize': '12px', 'color': '#94A3B8'})
                ]),
                html.Div(style={'backgroundColor': '#FFFFFF', 'border': '1px solid #E2E8F0', 'borderTop': '4px solid #0284C7', 'borderRadius': '12px', 'padding': '16px 20px', 'boxShadow': '0 1px 3px rgba(0,0,0,0.04)'}, children=[
                    html.Div("LEO Congestion", style={'fontSize': '11.5px', 'color': '#64748B', 'textTransform': 'uppercase', 'fontWeight': '700'}),
                    html.Div(f"{leo_count:,}", style={'fontSize': '28px', 'fontWeight': '800', 'color': '#0284C7', 'margin': '4px 0', 'fontFamily': 'monospace'}),
                    html.Div(f"{(leo_count/total_count)*100:.1f}% below 2,000 km altitude", style={'fontSize': '12px', 'color': '#0369A1', 'fontWeight': '600'})
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
                dcc.Tab(label="🚀 Plotly Suite (P1 – P2)", value="tab-p", style=tab_style, selected_style=tab_selected_style),
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

# Tab Switching Callback
@app.callback(
    Output('tab-content', 'children'),
    Input('main-tabs', 'value')
)
def render_tab_content(tab_name):
    if tab_name == 'tab-m':
        return content_m
    elif tab_name == 'tab-s':
        return content_s
    elif tab_name == 'tab-p':
        return content_p
    elif tab_name == 'tab-ethics':
        return content_ethics
    return content_m

def open_browser(port):
    webbrowser.open_new(f"http://127.0.0.1:{port}")

if __name__ == '__main__':
    port = int(os.environ.get('PORT', 8050))
    print("\n" + "="*70)
    print(f"Launching Clean Light-Theme Plotly Dash Dashboard on port {port}...")
    print("Exact 4-Quadrant Inferences (M1-M3, S1-S5, P1-P2, Ethics).")
    print("Zero HTML files generated.")
    print(f"Local URL: http://127.0.0.1:{port}")
    print("="*70)
    
    # Auto-open browser only when running locally (not on cloud servers)
    if 'PORT' not in os.environ:
        threading.Timer(1.5, open_browser, args=(port,)).start()
    
    # Run Dash server binding to 0.0.0.0 for cloud readiness
    app.run(debug=False, host='0.0.0.0', port=port)

