import streamlit as st
import pandas as pd
import requests
import json
from datetime import datetime, date, timedelta
import plotly.express as px
import plotly.graph_objects as go
from io import BytesIO
import gzip
import numpy as np
from scipy import stats
from sklearn.preprocessing import StandardScaler
import ot  # POT library for optimal transport
from scipy.optimize import minimize

# Configuration
GITHUB_REPO = "adityarai7297/Flowtation"
BASE_URL = f"https://github.com/{GITHUB_REPO}/releases/download/data-latest"

@st.cache_data(ttl=3600)  # Cache for 1 hour
def download_data():
    """Download and cache data from GitHub releases"""
    try:
        # Download metadata
        metadata_url = f"{BASE_URL}/metadata.json"
        metadata_response = requests.get(metadata_url)
        metadata_response.raise_for_status()
        metadata = metadata_response.json()
        
        # Download price data (using CSV.gz for web-friendly format)
        data_url = f"{BASE_URL}/prices_5y.csv.gz"
        data_response = requests.get(data_url)
        data_response.raise_for_status()
        
        # Decompress and read CSV
        with gzip.open(BytesIO(data_response.content), 'rt') as f:
            df = pd.read_csv(f)
        
        # Convert date column
        df['date'] = pd.to_datetime(df['date'])
        
        return df, metadata
        
    except Exception as e:
        st.error(f"Failed to download data: {str(e)}")
        return None, None

def prepare_data_for_analysis(df):
    """Prepare data for rank algorithms and analysis"""
    if df is None:
        return None
    
    # Create analysis-ready datasets
    analysis_data = {}
    
    # 1. Daily returns
    df_pivot = df.pivot(index='date', columns='ticker', values='adj_close')
    returns_daily = df_pivot.pct_change().dropna()
    analysis_data['returns_daily'] = returns_daily
    
    # 2. Weekly returns (for momentum calculations)
    returns_weekly = df_pivot.resample('W').last().pct_change().dropna()
    analysis_data['returns_weekly'] = returns_weekly
    
    # 3. Monthly returns
    returns_monthly = df_pivot.resample('M').last().pct_change().dropna()
    analysis_data['returns_monthly'] = returns_monthly
    
    # 4. Price data (for trend analysis)
    analysis_data['prices'] = df_pivot
    
    # 5. Volume data
    volume_pivot = df.pivot(index='date', columns='ticker', values='volume')
    analysis_data['volume'] = volume_pivot
    
    # 6. Recent performance metrics (ready for ranking)
    latest_prices = df_pivot.iloc[-1]
    analysis_data['latest_prices'] = latest_prices
    
    # Performance periods for ranking
    periods = {'1M': 21, '3M': 63, '6M': 126, '1Y': 252}
    performance = {}
    
    for period_name, days in periods.items():
        if len(df_pivot) >= days:
            period_return = (df_pivot.iloc[-1] / df_pivot.iloc[-days] - 1) * 100
            performance[period_name] = period_return.sort_values(ascending=False)
    
    analysis_data['performance_periods'] = performance
    
    # 7. Volatility metrics (for risk-adjusted rankings)
    volatility_periods = {
        '1M': returns_daily.tail(21).std() * (252**0.5) * 100,
        '3M': returns_daily.tail(63).std() * (252**0.5) * 100,
        '1Y': returns_daily.tail(252).std() * (252**0.5) * 100
    }
    analysis_data['volatility'] = volatility_periods
    
    return analysis_data



def plackett_luce_strength(rankings_matrix, max_iter=100, tol=1e-6):
    """
    Fit Plackett-Luce model to get latent sector strengths from rankings
    
    Args:
        rankings_matrix: DataFrame with sectors as columns, subperiods as rows
                        Values are ranks (1=best, higher=worse)
    """
    n_sectors = rankings_matrix.shape[1]
    n_periods = rankings_matrix.shape[0]
    
    # Initialize strengths uniformly
    theta = np.ones(n_sectors)
    
    for iteration in range(max_iter):
        theta_old = theta.copy()
        
        # E-step: compute expected counts
        numerator = np.zeros(n_sectors)
        denominator = np.zeros(n_sectors)
        
        for period in range(n_periods):
            ranks = rankings_matrix.iloc[period].values
            if np.any(np.isnan(ranks)):
                continue
                
            # Sort sectors by rank for this period
            sorted_indices = np.argsort(ranks)
            
            for pos, sector_idx in enumerate(sorted_indices):
                if np.isnan(ranks[sector_idx]):
                    continue
                    
                # Numerator: this sector was chosen at this position
                numerator[sector_idx] += 1
                
                # Denominator: sum of strengths of all sectors still available
                remaining_sectors = sorted_indices[pos:]
                remaining_strengths = theta[remaining_sectors]
                denominator[sector_idx] += np.sum(remaining_strengths) / theta[sector_idx]
        
        # M-step: update strengths
        with np.errstate(divide='ignore', invalid='ignore'):
            theta = numerator / denominator
            theta = np.nan_to_num(theta, nan=1e-8, posinf=1e8, neginf=1e-8)
        
        # Normalize to prevent numerical issues
        theta = theta / np.sum(theta) * n_sectors
        
        # Check convergence
        if np.allclose(theta, theta_old, rtol=tol):
            break
    
    return theta

def compute_correlation_costs(returns_df, window_years=3):
    """Compute correlation-based cost matrix for optimal transport"""
    # Use recent data for correlation computation
    recent_returns = returns_df.tail(int(252 * window_years))  # Last N years
    
    # Compute correlation matrix
    corr_matrix = recent_returns.corr()
    
    # Convert to cost matrix (distance = 1 - correlation)
    cost_matrix = 1 - corr_matrix.values
    
    # Ensure positive costs and handle NaN
    cost_matrix = np.maximum(cost_matrix, 0)
    cost_matrix = np.nan_to_num(cost_matrix, nan=1.0)
    
    # Make symmetric and ensure diagonal is 0
    cost_matrix = (cost_matrix + cost_matrix.T) / 2
    np.fill_diagonal(cost_matrix, 0)
    
    return cost_matrix, corr_matrix

def compute_sector_strengths_over_time(returns_df, window_units, window_type, signal_type='vol_adjusted'):
    """
    Compute sector strengths at t0 and t1 using Plackett-Luce model
    """
    # Convert window to days
    if window_type == 'weeks':
        window_days = window_units * 7
        subperiod_days = 1 if window_units <= 4 else 5  # daily vs weekly
    elif window_type == 'months':
        window_days = window_units * 30
        subperiod_days = 1 if window_units <= 1 else 5
    else:  # years
        window_days = window_units * 365
        subperiod_days = 5  # always weekly for years
    
    # Get the window of data
    end_date = returns_df.index[-1]
    start_date = end_date - timedelta(days=window_days)
    
    # Filter data to window
    window_data = returns_df[returns_df.index >= start_date].copy()
    
    if len(window_data) < 10:  # Not enough data
        return None, None, None
    
    # Create subperiods
    if subperiod_days == 1:  # Daily subperiods
        subperiods = window_data.groupby(window_data.index.date)
    else:  # Weekly subperiods
        subperiods = window_data.resample('W')
    
    # Compute rankings for each subperiod
    rankings_list = []
    subperiod_dates = []
    
    for date, period_data in subperiods:
        if len(period_data) == 0:
            continue
            
        if signal_type == 'vol_adjusted':
            # Vol-adjusted returns
            period_returns = period_data.sum()  # Sum over the subperiod
            period_vol = period_data.std()
            signal = period_returns / (period_vol + 1e-8)  # Add small epsilon
        else:  # simple returns
            signal = period_data.sum()  # Total log returns
        
        # Winsorize extremes (1-2%)
        signal_winsorized = stats.mstats.winsorize(signal, limits=[0.01, 0.01])
        
        # Convert to rankings (1 = best performer)
        ranks = stats.rankdata(-signal_winsorized, method='ordinal')
        
        rankings_list.append(ranks)
        subperiod_dates.append(date)
    
    if len(rankings_list) < 2:
        return None, None, None
    
    # Create rankings matrix
    rankings_df = pd.DataFrame(rankings_list, 
                              index=subperiod_dates,
                              columns=returns_df.columns)
    
    # Compute strengths at t0 (early window) and t1 (late window)
    mid_point = len(rankings_df) // 2
    
    rankings_t0 = rankings_df.iloc[:mid_point]
    rankings_t1 = rankings_df.iloc[mid_point:]
    
    # Fit Plackett-Luce models
    try:
        theta_t0 = plackett_luce_strength(rankings_t0)
        theta_t1 = plackett_luce_strength(rankings_t1)
        
        return theta_t0, theta_t1, rankings_df
    except:
        return None, None, None

def strengths_to_probabilities(theta, temperature=0.8):
    """Convert strengths to probabilities using temperature scaling"""
    exp_theta = np.exp(theta / temperature)
    return exp_theta / np.sum(exp_theta)

def compute_optimal_transport_flow(p0, p1, cost_matrix, epsilon=0.01):
    """Compute optimal transport flow using Sinkhorn algorithm"""
    try:
        # Ensure probabilities sum to 1
        p0 = p0 / np.sum(p0)
        p1 = p1 / np.sum(p1)
        
        # Compute optimal transport matrix
        T_optimal = ot.sinkhorn(p0, p1, cost_matrix, epsilon, numItermax=1000)
        
        return T_optimal
    except:
        # Fallback to uniform transport if optimization fails
        n = len(p0)
        return np.outer(p0, p1)

def create_animated_flow_visualization(flow_matrix, sector_names, p0, p1, min_flow_threshold=0.02):
    """Create animated flow visualization with resizing nodes"""
    import time
    
    n_sectors = len(sector_names)
    
    # Calculate net flows (inflow - outflow) for each sector
    outflows = np.sum(flow_matrix, axis=1)  # Money leaving each sector
    inflows = np.sum(flow_matrix, axis=0)   # Money entering each sector
    net_flows = inflows - outflows
    
    # Create node positions in a circle
    angles = np.linspace(0, 2*np.pi, n_sectors, endpoint=False)
    radius = 3
    x_pos = radius * np.cos(angles)
    y_pos = radius * np.sin(angles)
    
    # Calculate sizes - initial (p0) and final (p1) probabilities
    initial_sizes = p0 * 1000 + 100  # Scale for visibility, minimum size
    final_sizes = p1 * 1000 + 100
    
    # Create frames for animation
    n_frames = 30
    animation_frames = []
    
    for frame in range(n_frames + 1):
        # Interpolate between initial and final sizes
        t = frame / n_frames
        current_sizes = (1 - t) * initial_sizes + t * final_sizes
        
        # Create scatter plot for this frame
        frame_data = go.Scatter(
            x=x_pos,
            y=y_pos,
            mode='markers+text',
            marker=dict(
                size=current_sizes,
                color=[
                    'red' if net_flow < -0.02 else 
                    'green' if net_flow > 0.02 else 
                    'blue' for net_flow in net_flows
                ],
                opacity=0.7,
                line=dict(width=2, color='black')
            ),
            text=sector_names,
            textposition="middle center",
            textfont=dict(size=10, color='white'),
            name=f"Frame {frame}"
        )
        animation_frames.append(frame_data)
    
    # Create the figure with animation
    fig = go.Figure(data=[animation_frames[0]])
    
    # Add flow arrows for significant flows
    arrow_annotations = []
    for i in range(n_sectors):
        for j in range(n_sectors):
            if i != j and flow_matrix[i, j] > min_flow_threshold:
                # Calculate arrow position
                arrow_x = [x_pos[i], x_pos[j]]
                arrow_y = [y_pos[i], y_pos[j]]
                
                # Add arrow line
                fig.add_trace(go.Scatter(
                    x=arrow_x,
                    y=arrow_y,
                    mode='lines',
                    line=dict(
                        width=flow_matrix[i, j] * 50,  # Scale arrow thickness
                        color='rgba(135, 206, 250, 0.6)'
                    ),
                    showlegend=False,
                    hoverinfo='skip'
                ))
                
                # Add arrowhead annotation
                arrow_annotations.append(
                    dict(
                        x=x_pos[j],
                        y=y_pos[j],
                        ax=x_pos[i],
                        ay=y_pos[i],
                        arrowhead=3,
                        arrowsize=2,
                        arrowwidth=2,
                        arrowcolor='rgba(135, 206, 250, 0.8)',
                        showarrow=True
                    )
                )
    
    # Create animation frames
    frames = []
    for frame_idx, frame_data in enumerate(animation_frames):
        frames.append(go.Frame(
            data=[frame_data] + [trace for trace in fig.data[1:]],  # Keep arrows
            name=str(frame_idx)
        ))
    
    fig.frames = frames
    
    # Add animation controls
    fig.update_layout(
        title=dict(
            text="💰 Money Flow Animation<br><sub>🔴 Shrinking (Outflow) | 🟢 Growing (Inflow) | 🔵 Neutral</sub>",
            x=0.5
        ),
        xaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
        yaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
        showlegend=False,
        height=600,
        annotations=arrow_annotations,
        updatemenus=[{
            "buttons": [
                {
                    "args": [None, {"frame": {"duration": 100, "redraw": True},
                                   "fromcurrent": True, "transition": {"duration": 50}}],
                    "label": "▶️ Play",
                    "method": "animate"
                },
                {
                    "args": [[None], {"frame": {"duration": 0, "redraw": True},
                                     "mode": "immediate", "transition": {"duration": 0}}],
                    "label": "⏸️ Pause",
                    "method": "animate"
                }
            ],
            "direction": "left",
            "pad": {"r": 10, "t": 87},
            "showactive": False,
            "type": "buttons",
            "x": 0.1,
            "xanchor": "right",
            "y": 0,
            "yanchor": "top"
        }],
        sliders=[{
            "active": 0,
            "yanchor": "top",
            "xanchor": "left",
            "currentvalue": {
                "font": {"size": 20},
                "prefix": "Time: ",
                "visible": True,
                "xanchor": "right"
            },
            "transition": {"duration": 300, "easing": "cubic-in-out"},
            "pad": {"b": 10, "t": 50},
            "len": 0.9,
            "x": 0.1,
            "y": 0,
            "steps": [
                {
                    "args": [[f], {"frame": {"duration": 300, "redraw": True},
                                  "mode": "immediate", "transition": {"duration": 300}}],
                    "label": f"T+{i}",
                    "method": "animate"
                } for i, f in enumerate([str(k) for k in range(n_frames + 1)])
            ]
        }]
    )
    
    return fig

def create_flow_summary_table(flow_matrix, sector_names, p0, p1):
    """Create a summary table of the flows"""
    outflows = np.sum(flow_matrix, axis=1)
    inflows = np.sum(flow_matrix, axis=0)
    net_flows = inflows - outflows
    
    summary_df = pd.DataFrame({
        'Sector': sector_names,
        'Initial Size (%)': (p0 * 100).round(1),
        'Final Size (%)': (p1 * 100).round(1),
        'Size Change': ((p1 - p0) * 100).round(1),
        'Net Flow': (net_flows * 100).round(1),
        'Status': ['🔴 Shrinking' if nf < -0.02 else '🟢 Growing' if nf > 0.02 else '🔵 Neutral' 
                  for nf in net_flows]
    })
    
    return summary_df.sort_values('Size Change', ascending=False)

def money_flow_interface(analysis_data):
    """Money flow visualization interface"""
    st.header("💰 Money Flow Visualization")
    st.markdown("*Visualize capital flows between sectors using Plackett-Luce rankings and optimal transport*")
    
    # Time window selection
    col1, col2, col3 = st.columns([1, 1, 2])
    
    with col1:
        window_units = st.number_input("Number of units", min_value=1, max_value=60, value=3)
    
    with col2:
        window_type = st.selectbox("Time unit", ["weeks", "months", "years"])
    
    with col3:
        signal_type = st.selectbox("Signal type", 
                                  ["vol_adjusted", "simple_returns"],
                                  help="Vol-adjusted: return/volatility, Simple: total returns")
    
    # Advanced parameters
    with st.expander("Advanced Parameters"):
        col1, col2, col3 = st.columns(3)
        with col1:
            temperature = st.slider("Temperature (τ)", 0.1, 2.0, 0.8, 0.1,
                                   help="Controls concentration of probabilities")
        with col2:
            epsilon = st.slider("Smoothness (ε)", 0.001, 0.1, 0.01, 0.001,
                               help="Regularization for optimal transport")
        with col3:
            min_flow = st.slider("Min flow threshold", 0.01, 0.1, 0.02, 0.01,
                                help="Minimum flow to display in visualization")
    
    # Visualize button
    if st.button("🚀 Visualize Money Flows", type="primary"):
        with st.spinner("Computing sector strengths and optimal flows..."):
            
            # Get returns data
            returns_data = analysis_data['returns_daily']
            
            # Compute sector strengths
            theta_t0, theta_t1, rankings_df = compute_sector_strengths_over_time(
                returns_data, window_units, window_type, signal_type
            )
            
            if theta_t0 is None:
                st.error("❌ Insufficient data for the selected time window. Try a shorter period.")
                return
            
            # Convert to probabilities
            p0 = strengths_to_probabilities(theta_t0, temperature)
            p1 = strengths_to_probabilities(theta_t1, temperature)
            
            # Compute cost matrix
            cost_matrix, corr_matrix = compute_correlation_costs(returns_data)
            
            # Compute optimal transport flow
            flow_matrix = compute_optimal_transport_flow(p0, p1, cost_matrix, epsilon)
            
            # Main animated flow visualization
            st.subheader("🎬 Animated Money Flow")
            
            flow_fig = create_animated_flow_visualization(
                flow_matrix, returns_data.columns, p0, p1, min_flow
            )
            if flow_fig:
                st.plotly_chart(flow_fig, use_container_width=True)
                
                st.markdown("""
                **How to use the animation:**
                - ▶️ **Play**: Watch sectors grow/shrink based on money flows
                - 🔴 **Red nodes**: Money flowing OUT (shrinking)
                - 🟢 **Green nodes**: Money flowing IN (growing)  
                - 🔵 **Blue nodes**: Neutral/balanced flows
                - **Arrows**: Show direction and magnitude of flows
                - **Slider**: Scrub through time manually
                """)
                
                # Flow summary table
                st.subheader("📊 Flow Summary")
                summary_df = create_flow_summary_table(flow_matrix, returns_data.columns, p0, p1)
                
                # Color code the table
                def color_status(val):
                    if '🔴' in str(val):
                        return 'background-color: #f8d7da'
                    elif '🟢' in str(val):
                        return 'background-color: #d4edda'
                    else:
                        return 'background-color: #e2e3e5'
                
                styled_summary = summary_df.style.applymap(color_status, subset=['Status'])
                st.dataframe(styled_summary, use_container_width=True)
                
                # Quick metrics
                col1, col2, col3 = st.columns(3)
                total_flow = np.sum(flow_matrix) * 100
                max_flow = np.max(flow_matrix) * 100
                n_significant_flows = np.sum(flow_matrix > min_flow)
                
                with col1:
                    st.metric("Total Flow", f"{total_flow:.1f}%")
                with col2:
                    st.metric("Largest Single Flow", f"{max_flow:.1f}%")
                with col3:
                    st.metric("Significant Flows", n_significant_flows)

def main():
    st.set_page_config(
        page_title="SectorFlux Analytics",
        page_icon="📊",
        layout="wide"
    )
    
    st.title("📊 SectorFlux Analytics")
    st.markdown("*Real-time ETF sector analysis powered by automated data pipeline*")
    
    # Download data
    with st.spinner("🔄 Loading latest market data..."):
        df, metadata = download_data()
    
    if df is None:
        st.error("❌ Failed to load data. Please check your internet connection and try again.")
        st.stop()
    
    # Prepare data for analysis
    with st.spinner("🔧 Preparing data for analysis..."):
        analysis_data = prepare_data_for_analysis(df)
    
    if analysis_data is None:
        st.error("❌ Failed to process data.")
        st.stop()
    
    # Quick data summary
    col1, col2, col3 = st.columns(3)
    with col1:
        st.metric("Tickers", len(metadata['tickers']))
    with col2:
        st.metric("Records", f"{metadata['rows']:,}")
    with col3:
        data_age = (datetime.now().date() - pd.to_datetime(metadata['date_max']).date()).days
        st.metric("Data Age", f"{data_age} days")
    
    # Money flow visualization (main interface)
    money_flow_interface(analysis_data)

if __name__ == "__main__":
    main()
