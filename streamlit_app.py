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

def compute_sector_strengths_time_series(returns_df, window_units, window_type, signal_type='vol_adjusted'):
    """
    Compute sector strengths over time with weekly rolling windows for time-lapse animation
    """
    # Convert window to days
    if window_type == 'weeks':
        total_window_days = window_units * 7
    elif window_type == 'months':
        total_window_days = window_units * 30
    else:  # years
        total_window_days = window_units * 365
    
    # Get the full data window
    end_date = returns_df.index[-1]
    start_date = end_date - timedelta(days=total_window_days)
    
    # Filter data to window
    window_data = returns_df[returns_df.index >= start_date].copy()
    
    if len(window_data) < 14:  # Need at least 2 weeks
        return None, None, None
    
    # Create weekly periods for the time series
    weekly_data = window_data.resample('W')
    
    # Calculate rolling strengths over time
    weekly_dates = []
    weekly_strengths = []
    weekly_probabilities = []
    
    # Use a 4-week rolling window to compute strengths
    rolling_window_weeks = min(4, len(list(weekly_data)) // 2)
    
    all_weeks = list(weekly_data)
    for i in range(rolling_window_weeks, len(all_weeks)):
        # Get rolling window of weeks
        rolling_weeks = all_weeks[i-rolling_window_weeks:i]
        
        # Compute rankings for this rolling window
        rolling_rankings = []
        for date, week_data in rolling_weeks:
            if len(week_data) == 0:
                continue
                
            if signal_type == 'vol_adjusted':
                # Vol-adjusted returns for the week
                week_returns = week_data.sum()
                week_vol = week_data.std()
                signal = week_returns / (week_vol + 1e-8)
            else:  # simple returns
                signal = week_data.sum()
            
            # Winsorize extremes
            signal_winsorized = stats.mstats.winsorize(signal, limits=[0.01, 0.01])
            
            # Convert to rankings
            ranks = stats.rankdata(-signal_winsorized, method='ordinal')
            rolling_rankings.append(ranks)
        
        if len(rolling_rankings) < 2:
            continue
            
        # Create rankings matrix for this window
        rankings_matrix = pd.DataFrame(rolling_rankings, columns=returns_df.columns)
        
        # Compute strengths for this time point
        try:
            theta = plackett_luce_strength(rankings_matrix)
            prob = strengths_to_probabilities(theta, temperature=0.8)
            
            weekly_dates.append(all_weeks[i-1][0])  # Use the end date of the window
            weekly_strengths.append(theta)
            weekly_probabilities.append(prob)
        except:
            continue
    
    if len(weekly_strengths) < 2:
        return None, None, None
    
    return weekly_strengths, weekly_probabilities, weekly_dates

def compute_sector_strengths_over_time(returns_df, window_units, window_type, signal_type='vol_adjusted'):
    """
    Wrapper function to maintain compatibility - now returns time series data
    """
    weekly_strengths, weekly_probabilities, weekly_dates = compute_sector_strengths_time_series(
        returns_df, window_units, window_type, signal_type
    )
    
    if weekly_strengths is None:
        return None, None, None
    
    # Return first and last for compatibility
    theta_t0 = weekly_strengths[0]
    theta_t1 = weekly_strengths[-1]
    
    # Create a summary dataframe
    rankings_df = pd.DataFrame({
        'start_date': [weekly_dates[0]],
        'end_date': [weekly_dates[-1]],
        'n_periods': [len(weekly_dates)]
    })
    
    return theta_t0, theta_t1, rankings_df

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

def create_time_lapse_flow_visualization(returns_df, sector_names, window_units, window_type, signal_type, temperature=0.8, min_flow_threshold=0.02):
    """Create time-lapse animation showing week-over-week changes"""
    
    # Get time series data
    weekly_strengths, weekly_probabilities, weekly_dates = compute_sector_strengths_time_series(
        returns_df, window_units, window_type, signal_type
    )
    
    if weekly_strengths is None:
        return None
    
    n_sectors = len(sector_names)
    n_time_points = len(weekly_probabilities)
    
    # Create node positions in a circle
    angles = np.linspace(0, 2*np.pi, n_sectors, endpoint=False)
    radius = 5
    x_pos = radius * np.cos(angles)
    y_pos = radius * np.sin(angles)
    
    # Find max probability across all time points for consistent scaling
    all_probs = np.concatenate(weekly_probabilities)
    max_prob = np.max(all_probs)
    scale_factor = 60 / max_prob if max_prob > 0 else 1
    
    # Create smooth interpolated frames between weekly states
    animation_frames = []
    interpolation_steps = 8  # Number of smooth steps between each week
    
    for week_idx in range(len(weekly_probabilities)):
        # Current week data
        current_probs = weekly_probabilities[week_idx]
        current_sizes = current_probs * scale_factor + 20
        
        # Calculate colors based on trend
        if week_idx > 0:
            prev_probs = weekly_probabilities[week_idx - 1]
            changes = current_probs - prev_probs
            colors = [
                'red' if change < -0.01 else 
                'green' if change > 0.01 else 
                'blue' for change in changes
            ]
        else:
            colors = ['blue'] * n_sectors
        
        # If this is not the last week, create interpolated frames to next week
        if week_idx < len(weekly_probabilities) - 1:
            next_probs = weekly_probabilities[week_idx + 1]
            next_sizes = next_probs * scale_factor + 20
            
            # Calculate colors for the transition (based on where we're going)
            next_changes = next_probs - current_probs
            next_colors = [
                'red' if change < -0.01 else 
                'green' if change > 0.01 else 
                'blue' for change in next_changes
            ]
            
            # Create interpolated frames
            for step in range(interpolation_steps):
                t = step / interpolation_steps  # Interpolation factor (0 to 1)
                
                # Smooth interpolation between current and next states
                interpolated_sizes = (1 - t) * current_sizes + t * next_sizes
                
                # Gradually transition colors (use current colors for first half, next colors for second half)
                if t < 0.5:
                    frame_colors = colors
                else:
                    frame_colors = next_colors
                
                # Create frame
                frame_data = go.Scatter(
                    x=x_pos,
                    y=y_pos,
                    mode='markers+text',
                    marker=dict(
                        size=interpolated_sizes,
                        color=frame_colors,
                        opacity=0.7,
                        line=dict(width=2, color='black')
                    ),
                    text=sector_names,
                    textposition="middle center",
                    textfont=dict(size=12, color='white', family="Arial Black"),
                    name=f"Week {week_idx + 1}.{step + 1}"
                )
                animation_frames.append(frame_data)
        else:
            # Final week - just add the final frame
            frame_data = go.Scatter(
                x=x_pos,
                y=y_pos,
                mode='markers+text',
                marker=dict(
                    size=current_sizes,
                    color=colors,
                    opacity=0.7,
                    line=dict(width=2, color='black')
                ),
                text=sector_names,
                textposition="middle center",
                textfont=dict(size=12, color='white', family="Arial Black"),
                name=f"Week {week_idx + 1}"
            )
            animation_frames.append(frame_data)
    
    # Create the figure with animation
    fig = go.Figure(data=[animation_frames[0]])
    
    # Create animation frames
    frames = []
    for frame_idx, frame_data in enumerate(animation_frames):
        frames.append(go.Frame(
            data=[frame_data],
            name=str(frame_idx)
        ))
    
    fig.frames = frames
    
    # Add animation controls for time-lapse
    fig.update_layout(
        title=dict(
            text=f"📈 Time-Lapse: {window_units} {window_type.title()} Money Flow Evolution<br><sub>🔴 Declining | 🟢 Growing | 🔵 Stable (Week-over-Week)</sub>",
            x=0.5,
            font=dict(size=16)
        ),
        xaxis=dict(
            showgrid=False, 
            zeroline=False, 
            showticklabels=False,
            range=[-7, 7]
        ),
        yaxis=dict(
            showgrid=False, 
            zeroline=False, 
            showticklabels=False,
            range=[-7, 7],
            scaleanchor="x",
            scaleratio=1
        ),
        showlegend=False,
        height=800,
        margin=dict(l=50, r=50, t=100, b=100),
        updatemenus=[{
            "buttons": [
                {
                    "args": [None, {"frame": {"duration": 120, "redraw": True},
                                   "fromcurrent": True, "transition": {"duration": 80}}],
                    "label": "▶️ Play Smooth Time-Lapse",
                    "method": "animate"
                },
                {
                    "args": [None, {"frame": {"duration": 60, "redraw": True},
                                   "fromcurrent": True, "transition": {"duration": 40}}],
                    "label": "⚡ Play Fast",
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
                "font": {"size": 16},
                "prefix": "Week: ",
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
                    "args": [[f], {"frame": {"duration": 200, "redraw": True},
                                  "mode": "immediate", "transition": {"duration": 200}}],
                    "label": f"W{i//interpolation_steps + 1}" if i % interpolation_steps == 0 else "",
                    "method": "animate"
                } for i, f in enumerate([str(k) for k in range(len(animation_frames))])
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
    
    # Compact controls in a container
    with st.container():
        # Time window selection - more compact
        col1, col2, col3, col4 = st.columns([1, 1, 1.5, 1])
        
        with col1:
            window_units = st.number_input("Units", min_value=1, max_value=60, value=3)
        
        with col2:
            window_type = st.selectbox("Period", ["weeks", "months", "years"])
        
        with col3:
            signal_type = st.selectbox("Signal", 
                                      ["vol_adjusted", "simple_returns"],
                                      help="Vol-adjusted: return/volatility, Simple: total returns")
        
        with col4:
            visualize_btn = st.button("🚀 Visualize", type="primary", use_container_width=True)
        
        # Advanced parameters - collapsible for space
        with st.expander("⚙️ Advanced Parameters", expanded=False):
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
    
    # Visualize button (now moved to controls section)
    if visualize_btn:
        with st.spinner("Computing sector strengths and optimal flows..."):
            
            # Get returns data
            returns_data = analysis_data['returns_daily']
            
            # Main time-lapse flow visualization
            st.subheader("📈 Time-Lapse Money Flow Evolution")
            
            flow_fig = create_time_lapse_flow_visualization(
                returns_data, returns_data.columns, window_units, window_type, 
                signal_type, temperature, min_flow
            )
            if flow_fig:
                st.plotly_chart(flow_fig, use_container_width=True, height=800)
                
                st.markdown(f"""
                **📺 Smooth Time-Lapse Controls:**
                - ▶️ **Play Smooth**: Watch organic growth/shrinkage over {window_units} {window_type}
                - ⚡ **Play Fast**: Accelerated view for quick overview
                - 🔴 **Red nodes**: Declining trend vs previous week
                - 🟢 **Green nodes**: Growing trend vs previous week  
                - 🔵 **Blue nodes**: Stable trend vs previous week
                - **Node size**: Represents sector strength (larger = stronger)
                - **Smooth interpolation**: 8 steps between each weekly state
                - **Week slider**: Navigate through the timeline
                """)
                
                # Get time series data for summary
                weekly_strengths, weekly_probabilities, weekly_dates = compute_sector_strengths_time_series(
                    returns_data, window_units, window_type, signal_type
                )
                
                if weekly_strengths is not None:
                    # Summary with time series data
                    col1, col2 = st.columns([2, 1])
                    
                    with col1:
                        st.subheader("📊 Evolution Summary")
                        
                        # Create summary showing start vs end
                        start_probs = weekly_probabilities[0]
                        end_probs = weekly_probabilities[-1]
                        
                        evolution_df = pd.DataFrame({
                            'Sector': returns_data.columns,
                            'Week 1 (%)': (start_probs * 100).round(1),
                            f'Week {len(weekly_probabilities)} (%)': (end_probs * 100).round(1),
                            'Total Change (%)': ((end_probs - start_probs) * 100).round(1),
                            'Trend': ['🔴 Declined' if change < -1 else '🟢 Gained' if change > 1 else '🔵 Stable' 
                                     for change in (end_probs - start_probs) * 100]
                        })
                        
                        # Color code the table
                        def color_trend(val):
                            if '🔴' in str(val):
                                return 'background-color: #f8d7da'
                            elif '🟢' in str(val):
                                return 'background-color: #d4edda'
                            else:
                                return 'background-color: #e2e3e5'
                        
                        styled_evolution = evolution_df.style.applymap(color_trend, subset=['Trend'])
                        st.dataframe(styled_evolution, use_container_width=True, height=300)
                    
                    with col2:
                        st.subheader("📈 Time-Lapse Stats")
                        
                        # Calculate volatility of changes
                        all_changes = []
                        for i in range(1, len(weekly_probabilities)):
                            changes = weekly_probabilities[i] - weekly_probabilities[i-1]
                            all_changes.extend(np.abs(changes))
                        
                        avg_weekly_change = np.mean(all_changes) * 100
                        max_weekly_change = np.max(all_changes) * 100
                        
                        st.metric("Time Points", len(weekly_probabilities))
                        st.metric("Avg Weekly Change", f"{avg_weekly_change:.1f}%")
                        st.metric("Max Weekly Change", f"{max_weekly_change:.1f}%")

def main():
    st.set_page_config(
        page_title="SectorFlux Analytics",
        page_icon="📊",
        layout="wide",
        initial_sidebar_state="collapsed"  # Give more space to main content
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
