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

def compute_sector_strengths_time_series(returns_df, window_units, window_type, step_size, signal_type='vol_adjusted'):
    """
    Compute sector strengths over time with user-defined step size for time-lapse animation
    """
    # Convert window to days
    if window_type == 'weeks':
        total_window_days = window_units * 7
    elif window_type == 'months':
        total_window_days = window_units * 30
    else:  # years
        total_window_days = window_units * 365
    
    # Convert step size to days
    if step_size == '1 week':
        step_days = 7
        resample_freq = 'W'
    elif step_size == '2 weeks':
        step_days = 14
        resample_freq = '2W'
    elif step_size == '1 month':
        step_days = 30
        resample_freq = 'M'
    elif step_size == '3 months':
        step_days = 90
        resample_freq = '3M'
    else:  # 6 months
        step_days = 180
        resample_freq = '6M'
    
    # Get the full data window
    end_date = returns_df.index[-1]
    start_date = end_date - timedelta(days=total_window_days)
    
    # Filter data to window
    window_data = returns_df[returns_df.index >= start_date].copy()
    
    if len(window_data) < step_days:  # Need at least one step period
        return None, None, None
    
    # Create periods based on step size
    period_data = window_data.resample(resample_freq)
    
    # Calculate rolling strengths over time
    period_dates = []
    period_strengths = []
    period_probabilities = []
    
    # Use appropriate rolling window based on step size
    if step_days <= 14:  # Weekly/biweekly - use 4-period window
        rolling_window_periods = 4
    elif step_days <= 30:  # Monthly - use 3-period window
        rolling_window_periods = 3
    else:  # Quarterly/semi-annual - use 2-period window
        rolling_window_periods = 2
    
    all_periods = list(period_data)
    for i in range(rolling_window_periods, len(all_periods)):
        # Get rolling window of periods
        rolling_periods = all_periods[i-rolling_window_periods:i]
        
        # Compute rankings for this rolling window
        rolling_rankings = []
        for date, period_data_chunk in rolling_periods:
            if len(period_data_chunk) == 0:
                continue
                
            if signal_type == 'vol_adjusted':
                # Vol-adjusted returns for the period
                period_returns = period_data_chunk.sum()
                period_vol = period_data_chunk.std()
                signal = period_returns / (period_vol + 1e-8)
            else:  # simple returns
                signal = period_data_chunk.sum()
            
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
            
            period_dates.append(all_periods[i-1][0])  # Use the end date of the window
            period_strengths.append(theta)
            period_probabilities.append(prob)
        except:
            continue
    
    if len(period_strengths) < 2:
        return None, None, None
    
    return period_strengths, period_probabilities, period_dates

def compute_sector_strengths_over_time(returns_df, window_units, window_type, step_size, signal_type='vol_adjusted'):
    """
    Wrapper function to maintain compatibility - now returns time series data
    """
    period_strengths, period_probabilities, period_dates = compute_sector_strengths_time_series(
        returns_df, window_units, window_type, step_size, signal_type
    )
    
    if period_strengths is None:
        return None, None, None
    
    # Return first and last for compatibility
    theta_t0 = period_strengths[0]
    theta_t1 = period_strengths[-1]
    
    # Create a summary dataframe
    rankings_df = pd.DataFrame({
        'start_date': [period_dates[0]],
        'end_date': [period_dates[-1]],
        'n_periods': [len(period_dates)]
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

def validate_timeframe_step_combination(window_units, window_type, step_size):
    """Validate if the timeframe and step size combination will work"""
    # Convert window to days
    if window_type == 'weeks':
        total_window_days = window_units * 7
    elif window_type == 'months':
        total_window_days = window_units * 30
    else:  # years
        total_window_days = window_units * 365
    
    # Convert step size to days and get rolling window periods
    if step_size == '1 week':
        step_days = 7
        rolling_window_periods = 4
    elif step_size == '2 weeks':
        step_days = 14
        rolling_window_periods = 4
    elif step_size == '1 month':
        step_days = 30
        rolling_window_periods = 3
    elif step_size == '3 months':
        step_days = 90
        rolling_window_periods = 2
    else:  # 6 months
        step_days = 180
        rolling_window_periods = 2
    
    # Calculate periods needed
    max_possible_periods = total_window_days // step_days
    usable_periods = max(0, max_possible_periods - rolling_window_periods)
    
    # We need at least 2 usable periods for animation
    min_periods_needed = rolling_window_periods + 2  # rolling window + 2 usable periods
    min_required_days = min_periods_needed * step_days
    
    is_valid = usable_periods >= 2
    recommendation = None if is_valid else get_timeframe_recommendation(window_units, window_type, step_size, min_required_days)
    
    return {
        'is_valid': is_valid,
        'total_window_days': total_window_days,
        'min_required_days': min_required_days,
        'step_days': step_days,
        'rolling_window_periods': rolling_window_periods,
        'max_possible_periods': max_possible_periods,
        'usable_periods': usable_periods,
        'recommendation': recommendation
    }

def get_timeframe_recommendation(window_units, window_type, step_size, min_required_days):
    """Get recommendation for fixing timeframe/step issues"""
    if window_type == 'weeks':
        min_window_units = max(2, (min_required_days + 6) // 7)  # Round up
        return f"Try at least {min_window_units} weeks for {step_size} steps"
    elif window_type == 'months':
        min_window_units = max(2, (min_required_days + 29) // 30)  # Round up
        return f"Try at least {min_window_units} months for {step_size} steps"
    else:  # years
        min_window_units = max(1, (min_required_days + 364) // 365)  # Round up
        return f"Try at least {min_window_units} years for {step_size} steps"

def create_time_lapse_flow_visualization(returns_df, sector_names, window_units, window_type, step_size, signal_type, temperature=0.8, min_flow_threshold=0.02):
    """Create time-lapse animation showing period-over-period changes"""
    
    # Validate timeframe/step combination first
    validation = validate_timeframe_step_combination(window_units, window_type, step_size)
    if not validation['is_valid']:
        return None, validation
    
    # Get time series data
    period_strengths, period_probabilities, period_dates = compute_sector_strengths_time_series(
        returns_df, window_units, window_type, step_size, signal_type
    )
    
    if period_strengths is None:
        return None, validation
    
    n_sectors = len(sector_names)
    n_time_points = len(period_probabilities)
    
    # Create node positions in a circle
    angles = np.linspace(0, 2*np.pi, n_sectors, endpoint=False)
    radius = 5
    x_pos = radius * np.cos(angles)
    y_pos = radius * np.sin(angles)
    
    # Find max probability across all time points for consistent scaling
    all_probs = np.concatenate(period_probabilities)
    max_prob = np.max(all_probs)
    scale_factor = 60 / max_prob if max_prob > 0 else 1
    
    # Create smooth interpolated frames between period states
    animation_frames = []
    # More interpolation steps for smoother animation
    interpolation_steps = 12  # Number of smooth steps between each period
    
    for period_idx in range(len(period_probabilities)):
        # Current period data
        current_probs = period_probabilities[period_idx]
        current_sizes = current_probs * scale_factor + 20
        
        # Calculate colors based on trend
        if period_idx > 0:
            prev_probs = period_probabilities[period_idx - 1]
            changes = current_probs - prev_probs
            colors = [
                'red' if change < -0.01 else 
                'green' if change > 0.01 else 
                'blue' for change in changes
            ]
        else:
            colors = ['blue'] * n_sectors
        
        # If this is not the last period, create interpolated frames to next period
        if period_idx < len(period_probabilities) - 1:
            next_probs = period_probabilities[period_idx + 1]
            next_sizes = next_probs * scale_factor + 20
            
            # Calculate colors for the transition (based on where we're going)
            next_changes = next_probs - current_probs
            next_colors = [
                'red' if change < -0.01 else 
                'green' if change > 0.01 else 
                'blue' for change in next_changes
            ]
            
            # Create interpolated frames with easing
            for step in range(interpolation_steps):
                t = step / interpolation_steps  # Interpolation factor (0 to 1)
                
                # Apply easing function for more natural movement
                # Use ease-in-out cubic for smooth acceleration/deceleration
                eased_t = 3 * t**2 - 2 * t**3 if t < 1 else 1
                
                # Smooth interpolation between current and next states
                interpolated_sizes = (1 - eased_t) * current_sizes + eased_t * next_sizes
                
                # Gradually transition colors (use current colors for first third, transition in middle, next colors in last third)
                if t < 0.33:
                    frame_colors = colors
                elif t > 0.66:
                    frame_colors = next_colors
                else:
                    # Blend colors in the middle section
                    frame_colors = colors  # Keep it simple for now
                
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
                    name=f"Period {period_idx + 1}.{step + 1}"
                )
                animation_frames.append(frame_data)
        else:
            # Final period - just add the final frame
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
                name=f"Period {period_idx + 1}"
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
    
    # Create date lookup for hover functionality
    total_frames = len(animation_frames)
    frame_dates = []
    
    for frame_idx in range(total_frames):
        # Calculate which period we're in
        period_idx = frame_idx // interpolation_steps
        interpolation_progress = (frame_idx % interpolation_steps) / interpolation_steps
        
        # Get current and next period dates for interpolation
        if period_idx < len(period_dates):
            current_date = period_dates[period_idx]
            if period_idx + 1 < len(period_dates):
                next_date = period_dates[period_idx + 1]
                # Interpolate between current and next date
                total_days = (next_date - current_date).days
                interpolated_days = int(total_days * interpolation_progress)
                display_date = current_date + timedelta(days=interpolated_days)
            else:
                display_date = current_date
        else:
            display_date = period_dates[-1] if period_dates else date.today()
        
        frame_dates.append(display_date)
    
    # Add animation controls for time-lapse
    fig.update_layout(
        title=dict(
            text=f"📈 Time-Lapse: {window_units} {window_type.title()} Money Flow Evolution<br><sub>🔴 Declining | 🟢 Growing | 🔵 Stable ({step_size} steps)</sub>",
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
                    "args": [None, {"frame": {"duration": 80, "redraw": True},
                                   "fromcurrent": True, "transition": {"duration": 60}}],
                    "label": "▶️ Play Smooth",
                    "method": "animate"
                },
                {
                    "args": [None, {"frame": {"duration": 40, "redraw": True},
                                   "fromcurrent": True, "transition": {"duration": 30}}],
                    "label": "⚡ Play Fast",
                    "method": "animate"
                },
                {
                    "args": [None, {"frame": {"duration": 20, "redraw": True},
                                   "fromcurrent": True, "transition": {"duration": 15}}],
                    "label": "🚀 Play Ultra-Fast",
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
                "font": {"size": 14},
                "prefix": "Date: ",
                "visible": True,
                "xanchor": "right"
            },
            "transition": {"duration": 50, "easing": "linear"},
            "pad": {"b": 10, "t": 50},
            "len": 0.9,
            "x": 0.1,
            "y": 0,
            "steps": [
                {
                    "args": [[str(frame_idx)], {
                        "frame": {"duration": 0, "redraw": True},
                        "mode": "immediate", 
                        "transition": {"duration": 0}
                    }],
                    "label": "",  # No labels/dividers
                    "method": "animate",
                    "value": frame_idx / (total_frames - 1) if total_frames > 1 else 0  # Normalized position
                } for frame_idx in range(total_frames)
            ]
        }]
    )
    
    return fig, validation

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
        col1, col2, col3, col4, col5 = st.columns([1, 1, 1, 1.5, 1])
        
        with col1:
            window_units = st.number_input("Units", min_value=1, max_value=60, value=3)
        
        with col2:
            window_type = st.selectbox("Period", ["weeks", "months", "years"])
        
        with col3:
            step_size = st.selectbox("Step Size", 
                                   ["1 week", "2 weeks", "1 month", "3 months", "6 months"],
                                   index=0,
                                   help="How often to calculate new positions")
        
        with col4:
            signal_type = st.selectbox("Signal", 
                                      ["vol_adjusted", "simple_returns"],
                                      help="Vol-adjusted: return/volatility, Simple: total returns")
        
        with col5:
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
            
            # Validate timeframe/step combination first
            validation = validate_timeframe_step_combination(window_units, window_type, step_size)
            
            if not validation['is_valid']:
                st.error(f"""
                ❌ **Insufficient data for this timeframe/step combination**
                
                **Issue**: {window_units} {window_type} with {step_size} steps doesn't provide enough periods for analysis.
                
                **Details**:
                - Total window: {validation['total_window_days']} days ({window_units} {window_type})
                - Step size: {validation['step_days']} days ({step_size})
                - Rolling window needed: {validation['rolling_window_periods']} periods
                - Max possible periods: {validation['max_possible_periods']} 
                - Usable periods: {validation['usable_periods']} (need at least 2)
                - Minimum required: {validation['min_required_days']} days for this combination
                
                **💡 Recommendation**: {validation['recommendation']}
                
                **Alternative fixes**:
                - Increase timeframe (more {window_type})
                - Use smaller step size (e.g., "1 week" or "2 weeks")
                - Switch to longer period (e.g., months → years)
                """)
                
                # Show helpful suggestions
                st.info("""
                **Quick fixes**:
                - 📅 **8 weeks + 1 week steps** → Works great!
                - 📅 **3 months + 2 weeks steps** → Perfect balance
                - 📅 **6 months + 1 month steps** → Good for trends
                - 📅 **2 years + 3 months steps** → Long-term view
                """)
                return
            
            # Main time-lapse flow visualization
            st.subheader("📈 Time-Lapse Money Flow Evolution")
            st.success(f"✅ {validation['usable_periods']} periods available for analysis ({step_size} steps over {window_units} {window_type})")
            
            # Get time series data for date controls
            period_strengths, period_probabilities, period_dates = compute_sector_strengths_time_series(
                returns_data, window_units, window_type, step_size, signal_type
            )
            
            flow_fig, _ = create_time_lapse_flow_visualization(
                returns_data, returns_data.columns, window_units, window_type, 
                step_size, signal_type, temperature, min_flow
            )
            if flow_fig:
                # Add play-between-points controls
                col1, col2, col3 = st.columns([1, 1, 2])
                
                with col1:
                    start_date = st.date_input(
                        "Start Date",
                        value=period_dates[0] if period_dates else date.today(),
                        help="Set start point for animation segment"
                    )
                
                with col2:
                    end_date = st.date_input(
                        "End Date", 
                        value=period_dates[-1] if period_dates else date.today(),
                        help="Set end point for animation segment"
                    )
                
                with col3:
                    if st.button("🎬 Play Between Selected Dates", help="Play animation only between the selected date range"):
                        st.info(f"Playing from {start_date} to {end_date}")
                
                st.plotly_chart(flow_fig, use_container_width=True, height=800)
                
                st.markdown(f"""
                **📺 Clean Time-Lapse Controls:**
                - ▶️ **Play Smooth**: Organic growth/shrinkage over {window_units} {window_type} ({step_size} steps)
                - ⚡ **Play Fast**: Accelerated view for quick overview
                - 🚀 **Play Ultra-Fast**: Lightning-fast scan of the entire period
                - 🎬 **Play Between Dates**: Set custom start/end points above
                - 🔴 **Red nodes**: Declining trend vs previous period
                - 🟢 **Green nodes**: Growing trend vs previous period  
                - 🔵 **Blue nodes**: Stable trend vs previous period
                - **Node size**: Represents sector strength (larger = stronger)
                - **Clean slider**: No dividers, hover shows exact dates
                - **Date selection**: Use date pickers above to analyze specific periods
                """)
                
                # Data already loaded above for date controls
                
                if period_strengths is not None:
                    # Summary with time series data
                    col1, col2 = st.columns([2, 1])
                    
                    with col1:
                        st.subheader("📊 Evolution Summary")
                        
                        # Create summary showing start vs end
                        start_probs = period_probabilities[0]
                        end_probs = period_probabilities[-1]
                        
                        evolution_df = pd.DataFrame({
                            'Sector': returns_data.columns,
                            'Start (%)': (start_probs * 100).round(1),
                            'End (%)': (end_probs * 100).round(1),
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
                        for i in range(1, len(period_probabilities)):
                            changes = period_probabilities[i] - period_probabilities[i-1]
                            all_changes.extend(np.abs(changes))
                        
                        avg_period_change = np.mean(all_changes) * 100 if all_changes else 0
                        max_period_change = np.max(all_changes) * 100 if all_changes else 0
                        
                        st.metric("Time Points", len(period_probabilities))
                        st.metric(f"Avg {step_size} Change", f"{avg_period_change:.1f}%")
                        st.metric(f"Max {step_size} Change", f"{max_period_change:.1f}%")

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
