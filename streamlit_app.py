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
    """Download data from GitHub releases"""
    try:
        # Download prices
        prices_url = f"{BASE_URL}/prices_5y.csv.gz"
        prices_response = requests.get(prices_url)
        prices_response.raise_for_status()
        
        # Read compressed CSV
        prices_df = pd.read_csv(BytesIO(prices_response.content), compression='gzip', index_col=0, parse_dates=True)
        
        # Download metadata
        metadata_url = f"{BASE_URL}/metadata.json"
        metadata_response = requests.get(metadata_url)
        metadata_response.raise_for_status()
        metadata = metadata_response.json()
        
        return prices_df, metadata
        
    except Exception as e:
        st.error(f"Failed to download data: {e}")
        return None, None

def prepare_data_for_analysis(df):
    """Prepare data for analysis"""
    # Forward fill missing values
    df = df.fillna(method='ffill')
    
    # Calculate returns
    returns = df.pct_change().dropna()
    
    # Calculate basic metrics
    total_return = (df.iloc[-1] / df.iloc[0] - 1) * 100
    volatility = returns.std() * np.sqrt(252) * 100  # Annualized volatility
    
    # Recent performance (last 30 days)
    recent_return = (df.iloc[-1] / df.iloc[-30] - 1) * 100 if len(df) >= 30 else total_return
    
    analysis_data = {
        'prices': df,
        'returns': returns,
        'total_return': total_return,
        'volatility': volatility,
        'recent_return': recent_return,
        'start_date': df.index[0],
        'end_date': df.index[-1],
        'total_days': len(df)
    }
    
    return analysis_data

def plackett_luce_strength(rankings_matrix, max_iter=100, tol=1e-6):
    """Custom Plackett-Luce implementation"""
    n_items = rankings_matrix.shape[1]
    n_rankings = rankings_matrix.shape[0]
    
    # Initialize strengths uniformly
    theta = np.ones(n_items)
    
    for _ in range(max_iter):
        theta_old = theta.copy()
        
        # Update each strength
        for i in range(n_items):
            numerator = 0
            denominator = 0
            
            for ranking_idx in range(n_rankings):
                ranking = rankings_matrix[ranking_idx]
                # Find position of item i in this ranking
                try:
                    pos_i = np.where(ranking == i)[0][0]
                    
                    # Add contribution from this ranking
                    numerator += 1
                    
                    # Calculate denominator: sum of strengths of items ranked >= i
                    remaining_items = ranking[pos_i:]
                    denominator += 1 / np.sum(theta[remaining_items])
                    
                except IndexError:
                    continue
            
            if denominator > 0:
                theta[i] = numerator / denominator
        
        # Normalize to prevent overflow
        theta = theta / np.sum(theta) * n_items
        
        # Check convergence
        if np.linalg.norm(theta - theta_old) < tol:
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
    """Compute sector strengths over time using rolling windows"""
    # Convert step size to days
    step_map = {
        "1 week": 7, "2 weeks": 14, "1 month": 30, 
        "3 months": 90, "6 months": 180
    }
    step_days = step_map.get(step_size, 7)
    
    # Convert window to days
    if window_type == "weeks":
        window_days = window_units * 7
    elif window_type == "months":
        window_days = window_units * 30
    elif window_type == "years":
        window_days = window_units * 365
    else:
        window_days = window_units * 7
    
    # Rolling window size for ranking (3 periods for sufficient data)
    rolling_window_periods = 3
    rolling_window_days = rolling_window_periods * step_days
    
    # Generate time points
    start_date = returns_df.index[0] + timedelta(days=rolling_window_days)
    end_date = returns_df.index[-1]
    
    time_points = []
    current_date = start_date
    while current_date <= end_date:
        time_points.append(current_date)
        current_date += timedelta(days=step_days)
    
    if len(time_points) < 2:
        return None, None, None
    
    period_strengths = []
    period_probabilities = []
    period_dates = []
    
    for time_point in time_points:
        # Get rolling window of returns ending at this time point
        window_start = time_point - timedelta(days=rolling_window_days)
        window_returns = returns_df.loc[window_start:time_point]
        
        if len(window_returns) < 10:  # Need minimum data
            continue
        
        # Calculate signals for ranking
        if signal_type == 'vol_adjusted':
            # Volatility-adjusted returns
            mean_returns = window_returns.mean()
            vol = window_returns.std()
            signals = mean_returns / (vol + 1e-8)  # Add small epsilon to avoid division by zero
        else:
            # Simple returns
            signals = window_returns.mean()
        
        # Create rankings (higher signal = better rank)
        rankings = signals.rank(ascending=False).values - 1  # Convert to 0-indexed
        
        # Convert to ranking matrix format for Plackett-Luce
        ranking_matrix = np.array([rankings.astype(int)])
        
        # Calculate strengths using Plackett-Luce
        try:
            strengths = plackett_luce_strength(ranking_matrix)
            # Convert to probabilities
            probabilities = strengths / np.sum(strengths)
            
            period_strengths.append(strengths)
            period_probabilities.append(probabilities)
            period_dates.append(time_point.date())
        except:
            continue
    
    if len(period_strengths) < 2:
        return None, None, None
    
    return period_strengths, period_probabilities, period_dates

def simple_flow_visualization(returns_df, sector_names, window_units, window_type, step_size, signal_type):
    """SIMPLE flow visualization - one frame per period"""
    
    # Get time series data  
    period_strengths, period_probabilities, period_dates = compute_sector_strengths_time_series(
        returns_df, window_units, window_type, step_size, signal_type
    )
    
    if period_strengths is None:
        return None, None
    
    n_sectors = len(sector_names)
    
    # Create node positions in a circle
    angles = np.linspace(0, 2*np.pi, n_sectors, endpoint=False)
    radius = 5
    x_pos = radius * np.cos(angles)
    y_pos = radius * np.sin(angles)
    
    # Find max probability across all time points for consistent scaling
    all_probs = np.concatenate(period_probabilities)
    max_prob = np.max(all_probs)
    scale_factor = 60 / max_prob if max_prob > 0 else 1
    
    # Get correlation costs for flows
    correlation_costs, _ = compute_correlation_costs(returns_df, window_years=3)
    
    # Create frames: one per period
    animation_frames = []
    frame_dates = []
    
    for period_idx in range(len(period_probabilities)):
        current_probs = period_probabilities[period_idx]
        current_sizes = current_probs * scale_factor + 20
        
        # Node colors based on trend
        if period_idx > 0:
            prev_probs = period_probabilities[period_idx - 1]
            changes = current_probs - prev_probs
            colors = ['red' if change < -0.01 else 'green' if change > 0.01 else 'blue' for change in changes]
        else:
            colors = ['blue'] * n_sectors
        
        # Calculate flows to NEXT period (if exists)
        arrows = []
        if period_idx < len(period_probabilities) - 1:
            next_probs = period_probabilities[period_idx + 1]
            
            # Prepare arrays for optimal transport
            current_probs_np = np.array(current_probs, dtype=np.float64)
            next_probs_np = np.array(next_probs, dtype=np.float64)
            correlation_costs_np = np.array(correlation_costs, dtype=np.float64)
            
            # Normalize
            current_probs_np = current_probs_np / np.sum(current_probs_np)
            next_probs_np = next_probs_np / np.sum(next_probs_np)
            
            try:
                flow_matrix = ot.sinkhorn(current_probs_np, next_probs_np, correlation_costs_np, reg=0.01)
            except Exception:
                flow_matrix = np.zeros((n_sectors, n_sectors))
            
            # Create arrows for significant flows
            min_flow = 0.02
            for i in range(n_sectors):
                for j in range(n_sectors):
                    if i != j and flow_matrix[i, j] > min_flow:
                        flow_strength = flow_matrix[i, j]
                        arrow_opacity = min(0.8, flow_strength * 20)
                        
                        if arrow_opacity > 0.1:
                            arrow = go.Scatter(
                                x=[x_pos[i], x_pos[j]],
                                y=[y_pos[i], y_pos[j]], 
                                mode='lines',
                                line=dict(
                                    color=f'rgba(255, 140, 0, {arrow_opacity})',  # Orange
                                    width=max(2, flow_strength * 30)
                                ),
                                showlegend=False,
                                hoverinfo='skip'
                            )
                            arrows.append(arrow)
        
        # Create nodes
        nodes = go.Scatter(
            x=x_pos,
            y=y_pos,
            mode='markers+text',
            marker=dict(
                size=current_sizes,
                color=colors,
                opacity=0.8,
                line=dict(width=2, color='black')
            ),
            text=sector_names,
            textposition="middle center",
            textfont=dict(size=10, color='white', family="Arial Black"),
            name=f"Period {period_idx + 1}"
        )
        
        # Combine nodes and arrows
        frame_data = [nodes] + arrows
        animation_frames.append(frame_data)
        frame_dates.append(period_dates[period_idx])
    
    # Create figure
    fig = go.Figure(data=animation_frames[0])
    
    # Add frames
    frames = []
    for frame_idx, frame_data in enumerate(animation_frames):
        frames.append(go.Frame(data=frame_data, name=str(frame_idx)))
    
    fig.frames = frames
    
    # Layout
    fig.update_layout(
        title=dict(
            text=f"💰 Money Flow: {window_units} {window_type.title()}<br><sub>🔴 Declining | 🟢 Growing | 🔵 Stable</sub>",
            x=0.5,
            font=dict(size=16)
        ),
        xaxis=dict(showgrid=False, zeroline=False, showticklabels=False, range=[-7, 7]),
        yaxis=dict(showgrid=False, zeroline=False, showticklabels=False, range=[-7, 7]),
        showlegend=False,
        width=800,
        height=800,
        updatemenus=[{
            "buttons": [
                {
                    "args": [None, {"frame": {"duration": 1000, "redraw": True},
                                   "fromcurrent": True, "transition": {"duration": 300}}],
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
            "type": "buttons",
            "x": 0.1,
            "y": 0,
        }],
        sliders=[{
            "active": 0,
            "currentvalue": {"visible": False},
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
                    "label": frame_dates[frame_idx].strftime('%m/%d'),
                    "method": "animate"
                } for frame_idx in range(len(animation_frames))
            ]
        }]
    )
    
    return fig, frame_dates

def main():
    st.set_page_config(page_title="Flowtation - Simple", layout="wide")
    
    st.title("💰 Flowtation - Simple Money Flow")
    st.markdown("**Clean, logical visualization of sector money flows**")
    
    # Download data
    with st.spinner("Downloading latest data..."):
        prices_df, metadata = download_data()
    
    if prices_df is None:
        st.error("Failed to load data")
        return
    
    # Prepare data
    analysis_data = prepare_data_for_analysis(prices_df)
    returns_data = analysis_data['returns']
    
    # User inputs
    col1, col2, col3 = st.columns(3)
    
    with col1:
        window_units = st.slider("Timeframe", 1, 52, 12, help="Number of time units")
        window_type = st.selectbox("Unit", ["weeks", "months"], help="Time unit type")
    
    with col2:
        step_size = st.selectbox("Step Size", ["1 week", "2 weeks", "1 month"], 
                                index=1, help="How often to recalculate")
        signal_type = st.selectbox("Signal", ["vol_adjusted", "simple_returns"], 
                                  help="Ranking method")
    
    with col3:
        st.metric("Total Sectors", len(returns_data.columns))
        st.metric("Data Period", f"{analysis_data['total_days']} days")
    
    # Generate visualization
    if st.button("🎬 Generate Flow Animation", type="primary"):
        with st.spinner("Calculating flows..."):
            fig, frame_dates = simple_flow_visualization(
                returns_data, returns_data.columns, window_units, 
                window_type, step_size, signal_type
            )
            
            if fig:
                st.plotly_chart(fig, use_container_width=True, height=800)
                
                st.success(f"✅ Generated {len(frame_dates)} periods")
                
                # Show explanation
                st.markdown("""
                **🎯 How it works:**
                - **Node size**: Sector strength (larger = more money allocated)
                - **Node color**: 🔴 Declining | 🟢 Growing | 🔵 Stable
                - **Orange arrows**: Money flowing between sectors
                - **Arrow thickness**: Amount of money flowing
                
                **🎬 Controls:**
                - ▶️ **Play**: Watch the full sequence
                - **Slider**: Jump to specific dates
                - Each frame shows flows TO the next period
                """)
            else:
                st.error("Failed to generate visualization")

if __name__ == "__main__":
    main()
