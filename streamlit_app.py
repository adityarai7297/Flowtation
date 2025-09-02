import streamlit as st
import pandas as pd
import requests
import json
from datetime import datetime, date
import plotly.express as px
import plotly.graph_objects as go
from io import BytesIO
import gzip

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

def display_data_overview(df, metadata, analysis_data):
    """Display data overview and statistics"""
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.metric("Total Tickers", len(metadata['tickers']))
    
    with col2:
        st.metric("Total Records", f"{metadata['rows']:,}")
    
    with col3:
        st.metric("Date Range", f"{metadata['date_min']} to {metadata['date_max']}")
    
    with col4:
        data_age = (datetime.now().date() - pd.to_datetime(metadata['date_max']).date()).days
        st.metric("Data Age", f"{data_age} days")
    
    # Data quality checks
    st.subheader("📊 Data Quality")
    col1, col2 = st.columns(2)
    
    with col1:
        st.write("**Tickers Available:**")
        tickers = sorted(metadata['tickers'])
        st.write(", ".join(tickers))
    
    with col2:
        st.write("**Ready for Analysis:**")
        st.success(f"✅ Daily returns: {len(analysis_data['returns_daily'])} days")
        st.success(f"✅ Monthly returns: {len(analysis_data['returns_monthly'])} months")
        st.success(f"✅ Performance periods: {len(analysis_data['performance_periods'])} timeframes")

def display_recent_performance(analysis_data):
    """Display recent performance for quick overview"""
    st.subheader("🏆 Recent Performance Rankings")
    
    performance = analysis_data['performance_periods']
    
    # Create tabs for different periods
    periods = list(performance.keys())
    tabs = st.tabs(periods)
    
    for i, period in enumerate(periods):
        with tabs[i]:
            perf_data = performance[period]
            
            # Create a nice dataframe for display
            perf_df = pd.DataFrame({
                'Ticker': perf_data.index,
                'Return (%)': perf_data.values.round(2),
                'Rank': range(1, len(perf_data) + 1)
            })
            
            # Color coding
            def color_performance(val):
                if val > 5:
                    return 'background-color: #d4edda'  # Light green
                elif val > 0:
                    return 'background-color: #fff3cd'  # Light yellow
                else:
                    return 'background-color: #f8d7da'  # Light red
            
            styled_df = perf_df.style.applymap(color_performance, subset=['Return (%)'])
            st.dataframe(styled_df, use_container_width=True)

def display_data_sample(df):
    """Display sample of raw data"""
    st.subheader("📋 Raw Data Sample")
    
    # Show recent data
    recent_data = df.sort_values(['date', 'ticker']).tail(20)
    st.dataframe(recent_data, use_container_width=True)
    
    # Show data types and info
    with st.expander("Data Schema & Info"):
        col1, col2 = st.columns(2)
        
        with col1:
            st.write("**Data Types:**")
            schema_df = pd.DataFrame({
                'Column': df.dtypes.index,
                'Type': df.dtypes.values.astype(str)
            })
            st.dataframe(schema_df, use_container_width=True)
        
        with col2:
            st.write("**Summary Statistics:**")
            numeric_cols = df.select_dtypes(include=['float64', 'int64']).columns
            st.dataframe(df[numeric_cols].describe(), use_container_width=True)

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
    
    # Success message
    st.success(f"✅ Successfully loaded {len(df):,} records from {len(metadata['tickers'])} tickers")
    
    # Display overview
    display_data_overview(df, metadata, analysis_data)
    
    # Display recent performance
    display_recent_performance(analysis_data)
    
    # Data sample
    display_data_sample(df)
    
    # Analysis readiness indicator
    st.subheader("🚀 Ready for Rank Algorithms")
    st.info("""
    **Data is now prepared for rank algorithm processing:**
    - ✅ Daily, weekly, monthly returns calculated
    - ✅ Performance periods (1M, 3M, 6M, 1Y) computed
    - ✅ Volatility metrics available
    - ✅ Latest prices and volume data ready
    - ✅ All data cleaned and formatted for analysis
    
    **Available datasets in `analysis_data`:**
    - `returns_daily`, `returns_weekly`, `returns_monthly`
    - `prices`, `volume`, `latest_prices`
    - `performance_periods`, `volatility`
    """)
    
    # Export option for algorithms
    if st.button("📥 Export Processed Data for Algorithm Development"):
        # Create download links for processed data
        st.download_button(
            label="Download Daily Returns (CSV)",
            data=analysis_data['returns_daily'].to_csv(),
            file_name=f"daily_returns_{date.today()}.csv",
            mime="text/csv"
        )

if __name__ == "__main__":
    main()
