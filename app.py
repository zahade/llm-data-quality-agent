"""
Streamlit Web Interface for LLM Data Quality Agent
"""

import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from src.data_analyzer import DataAnalyzer
from src.llm_agent import LLMAgent
from src.rag_system import RAGSystem
from src.cleaning_engine import CleaningEngine
from src.utils import get_basic_stats, create_issue_summary, save_cleaned_data

# Page config
st.set_page_config(
    page_title="LLM Data Quality Agent",
    page_icon="🤖",
    layout="wide"
)

# Title
st.title("🤖 LLM Data Quality Agent")
st.markdown("**Autonomous AI-powered data quality analysis for energy consumption data**")
st.markdown("---")

# Initialize session state
if 'df' not in st.session_state:
    st.session_state.df = None
if 'issues' not in st.session_state:
    st.session_state.issues = None
if 'cleaned_df' not in st.session_state:
    st.session_state.cleaned_df = None

# Sidebar
with st.sidebar:
    st.header("⚙️ Configuration")
    
    # File upload
    uploaded_file = st.file_uploader("Upload CSV file", type=['csv'])
    
    if uploaded_file:
        st.session_state.df = pd.read_csv(uploaded_file)
        if 'timestamp' in st.session_state.df.columns:
            st.session_state.df['timestamp'] = pd.to_datetime(st.session_state.df['timestamp'])
        st.success(f"✅ Loaded {len(st.session_state.df)} rows")
    
    # Or use sample data
    if st.button("📊 Use Sample Data"):
        st.session_state.df = pd.read_csv("data/sample/energy_data_with_issues.csv")
        st.session_state.df['timestamp'] = pd.to_datetime(st.session_state.df['timestamp'])
        st.success(f"✅ Loaded sample data ({len(st.session_state.df)} rows)")
    
    st.markdown("---")
    
    # Cleaning strategy
    st.subheader("🧹 Cleaning Strategy")
    strategy = st.radio(
        "Select strategy:",
        ["conservative", "aggressive"],
        help="Conservative: Preserve questionable data. Aggressive: Remove all suspicious values."
    )

# Main content
if st.session_state.df is not None:
    df = st.session_state.df
    
    # Tabs
    tab1, tab2, tab3, tab4, tab5 = st.tabs([
        "📊 Data Overview", 
        "🔍 Quality Analysis", 
        "🤖 AI Insights",
        "🧹 Data Cleaning",
        "💬 Ask Questions"
    ])
    
    # TAB 1: Data Overview
    with tab1:
        st.subheader("📊 Data Overview")
        
        col1, col2, col3, col4 = st.columns(4)
        with col1:
            st.metric("Total Records", f"{len(df):,}")
        with col2:
            st.metric("Date Range", f"{(df['timestamp'].max() - df['timestamp'].min()).days} days")
        with col3:
            st.metric("Avg Consumption", f"{df['consumption_kwh'].mean():.2f} kWh")
        with col4:
            missing_pct = (df['consumption_kwh'].isna().sum() / len(df)) * 100
            st.metric("Missing Data", f"{missing_pct:.1f}%")
        
        st.markdown("### Sample Data")
        st.dataframe(df.head(20), use_container_width=True)
        
        st.markdown("### Statistical Summary")
        st.dataframe(df.describe(), use_container_width=True)
    
    # TAB 2: Quality Analysis
    with tab2:
        st.subheader("🔍 Data Quality Analysis")
        
        if st.button("🚀 Run Analysis"):
            with st.spinner("Analyzing data quality..."):
                # Run analyzer
                analyzer = DataAnalyzer(df)
                st.session_state.issues = analyzer.detect_all_issues()
                
                # Display results
                st.markdown("### Issues Detected")
                st.text(create_issue_summary(st.session_state.issues))
                
                # Visualizations
                col1, col2 = st.columns(2)
                
                with col1:
                    # Issue counts
                    issue_counts = {
                        k: v for k, v in st.session_state.issues.items() 
                        if isinstance(v, int) and v > 0
                    }
                    
                    if issue_counts:
                        fig, ax = plt.subplots(figsize=(8, 5))
                        plt.barh(list(issue_counts.keys()), list(issue_counts.values()))
                        plt.xlabel("Count")
                        plt.title("Data Quality Issues")
                        plt.tight_layout()
                        st.pyplot(fig)
                
                with col2:
                    # Distribution plot
                    valid_data = df[df['consumption_kwh'] >= 0]['consumption_kwh']
                    fig, ax = plt.subplots(figsize=(8, 5))
                    plt.hist(valid_data, bins=50, edgecolor='black')
                    plt.xlabel("Consumption (kWh)")
                    plt.ylabel("Frequency")
                    plt.title("Consumption Distribution")
                    plt.tight_layout()
                    st.pyplot(fig)
        
        if st.session_state.issues:
            st.success("✅ Analysis complete!")
    
    # TAB 3: AI Insights
    with tab3:
        st.subheader("🤖 AI-Powered Insights")
        
        if st.session_state.issues:
            if st.button("🧠 Generate AI Analysis"):
                with st.spinner("LLM Agent is analyzing your data..."):
                    # Initialize LLM agent
                    agent = LLMAgent()
                    
                    # Get stats
                    stats = get_basic_stats(df, 'consumption_kwh')
                    
                    # Generate analysis
                    analysis = agent.analyze_data_quality(
                        stats=stats,
                        issues=st.session_state.issues
                    )
                    
                    st.markdown("### 🤖 LLM Analysis Report")
                    st.markdown(analysis)
                    
                    # Priority analysis
                    st.markdown("### 📋 Prioritized Action Plan")
                    priorities = agent.prioritize_issues(st.session_state.issues)
                    st.markdown(priorities)
        else:
            st.info("👆 Run Quality Analysis first (Tab 2)")
    
    # TAB 4: Data Cleaning
    with tab4:
        st.subheader("🧹 Automated Data Cleaning")
        
        if st.session_state.issues:
            st.markdown(f"**Strategy:** {strategy.title()}")
            
            if st.button("🚀 Clean Data"):
                with st.spinner(f"Cleaning data ({strategy} strategy)..."):
                    # Initialize cleaning engine
                    cleaner = CleaningEngine(df)
                    
                    # Clean
                    cleaned_df = cleaner.clean_all(strategy=strategy)
                    st.session_state.cleaned_df = cleaned_df
                    
                    # Show summary
                    st.text(cleaner.get_cleaning_summary())
                    
                    # Before/After comparison
                    st.markdown("### 📊 Before vs After")
                    stats_comparison = cleaner.get_before_after_stats()
                    
                    col1, col2 = st.columns(2)
                    with col1:
                        st.markdown("**Before Cleaning**")
                        st.json(stats_comparison['before'])
                    with col2:
                        st.markdown("**After Cleaning**")
                        st.json(stats_comparison['after'])
                    
                    # Download button
                    csv = cleaned_df.to_csv(index=False)
                    st.download_button(
                        label="📥 Download Cleaned Data",
                        data=csv,
                        file_name="cleaned_energy_data.csv",
                        mime="text/csv"
                    )
                    
                    st.success("✅ Data cleaning complete!")
            
            if st.session_state.cleaned_df is not None:
                st.markdown("### Cleaned Data Preview")
                st.dataframe(st.session_state.cleaned_df.head(20), use_container_width=True)
        else:
            st.info("👆 Run Quality Analysis first (Tab 2)")
    
    # TAB 5: Ask Questions
    with tab5:
        st.subheader("💬 Ask Questions About Your Data")
        
        if st.session_state.issues:
            question = st.text_input("Ask a question about the data:")
            
            if question and st.button("🔍 Get Answer"):
                with st.spinner("LLM Agent is thinking..."):
                    agent = LLMAgent()
                    stats = get_basic_stats(df, 'consumption_kwh')
                    
                    # Create data summary
                    summary = f"""
Dataset: {len(df)} records
Date Range: {df['timestamp'].min()} to {df['timestamp'].max()}
Mean Consumption: {stats['mean']:.2f} kWh
Issues Found: {sum(v for v in st.session_state.issues.values() if isinstance(v, int))}
                    """
                    
                    answer = agent.interactive_qa(question, summary)
                    
                    st.markdown("### 🤖 Answer")
                    st.markdown(answer)
        else:
            st.info("👆 Run Quality Analysis first (Tab 2)")

else:
    # Welcome screen
    st.info("👈 Upload a CSV file or use sample data to get started")
    
    st.markdown("""
    ### 🚀 Features
    
    - **📊 Data Overview**: View statistics and sample data
    - **🔍 Quality Analysis**: Detect missing values, outliers, anomalies
    - **🤖 AI Insights**: LLM-powered analysis and recommendations
    - **🧹 Data Cleaning**: Automated cleaning with multiple strategies
    - **💬 Q&A**: Ask questions about your data
    
    ### 📁 Sample Data
    
    Click "Use Sample Data" in the sidebar to load pre-generated energy consumption data with intentional quality issues.
    """)