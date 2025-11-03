"""
Streamlit Dashboard Application
Interactive dashboard for visualizing Reddit social media analytics.
Uses Plotly for interactive visualizations and real-time MongoDB data.
"""

import os
import sys
import logging
from datetime import datetime
import streamlit as st
import plotly.express as px
import plotly.graph_objects as go
import pandas as pd
from PIL import Image

# Add parent directory to path for imports
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from config.mongo_config import get_collection
from data_analysis import DataAnalysis
from sentiment_analysis import SentimentAnalysis
from utils.visualization_utils import VisualizationHelper

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Page configuration
st.set_page_config(
    page_title="Reddit Social Media Analytics",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for better styling
st.markdown("""
    <style>
    .main {
        padding: 0rem 1rem;
    }
    .stMetric {
        background-color: #f0f2f6;
        padding: 15px;
        border-radius: 10px;
    }
    h1 {
        color: #1f77b4;
    }
    h2 {
        color: #2c3e50;
    }
    .stAlert {
        border-radius: 10px;
    }
    </style>
    """, unsafe_allow_html=True)


@st.cache_resource
def init_connections():
    """Initialize database connections and analysis objects."""
    try:
        collection = get_collection()
        analyzer = DataAnalysis()
        sentiment_analyzer = SentimentAnalysis()
        viz_helper = VisualizationHelper()
        return collection, analyzer, sentiment_analyzer, viz_helper
    except Exception as e:
        st.error(f"❌ Failed to initialize connections: {e}")
        return None, None, None, None


@st.cache_data(ttl=300)  # Cache for 5 minutes
def load_sentiment_distribution(_analyzer):
    """Load sentiment distribution data."""
    return _analyzer.get_sentiment_distribution()


@st.cache_data(ttl=300)
def load_top_subreddits(_analyzer, limit=10):
    """Load top subreddits data."""
    return _analyzer.get_top_subreddits(limit)


@st.cache_data(ttl=300)
def load_posts_by_day(_analyzer):
    """Load posts by day data."""
    return _analyzer.get_posts_by_day()


@st.cache_data(ttl=300)
def load_top_keywords(_analyzer, limit=20):
    """Load top keywords data."""
    return _analyzer.get_top_keywords(limit)


@st.cache_data(ttl=300)
def load_sentiment_by_subreddit(_analyzer, limit=10):
    """Load sentiment by subreddit data."""
    return _analyzer.get_sentiment_by_subreddit(limit)


@st.cache_data(ttl=300)
def load_posts_by_hour(_analyzer):
    """Load posts by hour data."""
    return _analyzer.get_posts_by_hour()


def create_sentiment_pie_chart(df):
    """Create sentiment distribution pie chart."""
    if df.empty:
        return None
    
    colors = {
        'Positive': '#2ecc71',
        'Negative': '#e74c3c',
        'Neutral': '#95a5a6'
    }
    
    color_list = [colors.get(sentiment, '#3498db') for sentiment in df['Sentiment']]
    
    fig = go.Figure(data=[go.Pie(
        labels=df['Sentiment'],
        values=df['Count'],
        hole=0.4,
        marker=dict(colors=color_list),
        textinfo='label+percent+value',
        textposition='auto'
    )])
    
    fig.update_layout(
        title="Sentiment Distribution",
        showlegend=True,
        height=400,
        margin=dict(l=20, r=20, t=60, b=20)
    )
    
    return fig


def create_time_series_chart(df):
    """Create time series chart for post activity."""
    if df.empty:
        return None
    
    fig = px.line(
        df,
        x='Date',
        y='Count',
        title='Post Activity Over Time',
        markers=True
    )
    
    fig.update_layout(
        xaxis_title="Date",
        yaxis_title="Number of Posts",
        height=400,
        hovermode='x unified',
        margin=dict(l=20, r=20, t=60, b=20)
    )
    
    fig.update_traces(
        line=dict(color='#3498db', width=3),
        marker=dict(size=8)
    )
    
    return fig


def create_keywords_bar_chart(df, top_n=15):
    """Create horizontal bar chart for top keywords."""
    if df.empty:
        return None
    
    df_top = df.head(top_n).iloc[::-1]  # Reverse for better display
    
    fig = go.Figure(go.Bar(
        x=df_top['Frequency'],
        y=df_top['Keyword'],
        orientation='h',
        marker=dict(
            color=df_top['Frequency'],
            colorscale='Viridis',
            showscale=False
        ),
        text=df_top['Frequency'],
        textposition='auto'
    ))
    
    fig.update_layout(
        title=f"Top {top_n} Keywords",
        xaxis_title="Frequency",
        yaxis_title="Keywords",
        height=500,
        showlegend=False,
        margin=dict(l=20, r=20, t=60, b=20)
    )
    
    return fig


def create_subreddit_bar_chart(df):
    """Create bar chart for top subreddits."""
    if df.empty:
        return None
    
    fig = px.bar(
        df,
        x='Subreddit',
        y='Posts',
        title='Top Subreddits by Post Count',
        color='Posts',
        color_continuous_scale='Blues',
        text='Posts'
    )
    
    fig.update_layout(
        showlegend=False,
        height=400,
        xaxis_tickangle=-45,
        margin=dict(l=20, r=20, t=60, b=20)
    )
    
    fig.update_traces(textposition='outside')
    
    return fig


def create_hourly_activity_chart(df):
    """Create chart showing post activity by hour."""
    if df.empty:
        return None
    
    fig = go.Figure(data=[
        go.Bar(
            x=df['Hour'],
            y=df['Count'],
            marker=dict(
                color=df['Count'],
                colorscale='Turbo',
                showscale=True,
                colorbar=dict(title="Posts")
            ),
            text=df['Count'],
            textposition='auto'
        )
    ])
    
    fig.update_layout(
        title='Post Activity by Hour of Day',
        xaxis_title='Hour (24-hour format)',
        yaxis_title='Number of Posts',
        height=400,
        xaxis=dict(tickmode='linear', tick0=0, dtick=1),
        margin=dict(l=20, r=20, t=60, b=20)
    )
    
    return fig


def create_sentiment_subreddit_chart(df):
    """Create grouped bar chart for sentiment by subreddit."""
    if df.empty:
        return None
    
    fig = go.Figure()
    
    fig.add_trace(go.Bar(
        name='Positive',
        x=df['Subreddit'],
        y=df['Positive'],
        marker_color='#2ecc71'
    ))
    
    fig.add_trace(go.Bar(
        name='Neutral',
        x=df['Subreddit'],
        y=df['Neutral'],
        marker_color='#95a5a6'
    ))
    
    fig.add_trace(go.Bar(
        name='Negative',
        x=df['Subreddit'],
        y=df['Negative'],
        marker_color='#e74c3c'
    ))
    
    fig.update_layout(
        title='Sentiment Distribution by Subreddit',
        xaxis_title='Subreddit',
        yaxis_title='Number of Posts',
        barmode='group',
        height=450,
        xaxis_tickangle=-45,
        legend=dict(
            orientation="h",
            yanchor="bottom",
            y=1.02,
            xanchor="right",
            x=1
        ),
        margin=dict(l=20, r=20, t=60, b=20)
    )
    
    return fig


def display_wordcloud(sentiment_filter=None):
    """Display word cloud image."""
    try:
        if sentiment_filter:
            filename = f'wordcloud_{sentiment_filter.lower()}.png'
        else:
            filename = 'wordcloud.png'
        
        filepath = os.path.join('outputs', filename)
        
        if os.path.exists(filepath):
            image = Image.open(filepath)
            st.image(image, use_column_width=True)
        else:
            st.warning(f"Word cloud not found. Generate it first by running sentiment_analysis.py")
            
    except Exception as e:
        st.error(f"Error loading word cloud: {e}")


def main():
    """Main dashboard application."""
    
    # Header
    st.title("📊 Reddit Social Media Analytics Dashboard")
    st.markdown("**Analyzing user behavior and sentiment on Reddit**")
    st.markdown("---")
    
    # Initialize connections
    collection, analyzer, sentiment_analyzer, viz_helper = init_connections()
    
    if collection is None or analyzer is None:
        st.error("❌ Failed to connect to database. Please check your configuration.")
        return
    
    # Sidebar
    st.sidebar.title("⚙️ Dashboard Controls")
    st.sidebar.markdown("---")
    
    # Refresh button
    if st.sidebar.button("🔄 Refresh Data"):
        st.cache_data.clear()
        st.rerun()
    
    # Get collection stats
    try:
        total_posts = collection.count_documents({'data_type': 'post'})
        analyzed_posts = collection.count_documents({'sentiment_score': {'$exists': True}})
        total_comments = collection.count_documents({'data_type': 'comment'})
    except Exception as e:
        st.error(f"Error fetching stats: {e}")
        return
    
    # Sidebar stats
    st.sidebar.markdown("### 📈 Quick Stats")
    st.sidebar.metric("Total Posts", f"{total_posts:,}")
    st.sidebar.metric("Analyzed Posts", f"{analyzed_posts:,}")
    st.sidebar.metric("Total Comments", f"{total_comments:,}")
    
    st.sidebar.markdown("---")
    st.sidebar.markdown("### 📋 Navigation")
    
    # Navigation
    page = st.sidebar.radio(
        "Select View:",
        ["📊 Overview", "💭 Sentiment Analysis", "🔑 Keyword Analysis", "📱 Subreddit Analysis", "⏰ Time Analysis"]
    )
    
    st.sidebar.markdown("---")
    st.sidebar.markdown("### ℹ️ About")
    st.sidebar.info(
        "This dashboard provides real-time analytics of Reddit data, "
        "including sentiment analysis, keyword trends, and user behavior patterns."
    )
    
    # Main content based on selected page
    if page == "📊 Overview":
        st.header("📊 Overview Dashboard")
        
        # Key metrics in columns
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            st.metric(
                label="📝 Total Posts",
                value=f"{total_posts:,}",
                delta=None
            )
        
        with col2:
            completion_rate = (analyzed_posts / total_posts * 100) if total_posts > 0 else 0
            st.metric(
                label="✅ Analysis Rate",
                value=f"{completion_rate:.1f}%",
                delta=None
            )
        
        with col3:
            st.metric(
                label="💬 Comments",
                value=f"{total_comments:,}",
                delta=None
            )
        
        with col4:
            try:
                subreddit_count = len(collection.distinct('subreddit'))
                st.metric(
                    label="📱 Subreddits",
                    value=subreddit_count,
                    delta=None
                )
            except:
                st.metric(label="📱 Subreddits", value="N/A")
        
        st.markdown("---")
        
        # Two column layout for overview
        col_left, col_right = st.columns(2)
        
        with col_left:
            st.subheader("💭 Sentiment Distribution")
            sentiment_df = load_sentiment_distribution(analyzer)
            if not sentiment_df.empty:
                fig = create_sentiment_pie_chart(sentiment_df)
                if fig:
                    st.plotly_chart(fig, use_container_width=True)
            else:
                st.info("No sentiment data available")
        
        with col_right:
            st.subheader("📈 Post Activity Over Time")
            time_df = load_posts_by_day(analyzer)
            if not time_df.empty:
                fig = create_time_series_chart(time_df)
                if fig:
                    st.plotly_chart(fig, use_container_width=True)
            else:
                st.info("No time series data available")
        
        st.markdown("---")
        
        # Full width charts
        st.subheader("📱 Top Subreddits")
        subreddit_df = load_top_subreddits(analyzer, 10)
        if not subreddit_df.empty:
            fig = create_subreddit_bar_chart(subreddit_df)
            if fig:
                st.plotly_chart(fig, use_container_width=True)
            
            # Show data table
            with st.expander("📊 View Detailed Subreddit Stats"):
                st.dataframe(subreddit_df, use_container_width=True)
        else:
            st.info("No subreddit data available")
    
    elif page == "💭 Sentiment Analysis":
        st.header("💭 Sentiment Analysis")
        
        # Sentiment distribution
        col1, col2 = st.columns([1, 1])
        
        with col1:
            st.subheader("Sentiment Distribution")
            sentiment_df = load_sentiment_distribution(analyzer)
            if not sentiment_df.empty:
                fig = create_sentiment_pie_chart(sentiment_df)
                if fig:
                    st.plotly_chart(fig, use_container_width=True)
                
                # Show counts
                st.dataframe(sentiment_df, use_container_width=True)
            else:
                st.info("No sentiment data available")
        
        with col2:
            st.subheader("Sentiment by Subreddit")
            sentiment_subreddit_df = load_sentiment_by_subreddit(analyzer, 10)
            if not sentiment_subreddit_df.empty:
                # Show average sentiment scores
                fig = px.bar(
                    sentiment_subreddit_df,
                    x='Subreddit',
                    y='Avg_Sentiment',
                    title='Average Sentiment Score by Subreddit',
                    color='Avg_Sentiment',
                    color_continuous_scale=['red', 'yellow', 'green'],
                    color_continuous_midpoint=0
                )
                fig.update_layout(height=400, xaxis_tickangle=-45)
                st.plotly_chart(fig, use_container_width=True)
            else:
                st.info("No sentiment by subreddit data available")
        
        st.markdown("---")
        
        # Sentiment distribution by subreddit
        st.subheader("Detailed Sentiment by Subreddit")
        if not sentiment_subreddit_df.empty:
            fig = create_sentiment_subreddit_chart(sentiment_subreddit_df)
            if fig:
                st.plotly_chart(fig, use_container_width=True)
        
        st.markdown("---")
        
        # Word clouds
        st.subheader("☁️ Word Clouds")
        
        wordcloud_option = st.selectbox(
            "Select Sentiment Filter:",
            ["All Posts", "Positive", "Negative", "Neutral"]
        )
        
        if wordcloud_option == "All Posts":
            display_wordcloud()
        else:
            display_wordcloud(wordcloud_option)
    
    elif page == "🔑 Keyword Analysis":
        st.header("🔑 Keyword Analysis")
        
        # Top keywords
        st.subheader("Most Frequent Keywords")
        
        keyword_limit = st.slider("Number of keywords to display:", 5, 30, 15)
        keywords_df = load_top_keywords(analyzer, keyword_limit)
        
        if not keywords_df.empty:
            fig = create_keywords_bar_chart(keywords_df, keyword_limit)
            if fig:
                st.plotly_chart(fig, use_container_width=True)
            
            # Show data table
            with st.expander("📊 View Full Keyword List"):
                st.dataframe(keywords_df, use_container_width=True)
        else:
            st.info("No keyword data available")
        
        st.markdown("---")
        
        # Word cloud
        st.subheader("☁️ Keyword Word Cloud")
        display_wordcloud()
    
    elif page == "📱 Subreddit Analysis":
        st.header("📱 Subreddit Analysis")
        
        # Top subreddits
        st.subheader("Top Subreddits by Activity")
        
        subreddit_df = load_top_subreddits(analyzer, 15)
        
        if not subreddit_df.empty:
            # Bar chart
            fig = create_subreddit_bar_chart(subreddit_df)
            if fig:
                st.plotly_chart(fig, use_container_width=True)
            
            # Detailed table
            st.subheader("Detailed Statistics")
            st.dataframe(subreddit_df, use_container_width=True)
            
            st.markdown("---")
            
            # Sentiment by subreddit
            st.subheader("Sentiment Analysis by Subreddit")
            sentiment_subreddit_df = load_sentiment_by_subreddit(analyzer, 10)
            
            if not sentiment_subreddit_df.empty:
                fig = create_sentiment_subreddit_chart(sentiment_subreddit_df)
                if fig:
                    st.plotly_chart(fig, use_container_width=True)
                
                st.dataframe(sentiment_subreddit_df, use_container_width=True)
            else:
                st.info("No sentiment data available by subreddit")
        else:
            st.info("No subreddit data available")
    
    elif page == "⏰ Time Analysis":
        st.header("⏰ Time-Based Analysis")
        
        # Daily activity
        st.subheader("📅 Daily Post Activity")
        time_df = load_posts_by_day(analyzer)
        
        if not time_df.empty:
            fig = create_time_series_chart(time_df)
            if fig:
                st.plotly_chart(fig, use_container_width=True)
            
            # Statistics
            col1, col2, col3 = st.columns(3)
            with col1:
                st.metric("Total Days", len(time_df))
            with col2:
                st.metric("Avg Posts/Day", f"{time_df['Count'].mean():.1f}")
            with col3:
                st.metric("Peak Day Posts", time_df['Count'].max())
        else:
            st.info("No daily activity data available")
        
        st.markdown("---")
        
        # Hourly activity
        st.subheader("🕐 Hourly Post Activity")
        hour_df = load_posts_by_hour(analyzer)
        
        if not hour_df.empty:
            fig = create_hourly_activity_chart(hour_df)
            if fig:
                st.plotly_chart(fig, use_container_width=True)
            
            # Show data
            with st.expander("📊 View Hourly Data"):
                st.dataframe(hour_df, use_container_width=True)
        else:
            st.info("No hourly activity data available")
    
    # Footer
    st.markdown("---")
    st.markdown(
        """
        <div style='text-align: center; color: #666; padding: 20px;'>
            <p>📊 Reddit Social Media Analytics Dashboard | Built with Streamlit & MongoDB</p>
            <p>Last updated: {}</p>
        </div>
        """.format(datetime.now().strftime("%Y-%m-%d %H:%M:%S")),
        unsafe_allow_html=True
    )


if __name__ == "__main__":
    main()
