"""
Visualization Utilities
Contains functions for creating visualizations and word clouds.
"""

import logging
import os
from typing import List, Dict, Optional
import matplotlib.pyplot as plt
from wordcloud import WordCloud
import plotly.express as px
import plotly.graph_objects as go
import pandas as pd

# Configure logging
logger = logging.getLogger(__name__)


class VisualizationHelper:
    """
    Helper class for creating various visualizations.
    Handles word clouds, charts, and dashboard components.
    """
    
    def __init__(self, output_dir: str = "outputs"):
        """
        Initialize visualization helper.
        
        Args:
            output_dir (str): Directory to save output files
        """
        self.output_dir = output_dir
        os.makedirs(output_dir, exist_ok=True)
        logger.info(f"Visualization output directory: {output_dir}")
    
    def generate_wordcloud(
        self,
        text: str,
        output_path: Optional[str] = None,
        width: int = 800,
        height: int = 400,
        background_color: str = 'white',
        colormap: str = 'viridis'
    ) -> str:
        """
        Generate and save a word cloud from text.
        
        Args:
            text (str): Input text for word cloud
            output_path (Optional[str]): Path to save image
            width (int): Width of the word cloud
            height (int): Height of the word cloud
            background_color (str): Background color
            colormap (str): Matplotlib colormap name
        
        Returns:
            str: Path to saved word cloud image
        """
        try:
            if not text or not text.strip():
                logger.warning("Empty text provided for word cloud")
                return ""
            
            # Generate word cloud
            wordcloud = WordCloud(
                width=width,
                height=height,
                background_color=background_color,
                colormap=colormap,
                max_words=100,
                relative_scaling=0.5,
                min_font_size=10
            ).generate(text)
            
            # Create figure
            plt.figure(figsize=(width/100, height/100), dpi=100)
            plt.imshow(wordcloud, interpolation='bilinear')
            plt.axis('off')
            plt.tight_layout(pad=0)
            
            # Save figure
            if output_path is None:
                output_path = os.path.join(self.output_dir, 'wordcloud.png')
            
            plt.savefig(output_path, format='png', dpi=300, bbox_inches='tight')
            plt.close()
            
            logger.info(f"✅ Word cloud saved to: {output_path}")
            return output_path
            
        except Exception as e:
            logger.error(f"❌ Error generating word cloud: {e}")
            return ""
    
    def create_sentiment_pie_chart(
        self,
        sentiment_counts: Dict[str, int],
        title: str = "Sentiment Distribution"
    ) -> go.Figure:
        """
        Create a pie chart for sentiment distribution.
        
        Args:
            sentiment_counts (Dict[str, int]): Dictionary with sentiment labels and counts
            title (str): Chart title
        
        Returns:
            go.Figure: Plotly figure object
        """
        labels = list(sentiment_counts.keys())
        values = list(sentiment_counts.values())
        
        colors = {
            'Positive': '#2ecc71',
            'Negative': '#e74c3c',
            'Neutral': '#95a5a6'
        }
        
        color_sequence = [colors.get(label, '#3498db') for label in labels]
        
        fig = go.Figure(data=[go.Pie(
            labels=labels,
            values=values,
            hole=0.3,
            marker=dict(colors=color_sequence),
            textinfo='label+percent',
            textposition='auto'
        )])
        
        fig.update_layout(
            title=title,
            showlegend=True,
            height=400
        )
        
        return fig
    
    def create_sentiment_bar_chart(
        self,
        sentiment_counts: Dict[str, int],
        title: str = "Sentiment Distribution"
    ) -> go.Figure:
        """
        Create a bar chart for sentiment distribution.
        
        Args:
            sentiment_counts (Dict[str, int]): Dictionary with sentiment labels and counts
            title (str): Chart title
        
        Returns:
            go.Figure: Plotly figure object
        """
        df = pd.DataFrame(list(sentiment_counts.items()), columns=['Sentiment', 'Count'])
        
        colors = {
            'Positive': '#2ecc71',
            'Negative': '#e74c3c',
            'Neutral': '#95a5a6'
        }
        
        df['Color'] = df['Sentiment'].map(lambda x: colors.get(x, '#3498db'))
        
        fig = px.bar(
            df,
            x='Sentiment',
            y='Count',
            title=title,
            color='Sentiment',
            color_discrete_map=colors,
            text='Count'
        )
        
        fig.update_traces(textposition='outside')
        fig.update_layout(
            showlegend=False,
            height=400,
            xaxis_title="Sentiment",
            yaxis_title="Number of Posts"
        )
        
        return fig
    
    def create_time_series_chart(
        self,
        df: pd.DataFrame,
        date_column: str,
        value_column: str,
        title: str = "Post Activity Over Time"
    ) -> go.Figure:
        """
        Create a time series line chart.
        
        Args:
            df (pd.DataFrame): DataFrame with time series data
            date_column (str): Name of date column
            value_column (str): Name of value column
            title (str): Chart title
        
        Returns:
            go.Figure: Plotly figure object
        """
        fig = px.line(
            df,
            x=date_column,
            y=value_column,
            title=title,
            markers=True
        )
        
        fig.update_layout(
            xaxis_title="Date",
            yaxis_title="Number of Posts",
            height=400,
            hovermode='x unified'
        )
        
        fig.update_traces(
            line=dict(color='#3498db', width=2),
            marker=dict(size=6)
        )
        
        return fig
    
    def create_top_keywords_chart(
        self,
        keywords_dict: Dict[str, int],
        title: str = "Top Keywords",
        top_n: int = 15
    ) -> go.Figure:
        """
        Create a horizontal bar chart for top keywords.
        
        Args:
            keywords_dict (Dict[str, int]): Dictionary with keywords and counts
            title (str): Chart title
            top_n (int): Number of top keywords to display
        
        Returns:
            go.Figure: Plotly figure object
        """
        # Sort and get top N
        sorted_keywords = sorted(keywords_dict.items(), key=lambda x: x[1], reverse=True)[:top_n]
        keywords, counts = zip(*sorted_keywords) if sorted_keywords else ([], [])
        
        # Reverse for better display (highest at top)
        keywords = list(keywords)[::-1]
        counts = list(counts)[::-1]
        
        fig = go.Figure(go.Bar(
            x=counts,
            y=keywords,
            orientation='h',
            marker=dict(
                color=counts,
                colorscale='Viridis',
                showscale=False
            ),
            text=counts,
            textposition='auto'
        ))
        
        fig.update_layout(
            title=title,
            xaxis_title="Frequency",
            yaxis_title="Keywords",
            height=500,
            showlegend=False
        )
        
        return fig
    
    def create_subreddit_distribution(
        self,
        subreddit_counts: Dict[str, int],
        title: str = "Posts by Subreddit",
        top_n: int = 10
    ) -> go.Figure:
        """
        Create a chart showing distribution of posts across subreddits.
        
        Args:
            subreddit_counts (Dict[str, int]): Dictionary with subreddit and counts
            title (str): Chart title
            top_n (int): Number of top subreddits to display
        
        Returns:
            go.Figure: Plotly figure object
        """
        # Sort and get top N
        sorted_subs = sorted(subreddit_counts.items(), key=lambda x: x[1], reverse=True)[:top_n]
        subreddits, counts = zip(*sorted_subs) if sorted_subs else ([], [])
        
        fig = px.bar(
            x=list(subreddits),
            y=list(counts),
            title=title,
            labels={'x': 'Subreddit', 'y': 'Number of Posts'},
            color=list(counts),
            color_continuous_scale='Blues'
        )
        
        fig.update_layout(
            showlegend=False,
            height=400,
            xaxis_tickangle=-45
        )
        
        return fig


def save_visualization(fig: go.Figure, filename: str, output_dir: str = "outputs"):
    """
    Save a Plotly figure to file.
    
    Args:
        fig (go.Figure): Plotly figure object
        filename (str): Output filename
        output_dir (str): Output directory
    """
    os.makedirs(output_dir, exist_ok=True)
    filepath = os.path.join(output_dir, filename)
    fig.write_html(filepath)
    logger.info(f"✅ Visualization saved to: {filepath}")


if __name__ == "__main__":
    # Test visualization helper
    viz = VisualizationHelper()
    
    # Test word cloud
    test_text = "artificial intelligence machine learning deep learning neural networks " * 10
    viz.generate_wordcloud(test_text, output_path="test_wordcloud.png")
    
    # Test sentiment chart
    sentiment_data = {'Positive': 150, 'Negative': 50, 'Neutral': 100}
    fig = viz.create_sentiment_pie_chart(sentiment_data)
    save_visualization(fig, "test_sentiment_pie.html")
    
    print("✅ Visualization tests completed")
