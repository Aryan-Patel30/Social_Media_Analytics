"""
Data Analysis Module
Performs analytical queries using MongoDB Aggregation Pipeline and Pandas.
Analyzes trends, patterns, and generates insights from Reddit data.
"""

import os
import sys
import logging
from datetime import datetime, timedelta
from typing import List, Dict, Optional
import pandas as pd
from dotenv import load_dotenv

# Add parent directory to path for imports
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from config.mongo_config import get_collection

# Load environment variables
load_dotenv()

# Configure logging
logging.basicConfig(
    level=getattr(logging, os.getenv('LOG_LEVEL', 'INFO')),
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


class DataAnalysis:
    """
    Handles data analysis operations using MongoDB aggregation and Pandas.
    Provides insights about Reddit posts and user behavior.
    """
    
    def __init__(self):
        """Initialize MongoDB connection."""
        try:
            self.collection = get_collection()
            logger.info("✅ Data analysis module initialized")
        except Exception as e:
            logger.error(f"❌ Failed to initialize data analysis: {e}")
            raise
    
    def get_sentiment_distribution(self) -> pd.DataFrame:
        """
        Get distribution of sentiment across all posts.
        
        Returns:
            pd.DataFrame: DataFrame with sentiment counts
        """
        try:
            pipeline = [
                {'$match': {'sentiment_label': {'$exists': True}}},
                {'$group': {
                    '_id': '$sentiment_label',
                    'count': {'$sum': 1}
                }},
                {'$sort': {'count': -1}}
            ]
            
            results = list(self.collection.aggregate(pipeline))
            df = pd.DataFrame(results)
            
            if not df.empty:
                df.columns = ['Sentiment', 'Count']
                logger.info(f"📊 Sentiment distribution: {len(df)} categories")
            
            return df
            
        except Exception as e:
            logger.error(f"❌ Error getting sentiment distribution: {e}")
            return pd.DataFrame()
    
    def get_top_subreddits(self, limit: int = 10) -> pd.DataFrame:
        """
        Get top subreddits by post count.
        
        Args:
            limit (int): Number of top subreddits to return
        
        Returns:
            pd.DataFrame: DataFrame with subreddit stats
        """
        try:
            pipeline = [
                {'$match': {'data_type': 'post'}},
                {'$group': {
                    '_id': '$subreddit',
                    'post_count': {'$sum': 1},
                    'avg_score': {'$avg': '$score'},
                    'avg_comments': {'$avg': '$num_comments'}
                }},
                {'$sort': {'post_count': -1}},
                {'$limit': limit}
            ]
            
            results = list(self.collection.aggregate(pipeline))
            df = pd.DataFrame(results)
            
            if not df.empty:
                df.columns = ['Subreddit', 'Posts', 'Avg_Score', 'Avg_Comments']
                df['Avg_Score'] = df['Avg_Score'].round(2)
                df['Avg_Comments'] = df['Avg_Comments'].round(2)
                logger.info(f"📊 Top subreddits: {len(df)} found")
            
            return df
            
        except Exception as e:
            logger.error(f"❌ Error getting top subreddits: {e}")
            return pd.DataFrame()
    
    def get_top_authors(self, limit: int = 10) -> pd.DataFrame:
        """
        Get most active authors by post count.
        
        Args:
            limit (int): Number of top authors to return
        
        Returns:
            pd.DataFrame: DataFrame with author stats
        """
        try:
            pipeline = [
                {'$match': {'author': {'$ne': '[deleted]'}}},
                {'$group': {
                    '_id': '$author',
                    'post_count': {'$sum': 1},
                    'avg_score': {'$avg': '$score'}
                }},
                {'$sort': {'post_count': -1}},
                {'$limit': limit}
            ]
            
            results = list(self.collection.aggregate(pipeline))
            df = pd.DataFrame(results)
            
            if not df.empty:
                df.columns = ['Author', 'Posts', 'Avg_Score']
                df['Avg_Score'] = df['Avg_Score'].round(2)
                logger.info(f"📊 Top authors: {len(df)} found")
            
            return df
            
        except Exception as e:
            logger.error(f"❌ Error getting top authors: {e}")
            return pd.DataFrame()
    
    def get_posts_by_day(self) -> pd.DataFrame:
        """
        Get post frequency by day.
        
        Returns:
            pd.DataFrame: DataFrame with daily post counts
        """
        try:
            pipeline = [
                {'$match': {'created_utc': {'$exists': True}}},
                {'$addFields': {
                    'date': {
                        '$dateFromString': {
                            'dateString': '$created_utc',
                            'onError': None
                        }
                    }
                }},
                {'$match': {'date': {'$ne': None}}},
                {'$group': {
                    '_id': {
                        '$dateToString': {
                            'format': '%Y-%m-%d',
                            'date': '$date'
                        }
                    },
                    'count': {'$sum': 1}
                }},
                {'$sort': {'_id': 1}}
            ]
            
            results = list(self.collection.aggregate(pipeline))
            df = pd.DataFrame(results)
            
            if not df.empty:
                df.columns = ['Date', 'Count']
                df['Date'] = pd.to_datetime(df['Date'])
                logger.info(f"📊 Post frequency: {len(df)} days")
            
            return df
            
        except Exception as e:
            logger.error(f"❌ Error getting posts by day: {e}")
            return pd.DataFrame()
    
    def get_posts_by_hour(self) -> pd.DataFrame:
        """
        Get post frequency by hour of day.
        
        Returns:
            pd.DataFrame: DataFrame with hourly post counts
        """
        try:
            pipeline = [
                {'$match': {'created_utc': {'$exists': True}}},
                {'$addFields': {
                    'date': {
                        '$dateFromString': {
                            'dateString': '$created_utc',
                            'onError': None
                        }
                    }
                }},
                {'$match': {'date': {'$ne': None}}},
                {'$group': {
                    '_id': {'$hour': '$date'},
                    'count': {'$sum': 1}
                }},
                {'$sort': {'_id': 1}}
            ]
            
            results = list(self.collection.aggregate(pipeline))
            df = pd.DataFrame(results)
            
            if not df.empty:
                df.columns = ['Hour', 'Count']
                logger.info(f"📊 Hourly distribution: {len(df)} hours")
            
            return df
            
        except Exception as e:
            logger.error(f"❌ Error getting posts by hour: {e}")
            return pd.DataFrame()
    
    def get_top_keywords(self, limit: int = 20) -> pd.DataFrame:
        """
        Get most common keywords across all posts.
        
        Args:
            limit (int): Number of top keywords to return
        
        Returns:
            pd.DataFrame: DataFrame with keyword frequencies
        """
        try:
            pipeline = [
                {'$match': {'keywords': {'$exists': True}}},
                {'$unwind': '$keywords'},
                {'$group': {
                    '_id': '$keywords',
                    'count': {'$sum': 1}
                }},
                {'$sort': {'count': -1}},
                {'$limit': limit}
            ]
            
            results = list(self.collection.aggregate(pipeline))
            df = pd.DataFrame(results)
            
            if not df.empty:
                df.columns = ['Keyword', 'Frequency']
                logger.info(f"📊 Top keywords: {len(df)} found")
            
            return df
            
        except Exception as e:
            logger.error(f"❌ Error getting top keywords: {e}")
            return pd.DataFrame()
    
    def get_sentiment_by_subreddit(self, limit: int = 10) -> pd.DataFrame:
        """
        Get average sentiment score by subreddit.
        
        Args:
            limit (int): Number of top subreddits to analyze
        
        Returns:
            pd.DataFrame: DataFrame with subreddit sentiment stats
        """
        try:
            pipeline = [
                {'$match': {
                    'sentiment_score': {'$exists': True},
                    'subreddit': {'$exists': True}
                }},
                {'$group': {
                    '_id': '$subreddit',
                    'avg_sentiment': {'$avg': '$sentiment_score'},
                    'post_count': {'$sum': 1},
                    'positive_count': {
                        '$sum': {
                            '$cond': [{'$eq': ['$sentiment_label', 'Positive']}, 1, 0]
                        }
                    },
                    'negative_count': {
                        '$sum': {
                            '$cond': [{'$eq': ['$sentiment_label', 'Negative']}, 1, 0]
                        }
                    },
                    'neutral_count': {
                        '$sum': {
                            '$cond': [{'$eq': ['$sentiment_label', 'Neutral']}, 1, 0]
                        }
                    }
                }},
                {'$sort': {'post_count': -1}},
                {'$limit': limit}
            ]
            
            results = list(self.collection.aggregate(pipeline))
            df = pd.DataFrame(results)
            
            if not df.empty:
                df.columns = ['Subreddit', 'Avg_Sentiment', 'Posts', 'Positive', 'Negative', 'Neutral']
                df['Avg_Sentiment'] = df['Avg_Sentiment'].round(3)
                logger.info(f"📊 Sentiment by subreddit: {len(df)} subreddits")
            
            return df
            
        except Exception as e:
            logger.error(f"❌ Error getting sentiment by subreddit: {e}")
            return pd.DataFrame()
    
    def get_sentiment_over_time(self) -> pd.DataFrame:
        """
        Get average sentiment score over time.
        
        Returns:
            pd.DataFrame: DataFrame with daily sentiment trends
        """
        try:
            pipeline = [
                {'$match': {
                    'created_utc': {'$exists': True},
                    'sentiment_score': {'$exists': True}
                }},
                {'$addFields': {
                    'date': {
                        '$dateFromString': {
                            'dateString': '$created_utc',
                            'onError': None
                        }
                    }
                }},
                {'$match': {'date': {'$ne': None}}},
                {'$group': {
                    '_id': {
                        '$dateToString': {
                            'format': '%Y-%m-%d',
                            'date': '$date'
                        }
                    },
                    'avg_sentiment': {'$avg': '$sentiment_score'},
                    'count': {'$sum': 1}
                }},
                {'$sort': {'_id': 1}}
            ]
            
            results = list(self.collection.aggregate(pipeline))
            df = pd.DataFrame(results)
            
            if not df.empty:
                df.columns = ['Date', 'Avg_Sentiment', 'Count']
                df['Date'] = pd.to_datetime(df['Date'])
                df['Avg_Sentiment'] = df['Avg_Sentiment'].round(3)
                logger.info(f"📊 Sentiment over time: {len(df)} days")
            
            return df
            
        except Exception as e:
            logger.error(f"❌ Error getting sentiment over time: {e}")
            return pd.DataFrame()
    
    def get_post_engagement_stats(self) -> pd.DataFrame:
        """
        Get statistics about post engagement (scores and comments).
        
        Returns:
            pd.DataFrame: DataFrame with engagement statistics
        """
        try:
            pipeline = [
                {'$match': {
                    'score': {'$exists': True},
                    'num_comments': {'$exists': True}
                }},
                {'$group': {
                    '_id': None,
                    'avg_score': {'$avg': '$score'},
                    'max_score': {'$max': '$score'},
                    'min_score': {'$min': '$score'},
                    'avg_comments': {'$avg': '$num_comments'},
                    'max_comments': {'$max': '$num_comments'},
                    'total_posts': {'$sum': 1}
                }}
            ]
            
            results = list(self.collection.aggregate(pipeline))
            
            if results:
                stats = results[0]
                df = pd.DataFrame([{
                    'Metric': 'Score',
                    'Average': round(stats.get('avg_score', 0), 2),
                    'Maximum': stats.get('max_score', 0),
                    'Minimum': stats.get('min_score', 0)
                }, {
                    'Metric': 'Comments',
                    'Average': round(stats.get('avg_comments', 0), 2),
                    'Maximum': stats.get('max_comments', 0),
                    'Minimum': stats.get('min_comments', 0)
                }])
                
                logger.info(f"📊 Engagement stats calculated")
                return df
            
            return pd.DataFrame()
            
        except Exception as e:
            logger.error(f"❌ Error getting engagement stats: {e}")
            return pd.DataFrame()
    
    def get_comprehensive_report(self) -> Dict[str, pd.DataFrame]:
        """
        Generate a comprehensive report with all analytics.
        
        Returns:
            Dict[str, pd.DataFrame]: Dictionary of analysis results
        """
        try:
            logger.info("📊 Generating comprehensive report...")
            
            report = {
                'sentiment_distribution': self.get_sentiment_distribution(),
                'top_subreddits': self.get_top_subreddits(10),
                'top_authors': self.get_top_authors(10),
                'top_keywords': self.get_top_keywords(20),
                'posts_by_day': self.get_posts_by_day(),
                'posts_by_hour': self.get_posts_by_hour(),
                'sentiment_by_subreddit': self.get_sentiment_by_subreddit(10),
                'sentiment_over_time': self.get_sentiment_over_time(),
                'engagement_stats': self.get_post_engagement_stats()
            }
            
            logger.info("✅ Comprehensive report generated")
            return report
            
        except Exception as e:
            logger.error(f"❌ Error generating comprehensive report: {e}")
            return {}
    
    def export_report_to_csv(self, output_dir: str = "outputs"):
        """
        Export all analysis results to CSV files.
        
        Args:
            output_dir (str): Directory to save CSV files
        """
        try:
            os.makedirs(output_dir, exist_ok=True)
            
            report = self.get_comprehensive_report()
            
            for name, df in report.items():
                if not df.empty:
                    filepath = os.path.join(output_dir, f"{name}.csv")
                    df.to_csv(filepath, index=False)
                    logger.info(f"✅ Exported {name} to {filepath}")
            
            logger.info(f"✅ All reports exported to {output_dir}")
            
        except Exception as e:
            logger.error(f"❌ Error exporting reports: {e}")


def main():
    """Main function for data analysis operations."""
    try:
        analyzer = DataAnalysis()
        
        print("\n" + "="*60)
        print("📊 DATA ANALYSIS MODULE")
        print("="*60)
        
        # Generate comprehensive report
        print("\n📊 Generating comprehensive analytics report...")
        report = analyzer.get_comprehensive_report()
        
        # Display results
        for name, df in report.items():
            print(f"\n{'='*60}")
            print(f"📈 {name.replace('_', ' ').title()}")
            print('='*60)
            
            if not df.empty:
                print(df.to_string(index=False))
            else:
                print("  No data available")
        
        # Export to CSV
        print("\n💾 Exporting reports to CSV...")
        analyzer.export_report_to_csv("outputs")
        
        print("\n" + "="*60)
        print("✅ Data analysis completed successfully!")
        print("="*60 + "\n")
        
    except Exception as e:
        logger.error(f"❌ Error in main execution: {e}")
        print(f"\n❌ Error: {e}\n")


if __name__ == "__main__":
    main()
