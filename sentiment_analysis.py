"""
Sentiment Analysis Module
Analyzes sentiment of Reddit posts using TextBlob and VADER.
Generates word clouds and updates MongoDB with sentiment data.
"""

import os
import sys
import logging
from typing import List, Dict
from collections import Counter
from dotenv import load_dotenv

# Add parent directory to path for imports
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from config.mongo_config import get_collection
from utils.sentiment_utils import SentimentAnalyzer, analyze_post_sentiment
from utils.visualization_utils import VisualizationHelper

# Load environment variables
load_dotenv()

# Configure logging
logging.basicConfig(
    level=getattr(logging, os.getenv('LOG_LEVEL', 'INFO')),
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


class SentimentAnalysis:
    """
    Handles sentiment analysis operations for Reddit posts.
    Analyzes sentiment and generates visualizations.
    """
    
    def __init__(self, output_dir: str = "outputs"):
        """
        Initialize sentiment analysis module.
        
        Args:
            output_dir (str): Directory for output files
        """
        try:
            self.collection = get_collection()
            self.sentiment_analyzer = SentimentAnalyzer()
            self.viz_helper = VisualizationHelper(output_dir)
            self.output_dir = output_dir
            os.makedirs(output_dir, exist_ok=True)
            logger.info("✅ Sentiment analysis module initialized")
        except Exception as e:
            logger.error(f"❌ Failed to initialize sentiment analysis: {e}")
            raise
    
    def analyze_single_post(self, post: Dict) -> Dict:
        """
        Analyze sentiment for a single post.
        
        Args:
            post (Dict): Post dictionary from MongoDB
        
        Returns:
            Dict: Post with sentiment fields added
        """
        try:
            # Use clean_text if available, otherwise use original text
            text = post.get('clean_text', '')
            
            if not text:
                title = post.get('title', '')
                body = post.get('body', '') or post.get('selftext', '')
                text = f"{title} {body}"
            
            if not text.strip():
                logger.warning(f"Empty text for post {post.get('id', 'unknown')}")
                post['sentiment_score'] = 0.0
                post['sentiment_label'] = 'Neutral'
                return post
            
            # Analyze sentiment using ensemble (VADER/TextBlob/Transformer)
            sentiment = self.sentiment_analyzer.analyze_sentiment(text, use_vader=True)
            
            # Add sentiment fields to post
            post['sentiment_score'] = sentiment['sentiment_score']
            post['sentiment_label'] = sentiment['sentiment_label']
            post['vader_positive'] = sentiment.get('positive_score', 0)
            post['vader_negative'] = sentiment.get('negative_score', 0)
            post['vader_neutral'] = sentiment.get('neutral_score', 0)
            post['sentiment_confidence'] = sentiment.get('confidence', 0.0)
            post['transformer_label'] = sentiment.get('transformer_label')
            post['transformer_score'] = sentiment.get('transformer_score')
            post['transformer_confidence'] = sentiment.get('transformer_confidence', 0.0)
            
            logger.debug(f"✅ Analyzed sentiment for post: {post.get('id')}")
            return post
            
        except Exception as e:
            logger.error(f"❌ Error analyzing post {post.get('id', 'unknown')}: {e}")
            return post
    
    def analyze_all_posts(self, batch_size: int = 100) -> int:
        """
        Analyze sentiment for all posts without sentiment data.
        
        Args:
            batch_size (int): Number of posts to process in each batch
        
        Returns:
            int: Number of posts analyzed
        """
        try:
            # Find posts without sentiment_score field
            query = {'sentiment_score': {'$exists': False}}
            total_posts = self.collection.count_documents(query)
            
            if total_posts == 0:
                logger.info("✅ All posts already have sentiment analysis")
                return 0
            
            logger.info(f"🧠 Analyzing sentiment for {total_posts} posts...")
            
            analyzed_count = 0
            posts = self.collection.find(query).limit(batch_size)
            
            for post in posts:
                try:
                    # Analyze the post
                    analyzed_post = self.analyze_single_post(post)
                    
                    # Update in database
                    update_fields = {
                        'sentiment_score': analyzed_post.get('sentiment_score', 0.0),
                        'sentiment_label': analyzed_post.get('sentiment_label', 'Neutral'),
                        'vader_positive': analyzed_post.get('vader_positive', 0),
                        'vader_negative': analyzed_post.get('vader_negative', 0),
                        'vader_neutral': analyzed_post.get('vader_neutral', 0),
                        'sentiment_confidence': analyzed_post.get('sentiment_confidence', 0.0),
                        'transformer_label': analyzed_post.get('transformer_label'),
                        'transformer_score': analyzed_post.get('transformer_score'),
                        'transformer_confidence': analyzed_post.get('transformer_confidence', 0.0)
                    }
                    
                    self.collection.update_one(
                        {'_id': post['_id']},
                        {'$set': update_fields}
                    )
                    
                    analyzed_count += 1
                    
                    if analyzed_count % 10 == 0:
                        logger.info(f"  Progress: {analyzed_count}/{total_posts} posts analyzed")
                        
                except Exception as e:
                    logger.error(f"Error processing post {post.get('id', 'unknown')}: {e}")
                    continue
            
            logger.info(f"✅ Successfully analyzed {analyzed_count} posts")
            return analyzed_count
            
        except Exception as e:
            logger.error(f"❌ Error in analyze_all_posts: {e}")
            return 0
    
    def analyze_posts_by_subreddit(self, subreddit: str) -> int:
        """
        Analyze sentiment for posts from a specific subreddit.
        
        Args:
            subreddit (str): Subreddit name
        
        Returns:
            int: Number of posts analyzed
        """
        try:
            query = {
                'subreddit': subreddit,
                'sentiment_score': {'$exists': False}
            }
            
            posts = self.collection.find(query)
            analyzed_count = 0
            
            for post in posts:
                try:
                    analyzed_post = self.analyze_single_post(post)
                    
                    update_fields = {
                        'sentiment_score': analyzed_post.get('sentiment_score', 0.0),
                        'sentiment_label': analyzed_post.get('sentiment_label', 'Neutral'),
                        'vader_positive': analyzed_post.get('vader_positive', 0),
                        'vader_negative': analyzed_post.get('vader_negative', 0),
                        'vader_neutral': analyzed_post.get('vader_neutral', 0),
                        'sentiment_confidence': analyzed_post.get('sentiment_confidence', 0.0),
                        'transformer_label': analyzed_post.get('transformer_label'),
                        'transformer_score': analyzed_post.get('transformer_score'),
                        'transformer_confidence': analyzed_post.get('transformer_confidence', 0.0)
                    }
                    
                    self.collection.update_one(
                        {'_id': post['_id']},
                        {'$set': update_fields}
                    )
                    
                    analyzed_count += 1
                    
                except Exception as e:
                    logger.error(f"Error processing post: {e}")
                    continue
            
            logger.info(f"✅ Analyzed {analyzed_count} posts from r/{subreddit}")
            return analyzed_count
            
        except Exception as e:
            logger.error(f"❌ Error analyzing subreddit posts: {e}")
            return 0
    
    def reanalyze_all_posts(self, batch_size: int = 100) -> int:
        """
        Re-analyze sentiment for all posts.
        Useful when sentiment algorithm is updated.
        
        Args:
            batch_size (int): Batch size for processing
        
        Returns:
            int: Number of posts re-analyzed
        """
        try:
            total_posts = self.collection.count_documents({})
            logger.info(f"🔄 Re-analyzing {total_posts} posts...")
            
            analyzed_count = 0
            skip = 0
            
            while skip < total_posts:
                posts = list(self.collection.find().skip(skip).limit(batch_size))
                
                for post in posts:
                    try:
                        analyzed_post = self.analyze_single_post(post)
                        
                        update_fields = {
                            'sentiment_score': analyzed_post.get('sentiment_score', 0.0),
                            'sentiment_label': analyzed_post.get('sentiment_label', 'Neutral'),
                            'vader_positive': analyzed_post.get('vader_positive', 0),
                            'vader_negative': analyzed_post.get('vader_negative', 0),
                            'vader_neutral': analyzed_post.get('vader_neutral', 0),
                            'sentiment_confidence': analyzed_post.get('sentiment_confidence', 0.0),
                            'transformer_label': analyzed_post.get('transformer_label'),
                            'transformer_score': analyzed_post.get('transformer_score'),
                            'transformer_confidence': analyzed_post.get('transformer_confidence', 0.0)
                        }
                        
                        self.collection.update_one(
                            {'_id': post['_id']},
                            {'$set': update_fields}
                        )
                        
                        analyzed_count += 1
                        
                    except Exception as e:
                        logger.error(f"Error processing post: {e}")
                        continue
                
                skip += batch_size
                logger.info(f"  Progress: {min(skip, total_posts)}/{total_posts} posts processed")
            
            logger.info(f"✅ Successfully re-analyzed {analyzed_count} posts")
            return analyzed_count
            
        except Exception as e:
            logger.error(f"❌ Error in reanalyze_all_posts: {e}")
            return 0
    
    def get_sentiment_stats(self) -> Dict:
        """
        Get statistics about sentiment analysis.
        
        Returns:
            Dict: Sentiment statistics
        """
        try:
            total_posts = self.collection.count_documents({})
            analyzed_posts = self.collection.count_documents({'sentiment_score': {'$exists': True}})
            
            # Get sentiment distribution
            pipeline = [
                {'$match': {'sentiment_label': {'$exists': True}}},
                {'$group': {
                    '_id': '$sentiment_label',
                    'count': {'$sum': 1}
                }},
                {'$sort': {'count': -1}}
            ]
            
            sentiment_dist = list(self.collection.aggregate(pipeline))
            sentiment_counts = {item['_id']: item['count'] for item in sentiment_dist}
            
            # Get average sentiment score
            avg_pipeline = [
                {'$match': {'sentiment_score': {'$exists': True}}},
                {'$group': {
                    '_id': None,
                    'avg_sentiment': {'$avg': '$sentiment_score'},
                    'max_sentiment': {'$max': '$sentiment_score'},
                    'min_sentiment': {'$min': '$sentiment_score'}
                }}
            ]
            
            avg_stats = list(self.collection.aggregate(avg_pipeline))
            
            stats = {
                'total_posts': total_posts,
                'analyzed_posts': analyzed_posts,
                'sentiment_distribution': sentiment_counts
            }
            
            if avg_stats:
                stats.update({
                    'avg_sentiment_score': round(avg_stats[0].get('avg_sentiment', 0), 3),
                    'max_sentiment_score': round(avg_stats[0].get('max_sentiment', 0), 3),
                    'min_sentiment_score': round(avg_stats[0].get('min_sentiment', 0), 3)
                })
            
            logger.info(f"📊 Sentiment stats: {stats}")
            return stats
            
        except Exception as e:
            logger.error(f"❌ Error getting sentiment stats: {e}")
            return {}
    
    def generate_wordcloud(self, sentiment_filter: str = None, output_filename: str = 'wordcloud.png') -> str:
        """
        Generate word cloud from posts' cleaned text.
        
        Args:
            sentiment_filter (str): Filter by sentiment ('Positive', 'Negative', 'Neutral', or None for all)
            output_filename (str): Output filename for word cloud
        
        Returns:
            str: Path to generated word cloud image
        """
        try:
            # Build query
            query = {'clean_text': {'$exists': True, '$ne': ''}}
            if sentiment_filter:
                query['sentiment_label'] = sentiment_filter
            
            # Get all cleaned text
            posts = self.collection.find(query, {'clean_text': 1})
            
            # Combine all text
            all_text = ' '.join([post.get('clean_text', '') for post in posts])
            
            if not all_text.strip():
                logger.warning("No text available for word cloud generation")
                return ""
            
            # Generate word cloud
            output_path = os.path.join(self.output_dir, output_filename)
            result_path = self.viz_helper.generate_wordcloud(all_text, output_path)
            
            logger.info(f"✅ Word cloud generated: {result_path}")
            return result_path
            
        except Exception as e:
            logger.error(f"❌ Error generating word cloud: {e}")
            return ""
    
    def generate_sentiment_wordclouds(self) -> Dict[str, str]:
        """
        Generate separate word clouds for each sentiment category.
        
        Returns:
            Dict[str, str]: Dictionary mapping sentiment to word cloud path
        """
        try:
            wordcloud_paths = {}
            
            sentiments = ['Positive', 'Negative', 'Neutral']
            
            for sentiment in sentiments:
                filename = f'wordcloud_{sentiment.lower()}.png'
                path = self.generate_wordcloud(sentiment_filter=sentiment, output_filename=filename)
                if path:
                    wordcloud_paths[sentiment] = path
            
            logger.info(f"✅ Generated {len(wordcloud_paths)} sentiment word clouds")
            return wordcloud_paths
            
        except Exception as e:
            logger.error(f"❌ Error generating sentiment word clouds: {e}")
            return {}
    
    def get_top_keywords(self, limit: int = 20) -> Dict[str, int]:
        """
        Get most common keywords across all posts.
        
        Args:
            limit (int): Number of top keywords to return
        
        Returns:
            Dict[str, int]: Dictionary of keywords and their frequencies
        """
        try:
            # Get all keywords from posts
            posts = self.collection.find(
                {'keywords': {'$exists': True}},
                {'keywords': 1}
            )
            
            # Count keyword frequencies
            keyword_counter = Counter()
            
            for post in posts:
                keywords = post.get('keywords', [])
                keyword_counter.update(keywords)
            
            # Get top keywords
            top_keywords = dict(keyword_counter.most_common(limit))
            
            logger.info(f"📊 Found {len(top_keywords)} top keywords")
            return top_keywords
            
        except Exception as e:
            logger.error(f"❌ Error getting top keywords: {e}")
            return {}
    
    def export_sentiment_data(self, output_file: str = 'sentiment_data.json', limit: int = 1000):
        """
        Export sentiment analysis results to JSON file.
        
        Args:
            output_file (str): Output filename
            limit (int): Maximum number of posts to export
        """
        try:
            import json
            
            posts = list(self.collection.find(
                {'sentiment_score': {'$exists': True}},
                {'_id': 0, 'title': 1, 'clean_text': 1, 'sentiment_score': 1, 
                 'sentiment_label': 1, 'subreddit': 1, 'created_utc': 1}
            ).limit(limit))
            
            output_path = os.path.join(self.output_dir, output_file)
            
            with open(output_path, 'w', encoding='utf-8') as f:
                json.dump(posts, f, indent=2, ensure_ascii=False)
            
            logger.info(f"✅ Exported {len(posts)} posts with sentiment to {output_path}")
            
        except Exception as e:
            logger.error(f"❌ Error exporting sentiment data: {e}")


def main():
    """Main function for sentiment analysis operations."""
    try:
        analyzer = SentimentAnalysis(output_dir="outputs")
        
        print("\n" + "="*60)
        print("🧠 SENTIMENT ANALYSIS MODULE")
        print("="*60)
        
        # Get initial stats
        print("\n📊 Current Sentiment Analysis Status:")
        stats = analyzer.get_sentiment_stats()
        for key, value in stats.items():
            print(f"  {key}: {value}")
        
        # Analyze all unanalyzed posts
        unanalyzed = stats.get('total_posts', 0) - stats.get('analyzed_posts', 0)
        if unanalyzed > 0:
            print(f"\n🧠 Starting sentiment analysis for {unanalyzed} posts...")
            analyzed_count = analyzer.analyze_all_posts(batch_size=100)
            print(f"✅ Analyzed {analyzed_count} posts")
        else:
            print("\n✅ All posts already have sentiment analysis!")
        
        # Get final stats
        print("\n📊 Final Sentiment Analysis Status:")
        final_stats = analyzer.get_sentiment_stats()
        for key, value in final_stats.items():
            print(f"  {key}: {value}")
        
        # Generate word cloud
        print("\n☁️ Generating word cloud...")
        wordcloud_path = analyzer.generate_wordcloud()
        if wordcloud_path:
            print(f"✅ Word cloud saved to: {wordcloud_path}")
        
        # Generate sentiment-specific word clouds
        print("\n☁️ Generating sentiment-specific word clouds...")
        sentiment_clouds = analyzer.generate_sentiment_wordclouds()
        for sentiment, path in sentiment_clouds.items():
            print(f"  {sentiment}: {path}")
        
        # Get top keywords
        print("\n🔑 Top Keywords:")
        top_keywords = analyzer.get_top_keywords(limit=15)
        for i, (keyword, count) in enumerate(list(top_keywords.items())[:15], 1):
            print(f"  {i}. {keyword}: {count}")
        
        # Show some examples
        print("\n📝 Sample Sentiment Analysis:")
        sample_posts = list(analyzer.collection.find(
            {'sentiment_score': {'$exists': True}},
            {'title': 1, 'sentiment_score': 1, 'sentiment_label': 1, 'clean_text': 1}
        ).limit(3))
        
        for i, post in enumerate(sample_posts, 1):
            print(f"\n  Post {i}:")
            print(f"    Title: {post.get('title', '')[:60]}")
            print(f"    Sentiment: {post.get('sentiment_label')} ({post.get('sentiment_score', 0):.3f})")
            print(f"    Text: {post.get('clean_text', '')[:80]}...")
        
        # Export sentiment data
        print("\n💾 Exporting sentiment data...")
        analyzer.export_sentiment_data()
        
        print("\n" + "="*60)
        print("✅ Sentiment analysis completed successfully!")
        print("="*60 + "\n")
        
    except Exception as e:
        logger.error(f"❌ Error in main execution: {e}")
        print(f"\n❌ Error: {e}\n")


if __name__ == "__main__":
    main()
