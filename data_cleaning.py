"""
Data Cleaning Module
Loads data from MongoDB and performs text preprocessing.
Uses regex, NLTK, and text cleaning utilities.
"""

import os
import sys
import logging
from typing import List, Dict
from dotenv import load_dotenv

# Add parent directory to path for imports
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from config.mongo_config import get_collection
from utils.text_utils import TextCleaner, clean_reddit_post

# Load environment variables
load_dotenv()

# Configure logging
logging.basicConfig(
    level=getattr(logging, os.getenv('LOG_LEVEL', 'INFO')),
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


class DataCleaning:
    """
    Handles data cleaning operations for Reddit posts.
    Cleans text data and updates MongoDB documents.
    """
    
    def __init__(self):
        """Initialize MongoDB connection and text cleaner."""
        try:
            self.collection = get_collection()
            self.text_cleaner = TextCleaner()
            logger.info("✅ Data cleaning module initialized")
        except Exception as e:
            logger.error(f"❌ Failed to initialize data cleaning: {e}")
            raise
    
    def clean_single_post(self, post: Dict) -> Dict:
        """
        Clean a single post's text data.
        
        Args:
            post (Dict): Post dictionary from MongoDB
        
        Returns:
            Dict: Post with cleaned text added
        """
        try:
            # Get text fields
            title = post.get('title', '')
            body = post.get('body', '') or post.get('selftext', '')
            
            # Combine text
            combined_text = f"{title} {body}".strip()
            
            if not combined_text:
                logger.warning(f"Empty text for post {post.get('id', 'unknown')}")
                post['clean_text'] = ''
                post['word_count'] = 0
                return post
            
            # Clean text
            clean_text = self.text_cleaner.clean_text(combined_text, remove_stopwords=True)
            
            # Add cleaned fields to post
            post['clean_text'] = clean_text
            post['word_count'] = len(clean_text.split())
            post['original_text'] = combined_text
            
            # Extract keywords
            keywords = self.text_cleaner.extract_keywords(combined_text, top_n=5)
            post['keywords'] = keywords
            
            logger.debug(f"✅ Cleaned post: {post.get('id')}")
            return post
            
        except Exception as e:
            logger.error(f"❌ Error cleaning post {post.get('id', 'unknown')}: {e}")
            return post
    
    def clean_all_posts(self, batch_size: int = 100) -> int:
        """
        Clean all posts in the database that don't have clean_text.
        
        Args:
            batch_size (int): Number of posts to process in each batch
        
        Returns:
            int: Number of posts cleaned
        """
        try:
            # Find posts (not comments) without clean_text field
            query = {'data_type': 'post', 'clean_text': {'$exists': False}}
            total_posts = self.collection.count_documents(query)
            
            if total_posts == 0:
                logger.info("✅ All posts are already cleaned")
                return 0
            
            logger.info(f"🧹 Cleaning {total_posts} posts...")
            
            cleaned_count = 0
            posts = self.collection.find(query).limit(batch_size)
            
            for post in posts:
                try:
                    # Clean the post
                    cleaned_post = self.clean_single_post(post)
                    
                    # Update in database
                    update_fields = {
                        'clean_text': cleaned_post.get('clean_text', ''),
                        'word_count': cleaned_post.get('word_count', 0),
                        'keywords': cleaned_post.get('keywords', [])
                    }
                    
                    self.collection.update_one(
                        {'_id': post['_id']},
                        {'$set': update_fields}
                    )
                    
                    cleaned_count += 1
                    
                    if cleaned_count % 10 == 0:
                        logger.info(f"  Progress: {cleaned_count}/{total_posts} posts cleaned")
                        
                except Exception as e:
                    logger.error(f"Error processing post {post.get('id', 'unknown')}: {e}")
                    continue
            
            logger.info(f"✅ Successfully cleaned {cleaned_count} posts")
            return cleaned_count
            
        except Exception as e:
            logger.error(f"❌ Error in clean_all_posts: {e}")
            return 0
    
    def clean_posts_by_subreddit(self, subreddit: str) -> int:
        """
        Clean posts from a specific subreddit.
        
        Args:
            subreddit (str): Subreddit name
        
        Returns:
            int: Number of posts cleaned
        """
        try:
            query = {
                'data_type': 'post',
                'subreddit': subreddit,
                'clean_text': {'$exists': False}
            }
            
            posts = self.collection.find(query)
            cleaned_count = 0
            
            for post in posts:
                try:
                    cleaned_post = self.clean_single_post(post)
                    
                    update_fields = {
                        'clean_text': cleaned_post.get('clean_text', ''),
                        'word_count': cleaned_post.get('word_count', 0),
                        'keywords': cleaned_post.get('keywords', [])
                    }
                    
                    self.collection.update_one(
                        {'_id': post['_id']},
                        {'$set': update_fields}
                    )
                    
                    cleaned_count += 1
                    
                except Exception as e:
                    logger.error(f"Error processing post: {e}")
                    continue
            
            logger.info(f"✅ Cleaned {cleaned_count} posts from r/{subreddit}")
            return cleaned_count
            
        except Exception as e:
            logger.error(f"❌ Error cleaning subreddit posts: {e}")
            return 0
    
    def reclean_all_posts(self, batch_size: int = 100) -> int:
        """
        Re-clean all posts, regardless of whether they have clean_text.
        Useful when cleaning algorithm is updated.
        
        Args:
            batch_size (int): Batch size for processing
        
        Returns:
            int: Number of posts re-cleaned
        """
        try:
            total_posts = self.collection.count_documents({'data_type': 'post'})
            logger.info(f"🔄 Re-cleaning {total_posts} posts...")
            
            cleaned_count = 0
            skip = 0
            
            while skip < total_posts:
                posts = list(self.collection.find({'data_type': 'post'}).skip(skip).limit(batch_size))
                
                for post in posts:
                    try:
                        cleaned_post = self.clean_single_post(post)
                        
                        update_fields = {
                            'clean_text': cleaned_post.get('clean_text', ''),
                            'word_count': cleaned_post.get('word_count', 0),
                            'keywords': cleaned_post.get('keywords', [])
                        }
                        
                        self.collection.update_one(
                            {'_id': post['_id']},
                            {'$set': update_fields}
                        )
                        
                        cleaned_count += 1
                        
                    except Exception as e:
                        logger.error(f"Error processing post: {e}")
                        continue
                
                skip += batch_size
                logger.info(f"  Progress: {min(skip, total_posts)}/{total_posts} posts processed")
            
            logger.info(f"✅ Successfully re-cleaned {cleaned_count} posts")
            return cleaned_count
            
        except Exception as e:
            logger.error(f"❌ Error in reclean_all_posts: {e}")
            return 0
    
    def get_cleaning_stats(self) -> Dict:
        """
        Get statistics about data cleaning status.
        
        Returns:
            Dict: Cleaning statistics
        """
        try:
            total_posts = self.collection.count_documents({'data_type': 'post'})
            cleaned_posts = self.collection.count_documents({'data_type': 'post', 'clean_text': {'$exists': True}})
            uncleaned_posts = total_posts - cleaned_posts
            
            # Get average word count
            pipeline = [
                {'$match': {'data_type': 'post', 'word_count': {'$exists': True}}},
                {'$group': {
                    '_id': None,
                    'avg_word_count': {'$avg': '$word_count'},
                    'max_word_count': {'$max': '$word_count'},
                    'min_word_count': {'$min': '$word_count'}
                }}
            ]
            
            word_stats = list(self.collection.aggregate(pipeline))
            
            stats = {
                'total_posts': total_posts,
                'cleaned_posts': cleaned_posts,
                'uncleaned_posts': uncleaned_posts,
                'cleaning_percentage': round((cleaned_posts / total_posts * 100), 2) if total_posts > 0 else 0
            }
            
            if word_stats:
                stats.update({
                    'avg_word_count': round(word_stats[0].get('avg_word_count', 0), 2),
                    'max_word_count': word_stats[0].get('max_word_count', 0),
                    'min_word_count': word_stats[0].get('min_word_count', 0)
                })
            
            logger.info(f"📊 Cleaning stats: {stats}")
            return stats
            
        except Exception as e:
            logger.error(f"❌ Error getting cleaning stats: {e}")
            return {}
    
    def remove_empty_posts(self) -> int:
        """
        Remove posts with empty or very short cleaned text.
        
        Returns:
            int: Number of posts removed
        """
        try:
            # Remove posts with less than 3 words in clean_text
            query = {
                'data_type': 'post',
                '$or': [
                    {'clean_text': ''},
                    {'clean_text': {'$exists': False}},
                    {'word_count': {'$lt': 3}}
                ]
            }
            
            result = self.collection.delete_many(query)
            deleted_count = result.deleted_count
            
            logger.info(f"🗑️ Removed {deleted_count} empty or very short posts")
            return deleted_count
            
        except Exception as e:
            logger.error(f"❌ Error removing empty posts: {e}")
            return 0
    
    def export_clean_data(self, output_file: str = 'clean_data.json', limit: int = 1000):
        """
        Export cleaned data to a JSON file.
        
        Args:
            output_file (str): Output filename
            limit (int): Maximum number of posts to export
        """
        try:
            import json
            
            posts = list(self.collection.find(
                {'data_type': 'post', 'clean_text': {'$exists': True}},
                {'_id': 0}
            ).limit(limit))
            
            with open(output_file, 'w', encoding='utf-8') as f:
                json.dump(posts, f, indent=2, ensure_ascii=False)
            
            logger.info(f"✅ Exported {len(posts)} cleaned posts to {output_file}")
            
        except Exception as e:
            logger.error(f"❌ Error exporting clean data: {e}")


def main():
    """Main function for data cleaning operations."""
    try:
        cleaner = DataCleaning()
        
        print("\n" + "="*60)
        print("🧹 DATA CLEANING MODULE")
        print("="*60)
        
        # Get initial stats
        print("\n📊 Current Cleaning Status:")
        stats = cleaner.get_cleaning_stats()
        for key, value in stats.items():
            print(f"  {key}: {value}")
        
        # Clean all uncleaned posts
        if stats.get('uncleaned_posts', 0) > 0:
            print(f"\n🧹 Starting cleaning process...")
            cleaned_count = cleaner.clean_all_posts(batch_size=100)
            print(f"✅ Cleaned {cleaned_count} posts")
        else:
            print("\n✅ All posts are already cleaned!")
        
        # Get final stats
        print("\n📊 Final Cleaning Status:")
        final_stats = cleaner.get_cleaning_stats()
        for key, value in final_stats.items():
            print(f"  {key}: {value}")
        
        # Show some examples
        print("\n📝 Sample Cleaned Posts:")
        sample_posts = list(cleaner.collection.find(
            {'clean_text': {'$exists': True}},
            {'title': 1, 'clean_text': 1, 'word_count': 1, 'keywords': 1}
        ).limit(3))
        
        for i, post in enumerate(sample_posts, 1):
            print(f"\n  Post {i}:")
            print(f"    Title: {post.get('title', '')[:60]}")
            print(f"    Clean Text: {post.get('clean_text', '')[:80]}...")
            print(f"    Word Count: {post.get('word_count', 0)}")
            print(f"    Keywords: {post.get('keywords', [])}")
        
        print("\n" + "="*60)
        print("✅ Data cleaning completed successfully!")
        print("="*60 + "\n")
        
    except Exception as e:
        logger.error(f"❌ Error in main execution: {e}")
        print(f"\n❌ Error: {e}\n")


if __name__ == "__main__":
    main()
