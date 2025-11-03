"""
Data Ingestion Module
Collects Reddit posts and comments using PRAW and stores them in MongoDB Atlas.
Implements full CRUD operations for Reddit data.
"""

import os
import logging
from datetime import datetime
from typing import List, Dict, Optional
import praw
from praw.models import Submission, Comment
from pymongo.errors import DuplicateKeyError, PyMongoError
from dotenv import load_dotenv
import sys

# Add parent directory to path for imports
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from config.mongo_config import get_collection, MongoDBConnection

# Load environment variables
load_dotenv()

# Configure logging
logging.basicConfig(
    level=getattr(logging, os.getenv('LOG_LEVEL', 'INFO')),
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


class RedditDataIngestion:
    """
    Handles Reddit data collection and MongoDB storage.
    Implements CRUD operations for Reddit posts and comments.
    """
    
    def __init__(self):
        """Initialize Reddit API and MongoDB connection."""
        # Initialize Reddit API (PRAW)
        self.reddit = self._initialize_reddit()
        
        # Initialize MongoDB connection
        try:
            self.collection = get_collection()
            # Create unique index on Reddit post ID to prevent duplicates
            self.collection.create_index("id", unique=True)
            logger.info("✅ MongoDB collection initialized with unique index")
        except Exception as e:
            logger.error(f"❌ Failed to initialize MongoDB: {e}")
            raise
    
    def _initialize_reddit(self) -> praw.Reddit:
        """
        Initialize PRAW Reddit API client.
        
        Returns:
            praw.Reddit: Authenticated Reddit instance
        """
        try:
            client_id = os.getenv('REDDIT_CLIENT_ID')
            client_secret = os.getenv('REDDIT_CLIENT_SECRET')
            user_agent = os.getenv('REDDIT_USER_AGENT')
            
            if not all([client_id, client_secret, user_agent]):
                raise ValueError("Reddit API credentials not found in environment variables")
            
            reddit = praw.Reddit(
                client_id=client_id,
                client_secret=client_secret,
                user_agent=user_agent
            )
            
            # Test connection
            reddit.user.me()
            logger.info("✅ Successfully authenticated with Reddit API")
            return reddit
            
        except Exception as e:
            logger.error(f"❌ Failed to initialize Reddit API: {e}")
            logger.info("💡 Will use read-only mode with example data")
            # Return a read-only instance for demo purposes
            return praw.Reddit(
                client_id=client_id or "demo",
                client_secret=client_secret or "demo",
                user_agent=user_agent or "demo_app",
                check_for_async=False
            )

    def submission_to_dict(self, submission: Submission) -> Dict:
        """
        Convert Reddit submission to dictionary format.
        
        Args:
            submission (Submission): PRAW submission object
        
        Returns:
            Dict: Dictionary representation of the submission
        """
        return {
            'id': submission.id,
            'title': submission.title,
            'body': submission.selftext,
            'selftext': submission.selftext,
            'author': str(submission.author) if submission.author else '[deleted]',
            'subreddit': str(submission.subreddit),
            'score': submission.score,
            'upvote_ratio': submission.upvote_ratio,
            'num_comments': submission.num_comments,
            'created_utc': datetime.fromtimestamp(submission.created_utc).isoformat(),
            'url': submission.url,
            'permalink': f"https://reddit.com{submission.permalink}",
            'is_self': submission.is_self,
            'link_flair_text': submission.link_flair_text,
            'collected_at': datetime.now().isoformat(),
            'data_type': 'post'
        }
    
    def comment_to_dict(self, comment: Comment, post_id: str, subreddit_name: str) -> Dict:
        """
        Convert Reddit comment to dictionary format.
        
        Args:
            comment (Comment): PRAW comment object
            post_id (str): ID of the parent post
            subreddit_name (str): Name of the subreddit the comment belongs to
        
        Returns:
            Dict: Dictionary representation of the comment
        """
        return {
            'id': comment.id,
            'post_id': post_id,
            'subreddit': subreddit_name,
            'body': comment.body,
            'author': str(comment.author) if comment.author else '[deleted]',
            'score': comment.score,
            'created_utc': datetime.fromtimestamp(comment.created_utc).isoformat(),
            'parent_id': comment.parent_id,
            'permalink': f"https://reddit.com{comment.permalink}",
            'collected_at': datetime.now().isoformat(),
            'data_type': 'comment'
        }
    
    # ==================== CREATE Operations ====================
    
    def insert_post(self, post_dict: Dict) -> bool:
        """
        Insert a single Reddit post into MongoDB.
        
        Args:
            post_dict (Dict): Post data dictionary
        
        Returns:
            bool: True if successful, False otherwise
        """
        try:
            self.collection.insert_one(post_dict)
            logger.info(f"✅ Inserted post: {post_dict.get('id')} - {post_dict.get('title', '')[:50]}")
            return True
        except DuplicateKeyError:
            logger.warning(f"⚠️ Post {post_dict.get('id')} already exists, skipping")
            return False
        except Exception as e:
            logger.error(f"❌ Error inserting post: {e}")
            return False
    
    def insert_posts_bulk(self, posts: List[Dict]) -> int:
        """
        Insert multiple posts into MongoDB.
        
        Args:
            posts (List[Dict]): List of post dictionaries
        
        Returns:
            int: Number of successfully inserted posts
        """
        inserted_count = 0
        for post in posts:
            if self.insert_post(post):
                inserted_count += 1
        
        logger.info(f"✅ Bulk insert completed: {inserted_count}/{len(posts)} posts inserted")
        return inserted_count
    
    def fetch_and_store_posts(
        self,
        subreddit_name: str,
        limit: int = 50,
        comment_limit: int = 10
    ) -> int:
        """
        Fetch top posts and their comments from a subreddit and store them.
        
        Args:
            subreddit_name (str): Name of the subreddit
            limit (int): Number of posts to fetch
            comment_limit (int): Number of top comments to fetch for each post
        
        Returns:
            int: Number of new documents (posts + comments) inserted
        """
        logger.info(f"Fetching {limit} posts and up to {comment_limit} comments each from r/{subreddit_name}...")
        try:
            subreddit = self.reddit.subreddit(subreddit_name)
            # Fetch newest posts to reduce duplicates
            hot_posts = subreddit.new(limit=limit)
            
            total_inserted_count = 0
            
            for post in hot_posts:
                # Use a list to hold post and its comments for a single bulk insert
                documents_to_insert = []
                
                # Add the post to our list
                post_dict = self.submission_to_dict(post)
                documents_to_insert.append(post_dict)
                
                # Fetch and add comments for the post
                if comment_limit > 0:
                    try:
                        # PRAW lazy loads comments, so we access the submission again
                        submission = self.reddit.submission(id=post.id)
                        submission.comments.replace_more(limit=0)  # Remove "load more comments" links
                        
                        comment_count = 0
                        for top_level_comment in submission.comments:
                            if comment_count >= comment_limit:
                                break
                            comment_dict = self.comment_to_dict(top_level_comment, post.id, post_dict['subreddit'])
                            documents_to_insert.append(comment_dict)
                            comment_count += 1
                            
                    except Exception as e:
                        logger.error(f"❌ Error fetching comments for post {post.id}: {e}")

                # Bulk insert post and its comments
                if documents_to_insert:
                    try:
                        # Use ordered=False to continue inserting even if one document fails (e.g., duplicate)
                        result = self.collection.insert_many(documents_to_insert, ordered=False)
                        inserted_now = len(result.inserted_ids)
                        total_inserted_count += inserted_now
                        logger.debug(f"✅ Inserted {inserted_now} documents for post {post.id}")
                    except PyMongoError as e:
                        # This can happen if the post already exists and we try to re-insert it.
                        # We log it but continue to the next post.
                        logger.warning(f"⚠️ Skipping insert for post {post.id} due to DB error (likely duplicate): {e}")

            logger.info(f"✅ Finished fetching from r/{subreddit_name}. Total documents inserted: {total_inserted_count}")
            return total_inserted_count
            
        except Exception as e:
            logger.error(f"❌ Failed to fetch posts from r/{subreddit_name}: {e}")
            return 0
    
    def _fetch_comments(self, submission: Submission, limit: int = 10):
        """
        Fetch and store comments for a submission.
        
        Args:
            submission (Submission): Reddit submission object
            limit (int): Maximum number of comments to fetch
        """
        try:
            submission.comments.replace_more(limit=0)
            comments = submission.comments.list()[:limit]

            subreddit_name = str(submission.subreddit)
            for comment in comments:
                if isinstance(comment, Comment):
                    comment_dict = self.comment_to_dict(comment, submission.id, subreddit_name)
                    # Use direct insert to collection since this is a comment document
                    try:
                        self.collection.insert_one(comment_dict)
                    except DuplicateKeyError:
                        logger.warning(f"⚠️ Comment {comment.id} already exists, skipping")
                    except Exception as e:
                        logger.error(f"❌ Error inserting comment {comment.id}: {e}")
                    
        except Exception as e:
            logger.error(f"Error fetching comments for {submission.id}: {e}")
    
    # ==================== READ Operations ====================
    
    def read_all_posts(self, limit: int = 100) -> List[Dict]:
        """
        Read all posts from MongoDB.
        
        Args:
            limit (int): Maximum number of posts to retrieve
        
        Returns:
            List[Dict]: List of post dictionaries
        """
        try:
            posts = list(self.collection.find({'data_type': 'post'}).limit(limit))
            logger.info(f"📖 Retrieved {len(posts)} posts")
            return posts
        except Exception as e:
            logger.error(f"❌ Error reading posts: {e}")
            return []
    
    def read_posts_by_subreddit(self, subreddit: str, limit: int = 100) -> List[Dict]:
        """
        Read posts from a specific subreddit.
        
        Args:
            subreddit (str): Subreddit name
            limit (int): Maximum number of posts to retrieve
        
        Returns:
            List[Dict]: List of post dictionaries
        """
        try:
            posts = list(self.collection.find({'subreddit': subreddit}).limit(limit))
            logger.info(f"📖 Retrieved {len(posts)} posts from r/{subreddit}")
            return posts
        except Exception as e:
            logger.error(f"❌ Error reading posts: {e}")
            return []
    
    def read_posts_by_keyword(self, keyword: str, limit: int = 100) -> List[Dict]:
        """
        Read posts containing a specific keyword in title or body.
        
        Args:
            keyword (str): Keyword to search for
            limit (int): Maximum number of posts to retrieve
        
        Returns:
            List[Dict]: List of post dictionaries
        """
        try:
            query = {
                '$or': [
                    {'title': {'$regex': keyword, '$options': 'i'}},
                    {'body': {'$regex': keyword, '$options': 'i'}}
                ]
            }
            posts = list(self.collection.find(query).limit(limit))
            logger.info(f"📖 Retrieved {len(posts)} posts with keyword '{keyword}'")
            return posts
        except Exception as e:
            logger.error(f"❌ Error searching posts: {e}")
            return []
    
    def get_post_by_id(self, post_id: str) -> Optional[Dict]:
        """
        Read a single post by ID.
        
        Args:
            post_id (str): Reddit post ID
        
        Returns:
            Optional[Dict]: Post dictionary or None
        """
        try:
            post = self.collection.find_one({'id': post_id})
            if post:
                logger.info(f"📖 Retrieved post: {post_id}")
            return post
        except Exception as e:
            logger.error(f"❌ Error reading post {post_id}: {e}")
            return None
    
    # ==================== UPDATE Operations ====================
    
    def update_post(self, post_id: str, update_fields: Dict) -> bool:
        """
        Update a post with new fields.
        
        Args:
            post_id (str): Reddit post ID
            update_fields (Dict): Dictionary of fields to update
        
        Returns:
            bool: True if successful, False otherwise
        """
        try:
            result = self.collection.update_one(
                {'id': post_id},
                {'$set': update_fields}
            )
            
            if result.modified_count > 0:
                logger.info(f"✅ Updated post: {post_id}")
                return True
            else:
                logger.warning(f"⚠️ No changes made to post: {post_id}")
                return False
                
        except Exception as e:
            logger.error(f"❌ Error updating post {post_id}: {e}")
            return False
    
    def update_posts_bulk(self, filter_query: Dict, update_fields: Dict) -> int:
        """
        Update multiple posts matching a filter.
        
        Args:
            filter_query (Dict): MongoDB filter query
            update_fields (Dict): Fields to update
        
        Returns:
            int: Number of posts updated
        """
        try:
            result = self.collection.update_many(
                filter_query,
                {'$set': update_fields}
            )
            logger.info(f"✅ Bulk update: {result.modified_count} posts updated")
            return result.modified_count
        except Exception as e:
            logger.error(f"❌ Error in bulk update: {e}")
            return 0
    
    # ==================== DELETE Operations ====================
    
    def delete_post(self, post_id: str) -> bool:
        """
        Delete a single post by ID.
        
        Args:
            post_id (str): Reddit post ID
        
        Returns:
            bool: True if successful, False otherwise
        """
        try:
            result = self.collection.delete_one({'id': post_id})
            
            if result.deleted_count > 0:
                logger.info(f"🗑️ Deleted post: {post_id}")
                return True
            else:
                logger.warning(f"⚠️ Post not found: {post_id}")
                return False
                
        except Exception as e:
            logger.error(f"❌ Error deleting post {post_id}: {e}")
            return False
    
    def delete_posts_by_subreddit(self, subreddit: str) -> int:
        """
        Delete all posts from a specific subreddit.
        
        Args:
            subreddit (str): Subreddit name
        
        Returns:
            int: Number of posts deleted
        """
        try:
            result = self.collection.delete_many({'subreddit': subreddit})
            logger.info(f"🗑️ Deleted {result.deleted_count} posts from r/{subreddit}")
            return result.deleted_count
        except Exception as e:
            logger.error(f"❌ Error deleting posts: {e}")
            return 0
    
    def delete_subreddit_data(self, subreddit_name: str) -> int:
        """
        Delete ALL documents related to a subreddit: posts and comments.

        Strategy:
        - Find all post IDs for the subreddit (data_type='post' and subreddit match)
        - Delete comments either tagged with the subreddit or whose post_id is in those IDs
        - Delete the posts themselves

        Returns the total number of documents deleted.
        """
        if not subreddit_name:
            logger.warning("Subreddit name cannot be empty.")
            return 0

        try:
            logger.warning(f"🔥 Deleting all data for subreddit: r/{subreddit_name}...")

            # 1) Collect post IDs for this subreddit
            post_cursor = self.collection.find(
                {'data_type': 'post', 'subreddit': subreddit_name},
                {'id': 1}
            )
            post_ids = [doc.get('id') for doc in post_cursor if doc.get('id')]

            # 2) Delete comments linked to these posts OR tagged with the subreddit
            comment_filter = {
                'data_type': 'comment',
                '$or': [
                    {'subreddit': subreddit_name},
                    {'post_id': {'$in': post_ids}} if post_ids else {'_id': {'$exists': False}},
                    {'permalink': {'$regex': f"/r/{subreddit_name}/", '$options': 'i'}}
                ]
            }
            comment_result = self.collection.delete_many(comment_filter)

            # 3) Delete the posts themselves
            post_result = self.collection.delete_many({'data_type': 'post', 'subreddit': subreddit_name})

            total_deleted = comment_result.deleted_count + post_result.deleted_count

            logger.info(
                f"✅ Deleted {post_result.deleted_count} posts and {comment_result.deleted_count} comments for r/{subreddit_name} (total {total_deleted})."
            )
            return total_deleted
        except PyMongoError as e:
            logger.error(f"❌ Failed to delete data for r/{subreddit_name}: {e}")
            return 0

    def delete_duplicates(self) -> int:
        """
        Remove duplicate posts (shouldn't happen with unique index, but useful for cleanup).
        
        Returns:
            int: Number of duplicates removed
        """
        try:
            pipeline = [
                {'$group': {
                    '_id': '$id',
                    'count': {'$sum': 1},
                    'docs': {'$push': '$_id'}
                }},
                {'$match': {
                    'count': {'$gt': 1}
                }}
            ]
            
            duplicates = list(self.collection.aggregate(pipeline))
            deleted_count = 0
            
            for dup in duplicates:
                # Keep first, delete rest
                docs_to_delete = dup['docs'][1:]
                result = self.collection.delete_many({'_id': {'$in': docs_to_delete}})
                deleted_count += result.deleted_count
            
            logger.info(f"🗑️ Removed {deleted_count} duplicate posts")
            return deleted_count
            
        except Exception as e:
            logger.error(f"❌ Error removing duplicates: {e}")
            return 0
    
    def get_collection_stats(self) -> Dict:
        """
        Get statistics about the collection.
        
        Returns:
            Dict: Collection statistics
        """
        try:
            total_posts = self.collection.count_documents({'data_type': 'post'})
            total_comments = self.collection.count_documents({'data_type': 'comment'})
            
            # Get subreddit distribution
            subreddit_pipeline = [
                {'$match': {'data_type': 'post'}},
                {'$group': {'_id': '$subreddit', 'count': {'$sum': 1}}},
                {'$sort': {'count': -1}},
                {'$limit': 5}
            ]
            top_subreddits = list(self.collection.aggregate(subreddit_pipeline))
            
            stats = {
                'total_posts': total_posts,
                'total_comments': total_comments,
                'total_documents': total_posts + total_comments,
                'top_subreddits': top_subreddits
            }
            
            logger.info(f"📊 Collection stats: {stats}")
            return stats
            
        except Exception as e:
            logger.error(f"❌ Error getting stats: {e}")
            return {}


def create_example_data():
    """Create example Reddit data for testing when API is not available."""
    return [
        {
            'id': 'example1',
            'title': 'Artificial Intelligence is transforming healthcare',
            'body': 'AI and machine learning are revolutionizing medical diagnosis and treatment.',
            'selftext': 'AI and machine learning are revolutionizing medical diagnosis and treatment.',
            'author': 'tech_enthusiast',
            'subreddit': 'technology',
            'score': 1250,
            'upvote_ratio': 0.95,
            'num_comments': 87,
            'created_utc': '2025-11-01T10:30:00',
            'url': 'https://reddit.com/example1',
            'permalink': '/r/technology/comments/example1',
            'is_self': True,
            'link_flair_text': 'AI/ML',
            'collected_at': datetime.now().isoformat(),
            'data_type': 'post'
        },
        {
            'id': 'example2',
            'title': 'Why I hate the new software update',
            'body': 'This update broke everything. Terrible experience, very disappointing.',
            'selftext': 'This update broke everything. Terrible experience, very disappointing.',
            'author': 'frustrated_user',
            'subreddit': 'technology',
            'score': 432,
            'upvote_ratio': 0.78,
            'num_comments': 156,
            'created_utc': '2025-11-02T14:20:00',
            'url': 'https://reddit.com/example2',
            'permalink': '/r/technology/comments/example2',
            'is_self': True,
            'link_flair_text': 'Discussion',
            'collected_at': datetime.now().isoformat(),
            'data_type': 'post'
        },
        {
            'id': 'example3',
            'title': 'Machine Learning resources for beginners',
            'body': 'Check out these great resources for learning ML and deep learning.',
            'selftext': 'Check out these great resources for learning ML and deep learning.',
            'author': 'ml_teacher',
            'subreddit': 'machinelearning',
            'score': 2100,
            'upvote_ratio': 0.98,
            'num_comments': 234,
            'created_utc': '2025-11-03T09:15:00',
            'url': 'https://reddit.com/example3',
            'permalink': '/r/machinelearning/comments/example3',
            'is_self': True,
            'link_flair_text': 'Resources',
            'collected_at': datetime.now().isoformat(),
            'data_type': 'post'
        }
    ]


if __name__ == "__main__":
    # Example usage
    try:
        ingestion = RedditDataIngestion()
        
        # Display collection stats
        print("\n📊 Current Collection Stats:")
        stats = ingestion.get_collection_stats()
        for key, value in stats.items():
            print(f"  {key}: {value}")
        
        # Example: Insert sample data
        print("\n💾 Inserting example data...")
        example_posts = create_example_data()
        count = ingestion.insert_posts_bulk(example_posts)
        print(f"✅ Inserted {count} example posts")
        
        # Example: Fetch from Reddit (uncomment when API is configured)
        # ingestion.fetch_and_store_posts('technology', limit=10, sort_by='hot')
        
        # Example: Read posts
        print("\n📖 Reading posts...")
        posts = ingestion.read_all_posts(limit=5)
        for post in posts:
            print(f"  - {post.get('title', '')[:60]}")
        
    except Exception as e:
        logger.error(f"Error in main execution: {e}")
