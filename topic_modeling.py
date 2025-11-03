"""
Topic Modeling Module
Identifies trending topics from collected Reddit data using TF-IDF.
"""

import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
import logging

logger = logging.getLogger(__name__)

class TopicModeler:
    """
    Extracts trending topics from a collection of documents.
    """
    def __init__(self, collection):
        """
        Initializes the TopicModeler.

        Args:
            collection: The MongoDB collection object.
        """
        self.collection = collection
        self.vectorizer = TfidfVectorizer(
            max_df=0.9, 
            min_df=2, 
            max_features=1000, 
            stop_words='english'
        )

    def get_trending_topics(self, top_n: int = 10) -> pd.DataFrame:
        """
        Identifies top trending keywords from recent, high-scoring posts.

        Args:
            top_n (int): The number of top keywords to return.

        Returns:
            pd.DataFrame: A DataFrame with 'topic' and 'score' columns, or empty if no data.
        """
        try:
            # Fetch recent, high-scoring posts that have been cleaned
            query = {
                "data_type": "post",
                "cleaned_text": {"$exists": True, "$ne": ""},
                "score": {"$gt": 10} # Focus on somewhat popular posts
            }
            # Sort by creation date and limit to recent 1000 posts for performance
            cursor = self.collection.find(query).sort("created_utc", -1).limit(1000)
            
            posts = list(cursor)
            if not posts:
                logger.warning("No suitable posts found for topic modeling.")
                return pd.DataFrame(columns=['topic', 'score'])

            df = pd.DataFrame(posts)
            
            # Get TF-IDF scores
            tfidf_matrix = self.vectorizer.fit_transform(df['cleaned_text'])
            feature_names = self.vectorizer.get_feature_names_out()
            
            # Sum scores across all documents
            summed_tfidf = tfidf_matrix.sum(axis=0)
            scores = [(feature_names[i], summed_tfidf[0, i]) for i in range(len(feature_names))]
            
            # Create a DataFrame and sort
            topics_df = pd.DataFrame(scores, columns=['topic', 'score']).sort_values(
                by='score', ascending=False
            ).head(top_n)
            
            logger.info(f"Successfully identified {len(topics_df)} trending topics.")
            return topics_df

        except Exception as e:
            logger.error(f"❌ Error getting trending topics: {e}")
            return pd.DataFrame(columns=['topic', 'score'])
