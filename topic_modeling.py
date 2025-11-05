"""
Topic Modeling Module
Identifies trending topics from collected Reddit data using TF-IDF.
"""

import logging
from typing import Optional

import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer

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
        self.default_vectorizer = TfidfVectorizer(
            max_df=0.9,
            min_df=2,
            max_features=1000,
            stop_words='english'
        )

    def get_trending_topics(self, top_n: int = 10, subreddit: Optional[str] = None) -> pd.DataFrame:
        """
        Identifies top trending keywords from recent, high-scoring posts.

        Args:
            top_n (int): The number of top keywords to return.
            subreddit (Optional[str]): Specific subreddit to filter posts. Uses all if None.

        Returns:
            pd.DataFrame: A DataFrame with 'topic' and 'score' columns, or empty if no data.
        """
        try:
            # Fetch recent, high-scoring posts that have been cleaned
            query = {
                "data_type": "post",
                "clean_text": {"$exists": True, "$ne": ""},
                "score": {"$gt": 10}
            }

            if subreddit:
                query["subreddit"] = subreddit
            # Sort by creation date and limit to recent 1000 posts for performance
            cursor = self.collection.find(query).sort("created_utc", -1).limit(1000)
            
            posts = list(cursor)
            if not posts:
                logger.warning("No suitable posts found for topic modeling.")
                return pd.DataFrame(columns=['topic', 'score'])

            df = pd.DataFrame(posts)

            df['clean_text'] = df['clean_text'].fillna('').astype(str).str.strip()
            df = df[df['clean_text'] != '']

            if df.empty:
                logger.warning("All candidate posts have empty clean_text after filtering.")
                return pd.DataFrame(columns=['topic', 'score'])

            doc_count = len(df)

            if doc_count < 5:
                vectorizer = TfidfVectorizer(
                    max_df=1.0,
                    min_df=1,
                    max_features=500,
                    stop_words='english'
                )
            else:
                vectorizer = self.default_vectorizer

            try:
                tfidf_matrix = vectorizer.fit_transform(df['clean_text'])
            except ValueError as ve:
                logger.warning(f"Primary TF-IDF fit failed ({ve}). Retrying with relaxed parameters.")
                vectorizer = TfidfVectorizer(
                    max_df=1.0,
                    min_df=1,
                    max_features=500,
                    stop_words='english'
                )
                try:
                    tfidf_matrix = vectorizer.fit_transform(df['clean_text'])
                except ValueError as ve_relaxed:
                    logger.error(f"TF-IDF fit failed even after relaxing parameters: {ve_relaxed}")
                    return pd.DataFrame(columns=['topic', 'score'])

            feature_names = vectorizer.get_feature_names_out()
            summed_tfidf = tfidf_matrix.sum(axis=0)
            scores = [(feature_names[i], summed_tfidf[0, i]) for i in range(len(feature_names))]
            
            # Create a DataFrame and sort
            topics_df = pd.DataFrame(scores, columns=['topic', 'score']).sort_values(
                by='score', ascending=False
            ).head(top_n)
            
            target = f"r/{subreddit}" if subreddit else "all subreddits"
            logger.info(f"Successfully identified {len(topics_df)} trending topics for {target}.")
            return topics_df

        except Exception as e:
            logger.error(f"❌ Error getting trending topics: {e}")
            return pd.DataFrame(columns=['topic', 'score'])
