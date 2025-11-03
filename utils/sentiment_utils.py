"""
Sentiment Analysis Utilities
Contains functions for sentiment analysis using TextBlob and VADER.
"""

import logging
from typing import Dict, Tuple, Optional
from textblob import TextBlob
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer

# Configure logging
logger = logging.getLogger(__name__)


class SentimentAnalyzer:
    """
    Sentiment analysis using both TextBlob and VADER.
    Provides polarity scores and sentiment classification.
    """
    
    def __init__(self):
        """Initialize sentiment analyzers."""
        self.vader = SentimentIntensityAnalyzer()
        logger.info("Sentiment analyzers initialized")
    
    def analyze_with_textblob(self, text: str) -> Dict[str, float]:
        """
        Analyze sentiment using TextBlob.
        
        Args:
            text (str): Input text
        
        Returns:
            Dict[str, float]: Dictionary with polarity and subjectivity scores
        """
        try:
            if not text or not isinstance(text, str):
                return {'polarity': 0.0, 'subjectivity': 0.0}
            
            blob = TextBlob(text)
            return {
                'polarity': blob.sentiment.polarity,
                'subjectivity': blob.sentiment.subjectivity
            }
        except Exception as e:
            logger.error(f"Error in TextBlob analysis: {e}")
            return {'polarity': 0.0, 'subjectivity': 0.0}
    
    def analyze_with_vader(self, text: str) -> Dict[str, float]:
        """
        Analyze sentiment using VADER.
        
        Args:
            text (str): Input text
        
        Returns:
            Dict[str, float]: Dictionary with sentiment scores
                - neg: Negative score
                - neu: Neutral score
                - pos: Positive score
                - compound: Compound score (-1 to 1)
        """
        try:
            if not text or not isinstance(text, str):
                return {'neg': 0.0, 'neu': 1.0, 'pos': 0.0, 'compound': 0.0}
            
            scores = self.vader.polarity_scores(text)
            return scores
        except Exception as e:
            logger.error(f"Error in VADER analysis: {e}")
            return {'neg': 0.0, 'neu': 1.0, 'pos': 0.0, 'compound': 0.0}
    
    def get_sentiment_label(self, compound_score: float, threshold: float = 0.05) -> str:
        """
        Convert compound score to sentiment label.
        
        Args:
            compound_score (float): VADER compound score (-1 to 1)
            threshold (float): Threshold for neutral classification
        
        Returns:
            str: 'Positive', 'Negative', or 'Neutral'
        """
        if compound_score >= threshold:
            return 'Positive'
        elif compound_score <= -threshold:
            return 'Negative'
        else:
            return 'Neutral'
    
    def analyze_sentiment(self, text: str, use_vader: bool = True) -> Dict:
        """
        Perform comprehensive sentiment analysis.
        
        Args:
            text (str): Input text
            use_vader (bool): If True, use VADER; otherwise use TextBlob
        
        Returns:
            Dict: Dictionary containing:
                - sentiment_score: Main sentiment score
                - sentiment_label: 'Positive', 'Negative', or 'Neutral'
                - detailed_scores: Detailed sentiment breakdown
        """
        if use_vader:
            vader_scores = self.analyze_with_vader(text)
            compound = vader_scores['compound']
            
            return {
                'sentiment_score': compound,
                'sentiment_label': self.get_sentiment_label(compound),
                'vader_scores': vader_scores,
                'positive_score': vader_scores['pos'],
                'negative_score': vader_scores['neg'],
                'neutral_score': vader_scores['neu']
            }
        else:
            textblob_scores = self.analyze_with_textblob(text)
            polarity = textblob_scores['polarity']
            
            return {
                'sentiment_score': polarity,
                'sentiment_label': self.get_sentiment_label(polarity),
                'textblob_scores': textblob_scores,
                'polarity': polarity,
                'subjectivity': textblob_scores['subjectivity']
            }
    
    def analyze_sentiment_hybrid(self, text: str) -> Dict:
        """
        Perform hybrid sentiment analysis using both TextBlob and VADER.
        
        Args:
            text (str): Input text
        
        Returns:
            Dict: Combined sentiment analysis results
        """
        vader_result = self.analyze_with_vader(text)
        textblob_result = self.analyze_with_textblob(text)
        
        # Use VADER compound score as primary
        compound = vader_result['compound']
        
        # Average the sentiment scores
        avg_score = (compound + textblob_result['polarity']) / 2
        
        return {
            'sentiment_score': avg_score,
            'sentiment_label': self.get_sentiment_label(avg_score),
            'vader_compound': compound,
            'textblob_polarity': textblob_result['polarity'],
            'vader_scores': vader_result,
            'textblob_scores': textblob_result
        }


def analyze_post_sentiment(post_dict: dict, text_field: str = 'clean_text') -> dict:
    """
    Analyze sentiment for a Reddit post dictionary.
    
    Args:
        post_dict (dict): Dictionary containing post data
        text_field (str): Field name containing text to analyze
    
    Returns:
        dict: Updated post dictionary with sentiment fields
    """
    analyzer = SentimentAnalyzer()
    
    # Get text to analyze
    text = post_dict.get(text_field, '')
    
    # If clean_text is empty, try original text fields
    if not text:
        title = post_dict.get('title', '')
        body = post_dict.get('body', '') or post_dict.get('selftext', '')
        text = f"{title} {body}"
    
    # Analyze sentiment
    sentiment = analyzer.analyze_sentiment(text, use_vader=True)
    
    # Add sentiment fields to post
    post_dict['sentiment_score'] = sentiment['sentiment_score']
    post_dict['sentiment_label'] = sentiment['sentiment_label']
    post_dict['vader_positive'] = sentiment['positive_score']
    post_dict['vader_negative'] = sentiment['negative_score']
    post_dict['vader_neutral'] = sentiment['neutral_score']
    
    return post_dict


def batch_analyze_sentiments(posts: list, text_field: str = 'clean_text') -> list:
    """
    Analyze sentiment for multiple posts.
    
    Args:
        posts (list): List of post dictionaries
        text_field (str): Field name containing text to analyze
    
    Returns:
        list: List of posts with sentiment analysis
    """
    analyzer = SentimentAnalyzer()
    analyzed_posts = []
    
    for post in posts:
        try:
            analyzed_post = analyze_post_sentiment(post, text_field)
            analyzed_posts.append(analyzed_post)
        except Exception as e:
            logger.error(f"Error analyzing post {post.get('id', 'unknown')}: {e}")
            analyzed_posts.append(post)
    
    logger.info(f"Analyzed sentiment for {len(analyzed_posts)} posts")
    return analyzed_posts


if __name__ == "__main__":
    # Test sentiment analysis
    test_texts = [
        "I absolutely love this new technology! It's amazing!",
        "This is terrible and disappointing. I hate it.",
        "The weather is okay today, nothing special.",
        "AI and machine learning are transforming the world! 🚀"
    ]
    
    analyzer = SentimentAnalyzer()
    
    print("Sentiment Analysis Results:\n")
    for text in test_texts:
        result = analyzer.analyze_sentiment(text)
        print(f"Text: {text}")
        print(f"Score: {result['sentiment_score']:.3f}")
        print(f"Label: {result['sentiment_label']}")
        print("-" * 50)
