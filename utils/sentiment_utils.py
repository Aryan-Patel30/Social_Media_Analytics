"""
Sentiment Analysis Utilities
Contains functions for sentiment analysis using TextBlob, VADER, and optional
transformer-based models.
"""

import logging
import os
from typing import Dict, Optional
from textblob import TextBlob
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer

try:
    from transformers import pipeline
    _TRANSFORMERS_AVAILABLE = True
except ImportError:
    _TRANSFORMERS_AVAILABLE = False

# Configure logging
logger = logging.getLogger(__name__)


TRANSFORMER_LABEL_MAP = {
    'LABEL_0': 'Negative',
    'LABEL_1': 'Neutral',
    'LABEL_2': 'Positive',
    'NEGATIVE': 'Negative',
    'NEUTRAL': 'Neutral',
    'POSITIVE': 'Positive'
}


class SentimentAnalyzer:
    """
    Sentiment analysis using both TextBlob and VADER.
    Provides polarity scores and sentiment classification.
    """
    
    def __init__(self):
        """Initialize sentiment analyzers."""
        self.vader = SentimentIntensityAnalyzer()
        self.transformer = self._load_transformer()
        logger.info("Sentiment analyzers initialized")

    def _load_transformer(self):
        """Load transformer-based sentiment pipeline if available."""
        if not _TRANSFORMERS_AVAILABLE:
            logger.info("Transformers library not available; skipping transformer sentiment model.")
            return None

        model_name = os.getenv('SENTIMENT_MODEL_NAME', 'cardiffnlp/twitter-roberta-base-sentiment-latest')
        try:
            classifier = pipeline('sentiment-analysis', model=model_name, tokenizer=model_name)
            logger.info(f"Transformer sentiment model loaded: {model_name}")
            return classifier
        except Exception as exc:
            logger.warning(f"Could not load transformer sentiment model '{model_name}': {exc}")
            return None
    
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

    def analyze_with_transformer(self, text: str) -> Optional[Dict[str, float]]:
        """Analyze sentiment using a transformer model when available."""
        if not text or not isinstance(text, str) or not self.transformer:
            return None

        try:
            result = self.transformer(text, truncation=True)[0]
            raw_label = result.get('label', '').upper()
            score = float(result.get('score', 0.0))

            label = TRANSFORMER_LABEL_MAP.get(raw_label, None)
            if label is None:
                if 'NEG' in raw_label:
                    label = 'Negative'
                elif 'POS' in raw_label:
                    label = 'Positive'
                elif 'NEU' in raw_label:
                    label = 'Neutral'
                else:
                    label = 'Positive' if score >= 0.5 else 'Negative'

            if label == 'Neutral':
                polarity = 0.0
            elif label == 'Positive':
                polarity = score
            else:
                polarity = -score

            return {
                'label': label,
                'confidence': score,
                'polarity': polarity
            }
        except Exception as exc:
            logger.warning(f"Transformer sentiment inference failed: {exc}")
            return None
    
    def get_sentiment_label(self, compound_score: float, threshold: float = 0.15) -> str:
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
        if not text or not isinstance(text, str):
            return {
                'sentiment_score': 0.0,
                'sentiment_label': 'Neutral',
                'vader_scores': {'neg': 0.0, 'neu': 1.0, 'pos': 0.0, 'compound': 0.0},
                'textblob_scores': {'polarity': 0.0, 'subjectivity': 0.0},
                'positive_score': 0.0,
                'negative_score': 0.0,
                'neutral_score': 1.0,
                'confidence': 0.0
            }

        # Normalize and pre-process
        text = text.strip()
        if not text:
            return {
                'sentiment_score': 0.0,
                'sentiment_label': 'Neutral',
                'vader_scores': {'neg': 0.0, 'neu': 1.0, 'pos': 0.0, 'compound': 0.0},
                'textblob_scores': {'polarity': 0.0, 'subjectivity': 0.0},
                'positive_score': 0.0,
                'negative_score': 0.0,
                'neutral_score': 1.0,
                'confidence': 0.0
            }

        # Run base analyzers
        vader_scores = self.analyze_with_vader(text)
        textblob_scores = self.analyze_with_textblob(text)

        compound = vader_scores.get('compound', 0.0)
        polarity = textblob_scores.get('polarity', 0.0)
        subjectivity = textblob_scores.get('subjectivity', 0.0)

        # Basic ensemble: weighted average
        weighted_score = (0.65 * compound) + (0.35 * polarity)

        # Token length heuristic
        tokens = text.split()
        token_count = len(tokens)

        # Short texts are noisier — prefer stronger signals
        if token_count < 5:
            # If VADER has a clear positive/negative dominance, keep it; otherwise neutralize
            if max(vader_scores.get('pos', 0.0), vader_scores.get('neg', 0.0)) < 0.45:
                weighted_score = 0.0

        # Question detection: many questions are neutral in sentiment
        is_question = (
            ('?' in text) or (
                tokens
                and tokens[0].lower() in [
                    'who', 'what', 'when', 'where', 'why', 'how', 'do', 'did',
                    'does', 'is', 'are', 'can', 'could', 'would', 'should'
                ]
            )
        )
        if is_question:
            # If VADER doesn't show a very strong signal, treat as neutral for questions
            if max(vader_scores.get('pos', 0.0), vader_scores.get('neg', 0.0)) < 0.6:
                weighted_score = 0.0

        # Small lexicon of strong negative tokens to correct headlines/titles
        NEG_STRONG = {
            'cuts', 'cut', 'layoff', 'layoffs', 'laying', 'assault', 'threaten', 'threatens', 'threat',
            'monopoly', 'cannot', "can't", 'cant', 'bankrupt', 'collapse', 'kill', 'problem', 'decline', 'danger'
        }
        text_lower = text.lower()
        has_neg_lex = any(tok in text_lower for tok in NEG_STRONG)

        # If lexicon indicates negativity and model isn't strongly positive, nudge negative
        if has_neg_lex and vader_scores.get('pos', 0.0) - vader_scores.get('neg', 0.0) < 0.25:
            # push weighted_score towards negative if not already
            weighted_score = min(weighted_score, -0.22)

        # If VADER strongly indicates a class, prefer it.
        # However, don't override question-neutralization unless VADER is very strong.
        vader_diff = vader_scores.get('pos', 0.0) - vader_scores.get('neg', 0.0)
        if vader_diff > 0.45 and (not is_question or vader_diff > 0.6):
            weighted_score = max(weighted_score, vader_scores.get('pos', 0.0))
        if -vader_diff > 0.45 and (not is_question or (-vader_diff) > 0.6):
            weighted_score = min(weighted_score, -vader_scores.get('neg', 0.0))

        # If subjectivity is very low, lean towards neutral unless strong compound
        # or we have a negative lexicon signal (e.g., 'layoffs', 'cuts')
        if subjectivity < 0.15 and abs(weighted_score) < 0.3 and not has_neg_lex:
            weighted_score = 0.0

        # Blend with transformer output if available
        transformer_result = self.analyze_with_transformer(text)
        transformer_label = None
        transformer_conf = 0.0
        transformer_score = 0.0

        if transformer_result:
            transformer_label = transformer_result['label']
            transformer_conf = transformer_result['confidence']
            transformer_score = transformer_result['polarity']

            # Blend transformer score with heuristic score
            weighted_score = (0.7 * transformer_score) + (0.3 * weighted_score)

            # If question and transformer confidence is modest, keep neutral
            if is_question and transformer_conf < 0.75:
                weighted_score = 0.0

        # Clip
        weighted_score = max(min(weighted_score, 1.0), -1.0)

        # Determine label with a slightly higher neutral band
        # If subjectivity is low, widen neutral threshold
        neutral_thresh = 0.18 if subjectivity >= 0.25 else 0.25
        label = self.get_sentiment_label(weighted_score, threshold=neutral_thresh)

        # Allow transformer to override label when confident
        if transformer_result and transformer_conf >= 0.55:
            label = transformer_label

        # Override label if strong negative lexicon present (headline-like signals)
        if has_neg_lex and weighted_score < -0.18:
            label = 'Negative'

        # Confidence: combination of VADER non-neutral, transformer confidence, and subjectivity
        confidence = max(vader_scores.get('pos', 0.0), vader_scores.get('neg', 0.0))
        confidence = max(confidence, transformer_conf)
        if token_count < 5 or is_question:
            confidence *= 0.6
        confidence = max(min(confidence, 1.0), 0.0)

        return {
            'sentiment_score': weighted_score,
            'sentiment_label': label,
            'vader_scores': vader_scores,
            'textblob_scores': textblob_scores,
            'positive_score': vader_scores.get('pos', 0.0),
            'negative_score': vader_scores.get('neg', 0.0),
            'neutral_score': vader_scores.get('neu', 0.0),
            'confidence': confidence,
            'transformer_label': transformer_label,
            'transformer_score': transformer_score,
            'transformer_confidence': transformer_conf
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


def analyze_post_sentiment(post_dict: dict, text_field: str = 'clean_text', analyzer: Optional[SentimentAnalyzer] = None) -> dict:
    """
    Analyze sentiment for a Reddit post dictionary.
    
    Args:
        post_dict (dict): Dictionary containing post data
        text_field (str): Field name containing text to analyze
        analyzer (SentimentAnalyzer, optional): Reusable analyzer instance
    
    Returns:
        dict: Updated post dictionary with sentiment fields
    """
    analyzer = analyzer or SentimentAnalyzer()
    
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
    post_dict['sentiment_confidence'] = sentiment['confidence']
    post_dict['transformer_label'] = sentiment.get('transformer_label')
    post_dict['transformer_score'] = sentiment.get('transformer_score')
    post_dict['transformer_confidence'] = sentiment.get('transformer_confidence')
    
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
            analyzed_post = analyze_post_sentiment(post, text_field, analyzer=analyzer)
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
