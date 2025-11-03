"""
Text Processing Utilities
Contains functions for text cleaning and preprocessing.
"""

import re
import logging
from typing import List, Optional
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize

# Configure logging
logger = logging.getLogger(__name__)

# Download required NLTK data
try:
    nltk.data.find('corpora/stopwords.zip')
except LookupError:
    logger.info("Downloading NLTK stopwords...")
    nltk.download('stopwords', quiet=True)

try:
    nltk.data.find('tokenizers/punkt.zip')
except LookupError:
    logger.info("Downloading NLTK punkt tokenizer...")
    nltk.download('punkt', quiet=True)


class TextCleaner:
    """
    Text cleaning and preprocessing utility class.
    Handles removal of URLs, special characters, emojis, and stopwords.
    """
    
    def __init__(self, language: str = 'english'):
        """
        Initialize TextCleaner with specified language.
        
        Args:
            language (str): Language for stopwords (default: 'english')
        """
        self.language = language
        try:
            self.stop_words = set(stopwords.words(language))
        except Exception as e:
            logger.warning(f"Could not load stopwords for {language}: {e}")
            self.stop_words = set()
    
    def remove_urls(self, text: str) -> str:
        """
        Remove URLs from text.
        
        Args:
            text (str): Input text
        
        Returns:
            str: Text with URLs removed
        """
        url_pattern = re.compile(r'https?://\S+|www\.\S+')
        return url_pattern.sub('', text)
    
    def remove_emojis(self, text: str) -> str:
        """
        Remove emojis from text.
        
        Args:
            text (str): Input text
        
        Returns:
            str: Text with emojis removed
        """
        emoji_pattern = re.compile(
            "["
            "\U0001F600-\U0001F64F"  # emoticons
            "\U0001F300-\U0001F5FF"  # symbols & pictographs
            "\U0001F680-\U0001F6FF"  # transport & map symbols
            "\U0001F1E0-\U0001F1FF"  # flags (iOS)
            "\U00002702-\U000027B0"
            "\U000024C2-\U0001F251"
            "]+", flags=re.UNICODE
        )
        return emoji_pattern.sub('', text)
    
    def remove_special_chars(self, text: str) -> str:
        """
        Remove special characters and keep only alphanumeric and spaces.
        
        Args:
            text (str): Input text
        
        Returns:
            str: Text with special characters removed
        """
        # Keep letters, numbers, and spaces
        return re.sub(r'[^a-zA-Z0-9\s]', '', text)
    
    def remove_extra_spaces(self, text: str) -> str:
        """
        Remove extra whitespaces from text.
        
        Args:
            text (str): Input text
        
        Returns:
            str: Text with normalized whitespace
        """
        return ' '.join(text.split())
    
    def remove_stopwords(self, text: str) -> str:
        """
        Remove stopwords from text.
        
        Args:
            text (str): Input text
        
        Returns:
            str: Text with stopwords removed
        """
        try:
            words = word_tokenize(text.lower())
            filtered_words = [word for word in words if word not in self.stop_words]
            return ' '.join(filtered_words)
        except Exception as e:
            logger.warning(f"Error removing stopwords: {e}")
            # Fallback to simple split
            words = text.lower().split()
            filtered_words = [word for word in words if word not in self.stop_words]
            return ' '.join(filtered_words)
    
    def clean_text(self, text: Optional[str], remove_stopwords: bool = True) -> str:
        """
        Perform complete text cleaning pipeline.
        
        Args:
            text (Optional[str]): Input text to clean
            remove_stopwords (bool): Whether to remove stopwords
        
        Returns:
            str: Cleaned text
        """
        if not text or not isinstance(text, str):
            return ""
        
        # Step 1: Remove URLs
        text = self.remove_urls(text)
        
        # Step 2: Remove emojis
        text = self.remove_emojis(text)
        
        # Step 3: Convert to lowercase
        text = text.lower()
        
        # Step 4: Remove special characters
        text = self.remove_special_chars(text)
        
        # Step 5: Remove extra spaces
        text = self.remove_extra_spaces(text)
        
        # Step 6: Remove stopwords (optional)
        if remove_stopwords:
            text = self.remove_stopwords(text)
        
        return text.strip()
    
    def extract_keywords(self, text: str, top_n: int = 10) -> List[str]:
        """
        Extract top keywords from text (words that aren't stopwords).
        
        Args:
            text (str): Input text
            top_n (int): Number of top keywords to return
        
        Returns:
            List[str]: List of top keywords
        """
        cleaned = self.clean_text(text, remove_stopwords=True)
        words = cleaned.split()
        
        # Count word frequency
        word_freq = {}
        for word in words:
            if len(word) > 2:  # Only words with more than 2 characters
                word_freq[word] = word_freq.get(word, 0) + 1
        
        # Sort by frequency and get top N
        sorted_words = sorted(word_freq.items(), key=lambda x: x[1], reverse=True)
        return [word for word, freq in sorted_words[:top_n]]


def clean_reddit_post(post_dict: dict) -> dict:
    """
    Clean text fields in a Reddit post dictionary.
    
    Args:
        post_dict (dict): Dictionary containing Reddit post data
    
    Returns:
        dict: Dictionary with added 'clean_text' field
    """
    cleaner = TextCleaner()
    
    # Combine title and body for cleaning
    title = post_dict.get('title', '')
    body = post_dict.get('body', '') or post_dict.get('selftext', '')
    
    combined_text = f"{title} {body}"
    clean_text = cleaner.clean_text(combined_text)
    
    post_dict['clean_text'] = clean_text
    return post_dict


if __name__ == "__main__":
    # Test text cleaning
    test_text = """
    Check out this amazing AI technology! 🚀 
    Visit https://example.com for more info.
    #AI #MachineLearning #Tech
    """
    
    cleaner = TextCleaner()
    cleaned = cleaner.clean_text(test_text)
    
    print("Original text:")
    print(test_text)
    print("\nCleaned text:")
    print(cleaned)
    print("\nKeywords:")
    print(cleaner.extract_keywords(test_text))
