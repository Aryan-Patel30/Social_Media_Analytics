"""
MongoDB Configuration Module
Handles database connection and configuration settings.
"""

import os
import logging
from pymongo import MongoClient
from pymongo.errors import ConnectionFailure, ServerSelectionTimeoutError
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# Configure logging
logging.basicConfig(
    level=getattr(logging, os.getenv('LOG_LEVEL', 'INFO')),
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


class MongoDBConnection:
    """
    MongoDB connection handler with singleton pattern.
    Manages database connections and provides access to collections.
    """
    
    _instance = None
    _client = None
    _db = None
    _collection = None
    
    def __new__(cls):
        """Implement singleton pattern for database connection."""
        if cls._instance is None:
            cls._instance = super(MongoDBConnection, cls).__new__(cls)
        return cls._instance
    
    def __init__(self):
        """Initialize MongoDB connection parameters."""
        if self._client is None:
            self.mongo_uri = os.getenv('MONGO_URI')
            self.db_name = os.getenv('MONGO_DB_NAME', 'social_media')
            self.collection_name = os.getenv('MONGO_COLLECTION', 'reddit_posts')
            
            if not self.mongo_uri:
                logger.error("MONGO_URI not found in environment variables")
                raise ValueError("MongoDB URI is required. Please set MONGO_URI in .env file")
    
    def connect(self):
        """
        Establish connection to MongoDB Atlas.
        
        Returns:
            tuple: (client, database, collection)
        
        Raises:
            ConnectionFailure: If unable to connect to MongoDB
        """
        try:
            if self._client is None:
                logger.info("Connecting to MongoDB Atlas...")
                self._client = MongoClient(
                    self.mongo_uri,
                    serverSelectionTimeoutMS=5000,
                    connectTimeoutMS=10000
                )
                
                # Verify connection
                self._client.admin.command('ping')
                logger.info("✅ Successfully connected to MongoDB Atlas")
                
                # Get database and collection
                self._db = self._client[self.db_name]
                self._collection = self._db[self.collection_name]
                
                logger.info(f"Using database: {self.db_name}, collection: {self.collection_name}")
            
            return self._client, self._db, self._collection
            
        except (ConnectionFailure, ServerSelectionTimeoutError) as e:
            logger.error(f"❌ Failed to connect to MongoDB: {str(e)}")
            raise
        except Exception as e:
            logger.error(f"❌ Unexpected error during MongoDB connection: {str(e)}")
            raise
    
    def get_collection(self):
        """
        Get the MongoDB collection object.
        
        Returns:
            Collection: PyMongo collection object
        """
        if self._collection is None:
            self.connect()
        return self._collection
    
    def get_database(self):
        """
        Get the MongoDB database object.
        
        Returns:
            Database: PyMongo database object
        """
        if self._db is None:
            self.connect()
        return self._db
    
    def close(self):
        """Close the MongoDB connection."""
        if self._client:
            self._client.close()
            self._client = None
            self._db = None
            self._collection = None
            logger.info("MongoDB connection closed")


def get_mongo_connection():
    """
    Helper function to get MongoDB connection.
    
    Returns:
        tuple: (client, database, collection)
    """
    mongo = MongoDBConnection()
    return mongo.connect()


def get_collection():
    """
    Helper function to get MongoDB collection.
    
    Returns:
        Collection: PyMongo collection object
    """
    mongo = MongoDBConnection()
    return mongo.get_collection()


if __name__ == "__main__":
    # Test MongoDB connection
    try:
        client, db, collection = get_mongo_connection()
        print("✅ MongoDB connection test successful!")
        print(f"Database: {db.name}")
        print(f"Collection: {collection.name}")
        
        # Close connection
        mongo = MongoDBConnection()
        mongo.close()
    except Exception as e:
        print(f"❌ Connection test failed: {str(e)}")
