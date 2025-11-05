"""
Quick Start Script for Social Media Analytics Project
Runs the complete data pipeline from ingestion to dashboard.
"""

import os
import sys
import logging
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def print_banner():
    """Print project banner."""
    banner = """
    ╔══════════════════════════════════════════════════════════╗
    ║                                                          ║
    ║     📊 SOCIAL MEDIA ANALYTICS - QUICK START 📊          ║
    ║                                                          ║
    ║     MongoDB Atlas + Reddit API + Streamlit              ║
    ║                                                          ║
    ╚══════════════════════════════════════════════════════════╝
    """
    print(banner)


def check_environment():
    """Check if environment is properly configured."""
    print("\n🔍 Checking environment configuration...")
    
    required_vars = [
        'MONGO_URI',
        'MONGO_DB_NAME',
        'MONGO_COLLECTION'
    ]
    
    missing_vars = []
    for var in required_vars:
        if not os.getenv(var):
            missing_vars.append(var)
    
    if missing_vars:
        print(f"\n❌ Missing environment variables: {', '.join(missing_vars)}")
        print("\n📝 Please copy .env.example to .env and configure your credentials.")
        return False
    
    print("✅ Environment configuration looks good!")
    return True


def run_pipeline():
    """Run the complete data pipeline."""
    try:
        print("\n" + "="*60)
        print("🚀 STARTING DATA PIPELINE")
        print("="*60)
        
        # Step 1: Data Ingestion
        print("\n📥 Step 1/4: Data Ingestion")
        print("-" * 60)
        from data_ingestion import RedditDataIngestion
        
        ingestion = RedditDataIngestion()
        
        # Optional: Delete data for specific subreddits
        delete_choice = input("\nDo you want to delete data for any subreddits before fetching? (y/n): ").lower()
        if delete_choice == 'y':
            subreddits_to_delete = input("Enter comma-separated names of subreddits to delete: ")
            if subreddits_to_delete:
                for subreddit in [s.strip() for s in subreddits_to_delete.split(',')]:
                    deleted = ingestion.delete_subreddit_data(subreddit)
                    print(f"Deleted {deleted} documents for r/{subreddit}.")

        # Prompt for subreddits to fetch (no default list, no y/n gate)
        subreddits_input = input("\nEnter comma-separated subreddits to fetch (leave blank to skip fetching): ")
        subreddits = [s.strip() for s in subreddits_input.split(',') if s.strip()] if subreddits_input else []

        if subreddits:
            limit_input = input("How many posts per subreddit? (default: 50): ")
            limit = int(limit_input) if limit_input.isdigit() else 50
            
            comment_limit_input = input("How many comments per post? (default: 5): ")
            comment_limit = int(comment_limit_input) if comment_limit_input.isdigit() else 5

            max_comments_total_input = input("Max total comments per subreddit (optional, blank = no cap): ")
            max_comments_total = int(max_comments_total_input) if max_comments_total_input.isdigit() else None

            print(f"\nFetching {limit} posts from {len(subreddits)} subreddits...")
            total_fetched = 0
            for subreddit in subreddits:
                print(f"Fetching from r/{subreddit}...")
                try:
                    fetched_count = ingestion.fetch_and_store_posts(
                        subreddit_name=subreddit, 
                        limit=limit, 
                        comment_limit=comment_limit,
                        max_comments_total=max_comments_total
                    )
                    total_fetched += fetched_count
                except Exception as e:
                    logger.error(f"❌ Failed to fetch from r/{subreddit}: {e}")
            print(f"\nTotal new documents fetched: {total_fetched}")


        # Check if we have data
        stats = ingestion.get_collection_stats()
        if stats.get('total_posts', 0) == 0:
            print("\nNo data found in the database. The pipeline cannot continue without data.")
            print("Please run the script again and enter at least one subreddit to fetch data from Reddit.")
            return False
        else:
            print(f"\n✅ Found {stats.get('total_posts', 0)} total posts in the database for processing.")
        
        # Step 2: Data Cleaning
        print("\n🧹 Step 2/4: Data Cleaning")
        print("-" * 60)
        from data_cleaning import DataCleaning
        
        cleaner = DataCleaning()
        cleaned_count = cleaner.clean_all_posts()
        print(f"✅ Cleaned {cleaned_count} posts")
        
        # Show cleaning stats
        cleaning_stats = cleaner.get_cleaning_stats()
        print(f"📊 Cleaning completion: {cleaning_stats.get('cleaning_percentage', 0):.1f}%")
        
        # Step 3: Sentiment Analysis
        print("\n🧠 Step 3/4: Sentiment Analysis")
        print("-" * 60)
        from sentiment_analysis import SentimentAnalysis
        
        sentiment = SentimentAnalysis()
        analyzed_count = sentiment.analyze_all_posts()
        print(f"✅ Analyzed sentiment for {analyzed_count} posts")
        
        # Generate word cloud
        print("☁️ Generating word cloud...")
        wordcloud_path = sentiment.generate_wordcloud()
        if wordcloud_path:
            print(f"✅ Word cloud saved: {wordcloud_path}")
        
        # Show sentiment stats
        sentiment_stats = sentiment.get_sentiment_stats()
        print(f"📊 Sentiment distribution: {sentiment_stats.get('sentiment_distribution', {})}")
        
        # Step 4: Data Analysis
        print("\n📊 Step 4/4: Data Analysis & Topic Modeling")
        print("-" * 60)
        from data_analysis import DataAnalysis
        from topic_modeling import TopicModeler

        # Standard Analysis
        analyzer = DataAnalysis()
        print("Generating comprehensive report...")
        analyzer.export_report_to_csv()
        print("✅ Standard reports exported to outputs/ directory")

        # Topic Modeling
        print("🔥 Identifying trending topics...")
        modeler = TopicModeler(ingestion.collection)
        topics_df = modeler.get_trending_topics()
        if not topics_df.empty:
            topics_path = os.path.join('outputs', 'trending_topics.csv')
            topics_df.to_csv(topics_path, index=False)
            print(f"✅ Trending topics report saved to {topics_path}")
        else:
            print("⚠️ Could not generate trending topics report.")
        
        # Pipeline complete
        print("\n" + "="*60)
        print("✅ DATA PIPELINE COMPLETED SUCCESSFULLY!")
        print("="*60)
        
        # Final summary
        print("\n📈 SUMMARY")
        print("-" * 60)
        final_stats = ingestion.get_collection_stats()
        print(f"Total Posts: {final_stats.get('total_posts', 0)}")
        print(f"Total Comments: {final_stats.get('total_comments', 0)}")
        print(f"Cleaned Posts: {cleaning_stats.get('cleaned_posts', 0)}")
        print(f"Analyzed Posts: {sentiment_stats.get('analyzed_posts', 0)}")
        print(f"Sentiment Distribution: {sentiment_stats.get('sentiment_distribution', {})}")
        
        return True
        
    except Exception as e:
        logger.error(f"❌ Pipeline failed: {e}")
        return False


def launch_dashboard():
    """Launch the Streamlit dashboard."""
    print("\n" + "="*60)
    print("🚀 LAUNCHING DASHBOARD")
    print("="*60)
    print("\n📊 Starting Streamlit application...")
    print("🌐 Dashboard will open in your browser at http://localhost:8501")
    print("\n⚠️ Press Ctrl+C to stop the dashboard\n")
    
    os.system("streamlit run dashboard_app.py")


def main():
    """Main function."""
    print_banner()
    
    # Check environment
    if not check_environment():
        print("\n❌ Please configure your environment before running the pipeline.")
        return
    
    # Ask user what to do
    print("\n📋 What would you like to do?")
    print("1. Run complete data pipeline")
    print("2. Launch dashboard only")
    print("3. Run pipeline and launch dashboard")
    print("4. Exit")
    
    choice = input("\nEnter your choice (1-4): ").strip()
    
    if choice == '1':
        success = run_pipeline()
        if success:
            print("\n✅ Pipeline completed! You can now launch the dashboard.")
            
    elif choice == '2':
        launch_dashboard()
        
    elif choice == '3':
        success = run_pipeline()
        if success:
            print("\n✅ Pipeline completed! Launching dashboard...")
            input("\nPress Enter to launch dashboard...")
            launch_dashboard()
        else:
            print("\n❌ Pipeline failed. Dashboard not launched.")
            
    elif choice == '4':
        print("\n👋 Goodbye!")
        
    else:
        print("\n❌ Invalid choice. Please run again and select 1-4.")


if __name__ == "__main__":
    main()
