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
        from data_ingestion import RedditDataIngestion, create_example_data
        
        ingestion = RedditDataIngestion()
        
        # Check if we have data
        stats = ingestion.get_collection_stats()
        if stats.get('total_posts', 0) == 0:
            print("No data found. Inserting example data...")
            example_posts = create_example_data()
            count = ingestion.insert_posts_bulk(example_posts)
            print(f"✅ Inserted {count} example posts")
        else:
            print(f"✅ Found {stats.get('total_posts', 0)} existing posts")
        
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
        print("\n📊 Step 4/4: Data Analysis")
        print("-" * 60)
        from data_analysis import DataAnalysis
        
        analyzer = DataAnalysis()
        print("Generating comprehensive report...")
        report = analyzer.get_comprehensive_report()
        
        print(f"✅ Generated {len(report)} analysis reports")
        
        # Export reports
        print("💾 Exporting reports to CSV...")
        analyzer.export_report_to_csv()
        print("✅ Reports exported to outputs/ directory")
        
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
