#!/usr/bin/env python3
"""
Simple script to run the Streamlit Chat Agent UI
"""

import subprocess
import sys
import os

def main():
    """Run the Streamlit app"""
    print("🚀 Starting AI Chat Agent with RAG - Streamlit UI")
    print("-" * 50)
    
    # Check if we're in the right directory
    if not os.path.exists("streamlit_app.py"):
        print("❌ streamlit_app.py not found. Please run this from the project directory.")
        return
    
    # Check if integrated_chat_agent.py exists
    if not os.path.exists("integrated_chat_agent.py"):
        print("❌ integrated_chat_agent.py not found. Please ensure the chat agent is available.")
        return
    
    print("📋 Before starting, make sure you have:")
    print("  • OpenAI API key in config.py")
    print("  • Weather API key in config.py (optional)")
    print("  • Replicate API key in config.py (optional for image generation)")
    print("")
    
    try:
        # Run Streamlit
        subprocess.run([
            sys.executable, "-m", "streamlit", "run", "streamlit_app.py",
            "--server.port", "8501",
            "--server.headless", "false"
        ])
    except KeyboardInterrupt:
        print("\n👋 Streamlit app stopped.")
    except Exception as e:
        print(f"❌ Error running Streamlit: {e}")
        print("\n💡 Try installing Streamlit first:")
        print("   pip install streamlit")

if __name__ == "__main__":
    main()
