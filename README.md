# AI Chat Agent with RAG

A comprehensive AI chatbot application built with Streamlit, featuring Retrieval-Augmented Generation (RAG), document processing, weather information, database queries, and image generation capabilities.

## Features

- 💬 **General Conversation**: Natural language interactions powered by OpenAI GPT
- 📄 **Document Q&A (RAG)**: Upload and query PDF documents using vector embeddings
- 🌤️ **Weather Information**: Real-time weather data integration
- 🗄️ **Database Queries**: Query structured data from SQLite databases
- 🎨 **Image Generation**: AI-powered image creation using Replicate
- 🎯 **Event Recommendations**: Smart event suggestions and recommendations

## Architecture

- **Frontend**: Streamlit web interface
- **Backend**: Integrated chat agent with multiple capabilities
- **Vector Database**: ChromaDB for document embeddings
- **LLM**: OpenAI GPT models via LangChain
- **Document Processing**: PyMuPDF, pypdf, unstructured
- **Image Generation**: Replicate API integration

## Installation

1. **Clone the repository**:
   ```bash
   git clone <your-repo-url>
   cd "NUS AI Capstone Project Submission"
   ```

2. **Install dependencies**:
   ```bash
   pip install -r requirements.txt
   ```

3. **Set up configuration**:
   ```bash
   # Copy the template and add your API keys
   cp config_template.py config.py
   ```
   
   Then edit `config.py` with your actual API keys:
   - **OpenAI API key** (required): Get from [OpenAI Platform](https://platform.openai.com/api-keys)
   - **Weather API key** (optional): Get from [OpenWeatherMap](https://openweathermap.org/api)
   - **Replicate API key** (optional): Get from [Replicate Account](https://replicate.com/account)

## Usage

### Running the Application

**Option 1: Using the run script**
```bash
python run_streamlit.py
```

**Option 2: Direct Streamlit command**
```bash
streamlit run streamlit_app.py
```

The application will start on `http://localhost:8501`

### Using the Features

1. **Document Upload**: Use the sidebar to upload PDF files
2. **Chat Interface**: Ask questions in natural language
3. **Example Questions**: Try the pre-built example buttons
4. **Multi-modal Responses**: Get text, images, and structured data

### Example Queries

- **Document Q&A**: "What are the key financial highlights in the uploaded document?"
- **Weather**: "What's the weather like in Tokyo?"
- **Database**: "Show me all employees earning more than 70000"
- **Image Generation**: "Generate an image of a beautiful sunset over mountains"
- **Events**: "Recommend events for tomorrow in Singapore"

## Project Structure

```
├── integrated_chat_agent.py    # Main chat agent implementation
├── streamlit_app.py           # Streamlit web interface
├── run_streamlit.py           # Application launcher script
├── requirements.txt           # Python dependencies
├── config.py                  # API keys configuration (create this)
├── *.db                      # SQLite database files
└── README.md                 # This file
```

## Dependencies

- **AI/LLM**: langchain, openai, langchain-community, langchain-openai
- **Vector DB**: chromadb, langchain-chroma, sentence-transformers
- **Documents**: pymupdf, pypdf, unstructured, python-docx
- **Web UI**: streamlit
- **APIs**: requests, replicate
- **Utilities**: tiktoken, python-dotenv

## Configuration Notes

- Ensure you have valid API keys for the services you want to use
- OpenAI API key is required for core functionality
- Weather and Replicate APIs are optional but enhance capabilities
- Database files (*.db) are included for demo purposes

## License

This project is part of NUS AI Capstone Project Submission.

## Troubleshooting

### Common Issues

1. **Import Errors**: Make sure all dependencies are installed with `pip install -r requirements.txt`
2. **API Key Errors**: Verify your API keys are correctly set in `config.py`
3. **Permission Errors**: Ensure you have write permissions in the project directory

### Python Version Compatibility

- **Supported**: Python 3.8 - 3.13
- **Recommended**: Python 3.9 or higher for optimal performance
- All dependencies are compatible with the latest Python versions
