# Dependencies Information

## Python Version Compatibility
- **Minimum Required**: Python 3.9+
- **Tested on**: Python 3.13.5

## Pinned Library Versions

All dependencies have been pinned to specific versions that are tested and compatible with Python 3.9+:

### Core AI & LLM Dependencies
- `openai==1.51.0` - Latest OpenAI Python client
- `langchain==0.3.7` - LangChain framework
- `langchain-community==0.3.7` - Community integrations
- `langchain-core==0.3.15` - Core LangChain functionality
- `langchain-openai==0.2.8` - OpenAI integration for LangChain
- `langchain-chroma==0.1.4` - ChromaDB integration
- `langchain-text-splitters==0.3.2` - Text splitting utilities

### Vector Database & Storage
- `chromadb==0.5.20` - Vector database for embeddings

### Document Processing
- `pymupdf==1.24.13` - PDF processing library

### Text Processing
- `tiktoken==0.8.0` - OpenAI tokenization

### API & Web Services
- `requests==2.32.3` - HTTP library for API calls
- `replicate==0.32.1` - Image generation API client
- `streamlit==1.39.0` - Web UI framework

## Installation

To install all dependencies with pinned versions:

```bash
pip install -r requirements.txt
```

## Notes

- All versions have been tested for compatibility with Python 3.9+
- SQLite3 is used for database functionality (built into Python)
- These pinned versions ensure reproducible deployments across different environments
- The dependencies have been cleaned to remove unused libraries for a leaner installation

## Verification

The application has been tested and verified to work correctly with these pinned versions.
