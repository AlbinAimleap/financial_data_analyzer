
# RAG-Based Financial Document Analyzer

A Python application that uses Retrieval-Augmented Generation (RAG) to analyze financial documents and extract transaction details.

## Features

- PDF document processing and analysis
- Transaction details extraction using OpenAI's GPT models
- Interactive web interface using Streamlit
- Asynchronous document processing
- Downloadable analysis results
- Summary statistics and data

## Requirements

- Python 3.8+
- OpenAI API key
- Required packages (see requirements.txt)

## Installation


pip install -r requirements.txt


## Usage

1. Run the application:

streamlit run app.py


2. Enter your OpenAI API key in the sidebar
3. Upload PDF financial documents
4. Click "Analyze Documents" to process
5. View and download results

## Key Components

### PDFProcessor (utils.py)
- Handles PDF parsing and text extraction
- Creates text chunks for processing
- Generates vector embeddings
- Creates QA chain for document analysis

### TransactionDetailsExtractor (app.py)
- Manages the web interface
- Handles file uploads
- Processes documents asynchronously
- Displays results and statistics
- Provides data export functionality

## Configuration

Default settings in PDFConfig:
- Chunk size: 4000
- Chunk overlap: 50
- Temperature: 0.5
- Model: gpt-4o-mini-2024-07-18
- Embedding model: text-embedding-3-large

## License

MIT License

