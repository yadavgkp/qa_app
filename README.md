# Enhanced PDF Question Answering Application

This application allows users to upload PDF documents from any domain and ask questions about their content. It leverages AI to analyze, summarize, and extract information from documents.

## Features

- **Multi-domain PDF support**: Works with research papers, financial reports, legal documents, technical manuals, and more
- **Document analysis**: Provides statistics, summaries, and visualizations of document content
- **Interactive Q&A**: Ask questions about document content and get accurate answers
- **Source citations**: Answers include references to the specific parts of the document used
- **Chat history**: Maintains conversation context for follow-up questions
- **Document preview**: View thumbnail previews of document pages
- **Customizable settings**: Adjust processing parameters to optimize for different document types

## Installation

1. Clone this repository:
```bash
git clone https://github.com/yourusername/enhanced-pdf-qa.git
cd enhanced-pdf-qa
```

2. Create a virtual environment and activate it:
```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

3. Install the required dependencies:
```bash
pip install -r requirements.txt
```

4. Create a `.streamlit` directory and copy the config file:
```bash
mkdir -p .streamlit
cp config.toml .streamlit/
```

5. Add your OpenAI API key to the `.env` file or set it directly in the application.

## Usage

1. Run the Streamlit application:
```bash
streamlit run app.py
```

2. Open your web browser and navigate to the URL displayed in the terminal (typically http://localhost:8501)

3. Upload your PDF document(s) using the sidebar

4. Click "Process Documents" to analyze the PDF

5. Explore the document analysis and ask questions about the content

## Customization

You can customize the application behavior through the "Advanced Settings" section in the sidebar:

- **Chunk Size**: Controls how documents are split (larger chunks retain more context)
- **Chunk Overlap**: Amount of overlap between chunks to maintain continuity
- **LLM Model**: Choose between GPT-3.5 Turbo (faster) or GPT-4 (more accurate)
- **Temperature**: Control randomness in responses (lower = more deterministic)
- **Embeddings**: Choose between HuggingFace (free) or OpenAI (higher quality) embeddings

## Project Structure

```
.
├── app.py                 # Main application code
├── requirements.txt       # Python dependencies
├── .env                   # Environment variables
└── .streamlit/
    └── config.toml        # Streamlit configuration
```

## Requirements

- Python 3.8+
- OpenAI API key
- Internet connection for API access

## License

MIT

## Acknowledgements

This application uses the following technologies:
- Streamlit for the web interface
- LangChain for document processing and QA chain
- OpenAI models for text processing
- PyMuPDF for PDF processing
- FAISS for vector storage and retrieval
