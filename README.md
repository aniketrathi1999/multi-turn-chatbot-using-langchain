# MedBot - Medical Information Assistant

A multi-turn chatbot built with LangChain and Pinecone that provides medical information and answers questions about medicines, their uses, and side effects.

## Features

- **Conversational AI**: Natural, multi-turn conversations about medical information
- **Knowledge Base**: Powered by Pinecone vector store for efficient information retrieval
- **Context-Aware**: Maintains conversation history for coherent, relevant responses
- **Data-Driven**: Uses structured medical data for accurate information

## Prerequisites

- Python 3.8+
- Pinecone account and API key
- OpenAI API key
- Python virtual environment (recommended)

## Installation

1. Clone the repository:
   ```bash
   git clone https://github.com/yourusername/medbot-langchain.git
   cd medbot-langchain
   ```

2. Create and activate a virtual environment:
   ```bash
   python -m venv venv
   .\venv\Scripts\activate  # Windows
   source venv/bin/activate  # Linux/Mac
   ```

3. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```

4. Create a `.env.local` file with your API keys:
   ```env
   PINECONE_API_KEY=your_pinecone_api_key
   OPENAI_API_KEY=your_openai_api_key
   CSV_PATH=./data/medicine_data.csv
   ```

## Usage

### Data Ingestion

To load medical data into the vector store:

```bash
python src/ingest.py
```

### Starting the Chat Interface

```bash
python src/chat.py
```

## Project Structure

```
multi-turn-chatbot-using-langchain/
├── data/                   # Medical data files
│   ├── Medicine_Details.csv
│   └── medicine_data.csv
├── src/
│   ├── chat.py             # Chat interface
│   ├── graph_builder.py    # Conversation graph definition
│   ├── ingest.py           # Data ingestion script
│   └── vector_store.py     # Vector store configuration
├── .env.template          # Template for environment variables
├── .gitignore
├── README.md
└── requirements.txt
```

## Configuration

The application can be configured using the following environment variables:

- `PINECONE_API_KEY`: Your Pinecone API key
- `OPENAI_API_KEY`: Your OpenAI API key
- `CSV_PATH`: Path to the medical data CSV file
- `INDEX_NAME`: Name of the Pinecone index (default: "medicines-index")

## Contributing

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/amazing-feature`)
3. Commit your changes (`git commit -m 'Add some amazing feature'`)
4. Push to the branch (`git push origin feature/amazing-feature`)
5. Open a Pull Request

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## Acknowledgments

- [LangChain](https://python.langchain.com/) for the LLM framework
- [Pinecone](https://www.pinecone.io/) for vector storage
- [OpenAI](https://openai.com/) for language models
