# Document Q&A Chatbot

A Streamlit-based chatbot that allows you to upload text files and ask questions about their content using RAG (Retrieval-Augmented Generation).

## Features

- Upload up to 5 text files
- Interactive chat interface
- Memory of conversation context
- Real-time responses
- Document-based question answering
- Shows top 3 most relevant chunks for each answer
- Uses cosine similarity for better text matching

## Sample Files

The repository includes sample text files for testing:
- `company_overview.txt`: Contains information about the company's history, mission, and values
- `employee_data.txt`: Contains information about employee statistics and policies
- `financial_data.txt`: Contains financial information and metrics
- `product_portfolio.txt`: Contains details about products and services

You can use these files to test the chatbot or replace them with your own text files.

## Setup

1. Clone this repository
2. Install the required dependencies:
   ```bash
   pip install -r requirements.txt
   ```
3. Create a `.env` file in the root directory and add your OpenAI API key:
   ```
   OPENAI_API_KEY=your_api_key_here
   ```

## Usage

1. Run the Streamlit app:
   ```bash
   streamlit run app.py
   ```
2. Open your web browser and navigate to the provided local URL (usually http://localhost:8501)
3. Upload your text files (up to 5)
4. Wait for the files to be processed
5. Start asking questions about the content of your documents

## How it Works

1. The application uses LangChain to process and chunk your documents
2. Documents are embedded using OpenAI's embeddings
3. When you ask a question:
   - The system finds the 3 most relevant chunks using cosine similarity
   - These chunks are used to generate a context-aware response
   - You can view the relevant chunks used for each answer

## Requirements

- Python 3.8+
- OpenAI API key
- Internet connection for API calls

## Note

Make sure you have a valid OpenAI API key and sufficient credits in your account to use the application. 