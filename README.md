# Document Q&A Chatbot

A Streamlit-based chatbot that allows you to upload text files and ask questions about their content using RAG (Retrieval-Augmented Generation).

## Features

- Upload up to 5 text files
- Interactive chat interface
- Memory of conversation context
- Real-time responses
- Document-based question answering

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

## Requirements

- Python 3.8+
- OpenAI API key
- Internet connection for API calls

## Note

Make sure you have a valid OpenAI API key and sufficient credits in your account to use the application. 