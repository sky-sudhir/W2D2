# Fill-in-the-Blank and Tokenization Analysis

This project provides a Python script to analyze how different tokenization algorithms (BPE, WordPiece, SentencePiece) process sentences and uses a masked language model (`bert-base-uncased`) to predict missing words.

## Features

-   **Multiple Tokenizers**: Tokenizes input sentences using BPE, WordPiece, and SentencePiece (Unigram) to compare their outputs.
-   **Masked Word Prediction**: Masks two tokens in a sentence and uses the `fill-mask` pipeline with `bert-base-uncased` to predict the top 3 most likely words for each blank.
-   **Interactive & Persistent**: Runs in a continuous loop, allowing you to process multiple sentences in one session.
-   **JSON Output**: Saves all results, including the original sentence and the analysis, to `predections.json`. New results are appended, preserving your history.

## Setup

1.  **Clone the Repository**
    ```bash
    git clone <repository-url>
    cd <repository-directory>
    ```

2.  **Create a Virtual Environment (Recommended)**
    ```bash
    python -m venv .venv
    source .venv/bin/activate  # On Windows, use `.venv\Scripts\activate`
    ```

3.  **Install Dependencies**
    A `requirements.txt` file is included for easy installation.
    ```bash
    pip install -r requirements.txt
    ```

## Usage

1.  **Run the Script**
    ```bash
    python tokenise.py
    ```

2.  **Enter a Sentence**
    When prompted, type or paste a sentence and press Enter.

3.  **Exit the Script**
    To finish your session, simply press Enter on an empty line.

4.  **View Results**
    The analysis for each sentence is printed to the console and saved in the `predections.json` file for later review. 