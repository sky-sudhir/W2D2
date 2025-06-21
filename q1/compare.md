# Comparison of Tokenization Algorithms

Tokenization is the process of breaking down a text into smaller units called tokens. These tokens can be words, subwords, or characters. The choice of algorithm significantly impacts how a model "sees" a sentence. This project uses three popular algorithms: BPE, WordPiece, and SentencePiece.

### BPE (Byte-Pair Encoding)

-   **How it Works**: BPE starts with a vocabulary of individual characters and iteratively merges the most frequently occurring pair of adjacent tokens. This process continues until the vocabulary reaches a desired size.
-   **Strengths**: It's effective at handling rare words by breaking them down into known subwords, thus avoiding "unknown" tokens.
-   **Observed Behavior**: In this project, BPE often creates a mix of full words (for common terms like `cat`) and subword units (like `be` and `cause` for `because`). The resulting tokens are highly dependent on the training data.

### WordPiece

-   **How it Works**: Similar to BPE, but it prioritizes preserving whole words. A word is only split into subword units if it's not already in the vocabulary. WordPiece also adds a special prefix (typically `##`) to denote subwords that are not at the beginning of a word.
-   **Strengths**: This is the algorithm used by BERT. It strikes a good balance between vocabulary size and the number of out-of-vocabulary words, often creating more intuitive splits than BPE.
-   **Observed Behavior**: You will often see tokens like `sat` preserved as a whole word, while a less common word might be split into `play` and `##ing`. This is very effective for the `fill-mask` task.

### SentencePiece (Unigram)

-   **How it Works**: SentencePiece operates on a different principle. It starts with a large number of potential tokens and progressively removes the ones that are least likely to occur, based on a unigram language model, until the vocabulary is pruned to the desired size. It also treats whitespace as a character to be tokenized.
-   **Strengths**: It is fully reversible (including whitespace) and language-agnostic, making it very powerful for multilingual models.
-   **Observed Behavior**: This algorithm typically produces the longest sequence of tokens because it tokenizes everything, including spaces (represented as ` `). While providing a very granular view, it can be less efficient for models that have a maximum sequence length.

### Summary

| Algorithm     | Key Feature                                         | Main Advantage                                  |
| :------------ | :-------------------------------------------------- | :---------------------------------------------- |
| **BPE**       | Merges frequent pairs of tokens.                    | Good at handling rare words with subword units. |
| **WordPiece** | Prioritizes whole words; uses `##` for subwords.    | Creates intuitive splits, used by BERT.         |
| **SentencePiece** | Language model-based pruning; tokenizes spaces. | Language-agnostic and perfectly reversible.     |

</rewritten_file> 