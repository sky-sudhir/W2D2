import os
import json
import sys
from transformers import (
    AutoTokenizer, AutoModelForMaskedLM, pipeline
)
from tokenizers import Tokenizer, models, trainers, pre_tokenizers
import sentencepiece as spm

# Define the output file
PRED_FILE = "predictions.json"

def confirm_download(msg):
    """Helper to confirm large downloads."""
    resp = input(f"{msg} [y/N]: ").strip().lower()
    return resp == 'y'

def main():
    """Main function to run the tokenization and prediction loop."""
    # --- Load existing results ---
    if os.path.exists(PRED_FILE):
        with open(PRED_FILE, 'r', encoding='utf-8') as f:
            try:
                all_results = json.load(f)
                if not isinstance(all_results, list):
                    print(f"Warning: '{PRED_FILE}' does not contain a list. Starting fresh.")
                    all_results = []
            except json.JSONDecodeError:
                all_results = []
    else:
        all_results = []

    # --- Setup model and pipeline once ---
    model_id = "bert-base-uncased"
    try:
        tokenizer = AutoTokenizer.from_pretrained(model_id)
        model = AutoModelForMaskedLM.from_pretrained(model_id)
    except Exception as e:
        print(f"Model/tokenizer not found locally: {e}")
        if confirm_download(f"Download model/tokenizer '{model_id}'? (~440MB)"):
            tokenizer = AutoTokenizer.from_pretrained(model_id)
            model = AutoModelForMaskedLM.from_pretrained(model_id)
        else:
            print("Aborted model download. Exiting.")
            sys.exit(1)

    fill_mask = pipeline("fill-mask", model=model, tokenizer=tokenizer, device=-1)

    # --- Input loop for sentences ---
    while True:
        sentence = input("\nEnter a sentence to process (or press Enter to finish): ").strip()
        if not sentence:
            break

        print(f"\n--- Processing sentence: '{sentence}' ---")
        current_result = {'original_sentence': sentence}

        # --- 1. Tokenization ---
        tokenization_data = {}
        
        # BPE
        bpe_tokenizer = Tokenizer(models.BPE())
        bpe_tokenizer.pre_tokenizer = pre_tokenizers.Whitespace()
        bpe_trainer = trainers.BpeTrainer(vocab_size=20, show_progress=False)
        bpe_tokenizer.train_from_iterator([sentence], trainer=bpe_trainer)
        bpe_tokens = bpe_tokenizer.encode(sentence)
        tokenization_data['BPE'] = {'tokens': bpe_tokens.tokens, 'ids': bpe_tokens.ids, 'count': len(bpe_tokens.tokens)}

        # WordPiece
        wp_tokenizer = Tokenizer(models.WordPiece())
        wp_tokenizer.pre_tokenizer = pre_tokenizers.Whitespace()
        wp_trainer = trainers.WordPieceTrainer(vocab_size=20, show_progress=False)
        wp_tokenizer.train_from_iterator([sentence], trainer=wp_trainer)
        wp_tokens = wp_tokenizer.encode(sentence)
        tokenization_data['WordPiece'] = {'tokens': wp_tokens.tokens, 'ids': wp_tokens.ids, 'count': len(wp_tokens.tokens)}

        # SentencePiece
        unique_chars = set(sentence)
        spm_vocab_size = min(max(len(unique_chars) + 3, 10), 30)
        spm_model_prefix = 'spm_unigram'
        spm_model_file = f"{spm_model_prefix}.model"
        spm.SentencePieceTrainer.Train(
            sentence_iterator=iter([sentence]), model_prefix=spm_model_prefix, vocab_size=spm_vocab_size, model_type='unigram', train_extremely_large_corpus=False
        )
        sp = spm.SentencePieceProcessor()
        sp.load(spm_model_file)
        sp_tokens = sp.encode(sentence, out_type=str)
        sp_ids = sp.encode(sentence, out_type=int)
        tokenization_data['SentencePiece'] = {'tokens': sp_tokens, 'ids': sp_ids, 'count': len(sp_tokens)}
        if os.path.exists(spm_model_file): os.remove(spm_model_file)
        if os.path.exists(f"{spm_model_prefix}.vocab"): os.remove(f"{spm_model_prefix}.vocab")

        tokenization_data['tokenization_note'] = "BPE merges frequent pairs, WordPiece prefers whole words, and SentencePiece optimizes likelihood."
        current_result['tokenization'] = tokenization_data
        
        # Print tokenization results
        print("\n--- Tokenization Results ---")
        for algo, data in tokenization_data.items():
            if algo.endswith('note'): continue
            print(f"\n{algo}: Tokens: {data['tokens']}, Count: {data['count']}")
        print("\nNote:", tokenization_data['tokenization_note'])

        # --- 2. Mask & Predict ---
        tok_sentence = tokenizer.tokenize(sentence)
        if len(tok_sentence) < 3: # Need at least 3 for 2 masks + 1 token
            masking_data = {"status": "Skipped, sentence too short to mask two tokens."}
            print(f"\n{masking_data['status']}")
        else:
            mask_token = tokenizer.mask_token
            masked_sentence_list = tok_sentence.copy()
            mask_indices = [1, min(6, len(tok_sentence) - 2)]

            predictions = []
            for i, idx in enumerate(mask_indices):
                temp_sentence_list = tok_sentence.copy()
                temp_sentence_list[idx] = mask_token
                temp_text = tokenizer.convert_tokens_to_string(temp_sentence_list)

                print(f"\nPrediction for blank {i+1} (mask at position {idx}):")
                preds = fill_mask(temp_text, top_k=3)
                
                for rank, pred in enumerate(preds, 1):
                    print(f"{rank}. {pred['sequence']}")
                predictions.append({
                    'blank': i+1, 'mask_index': idx, 'top_3': [p['token_str'] for p in preds]
                })
            
            masking_data = {
                'original_tokens': tok_sentence,
                'mask_indices': mask_indices,
                'predictions': predictions,
                'comment': "Predictions are plausible if they fit the context and grammar."
            }
        current_result['masking'] = masking_data
        
        all_results.append(current_result)
        
        with open(PRED_FILE, 'w', encoding='utf-8') as f:
            json.dump(all_results, f, indent=2)
        print(f"\nResults for '{sentence}' appended to {PRED_FILE}")

    print("\nNo sentence entered. Exiting.")

if __name__ == "__main__":
    main()