#!/usr/bin/env python3
import subprocess
import os
import csv
import tempfile
import re
from scipy.stats import spearmanr

# Configuration paths - Update these to your actual paths
SRILM_DIR = "./srilm"  
WORK_DIR = "./"

UNIGRAM_LM = os.path.join(SRILM_DIR, "unigram.lm")
BIGRAM_LM = os.path.join(SRILM_DIR, "bigram.lm")
TRIGRAM_LM = os.path.join(SRILM_DIR, "trigram.lm")
BNC_CSV = os.path.join(WORK_DIR, "bnc.csv")

def batch_score_sentences(model_file, sentences_list):
    """Scores a batch of sentences and strictly validates the output."""
    with tempfile.NamedTemporaryFile(mode='w', suffix='.txt', delete=False, encoding='utf-8') as f:
        for text in sentences_list:
            safe_text = text.strip()
            if not safe_text:
                safe_text = "<s>" 
            f.write(safe_text + "\n")
        temp_file = f.name
    
    try:
        cmd = [os.path.join(SRILM_DIR, "ngram"), "-lm", model_file, "-ppl", temp_file, "-debug", "1"]
        result = subprocess.run(cmd, capture_output=True, text=True, timeout=120, encoding='utf-8')
        
        output = result.stdout + "\n" + result.stderr
        scores = []
        
        # Looking for the word "logprob="
        for line in output.split('\n'):
            if 'logprob=' in line and 'ppl=' in line:
                # Split the line by spaces, find where 'logprob=' is, and grab the number next to it
                parts = line.split()
                for i, part in enumerate(parts):
                    if part.startswith('logprob='):
                        try:
                            # Sometimes it's attached like logprob=-4.5, sometimes there's a space
                            if len(part) > 8:
                                scores.append(float(part.split('=')[1]))
                            else:
                                scores.append(float(parts[i+1]))
                        except (ValueError, IndexError):
                            pass
                            
        # FAILSAFE: If we didn't get a score for nearly every sentence, SRILM is hiding them.
        if len(scores) < len(sentences_list):
            print(f"\n[CRITICAL ERROR] Expected {len(sentences_list)} scores, but only found {len(scores)}!")
            print("This is why the correlations were identical. Here is the raw SRILM output:")
            print("-" * 50)
            print(output[:1500])  # Print the first 1500 characters so we can see the format
            print("-" * 50)
            return None
            
        # Slice off the final overall file summary score
        return scores[:len(sentences_list)]
        
    finally:
        if os.path.exists(temp_file):
            os.remove(temp_file)

def main():
    print("Loading BNC corpus...")
    texts, ratings, lengths = [], [], []
    
    with open(BNC_CSV, 'r', encoding='utf-8') as f:
        reader = csv.DictReader(f, delimiter='\t')
        for row in reader:
            texts.append(row['text'].strip())
            ratings.append(float(row['mean_rating']))
            lengths.append(int(row['length']))

    print("Scoring Unigram...")
    uni_logprobs = batch_score_sentences(UNIGRAM_LM, texts)
    if uni_logprobs is None: return
    
    print("Scoring Bigram...")
    bi_logprobs = batch_score_sentences(BIGRAM_LM, texts)
    if bi_logprobs is None: return
    
    print("Scoring Trigram...")
    tri_logprobs = batch_score_sentences(TRIGRAM_LM, texts)
    if tri_logprobs is None: return

    uni_avg, bi_avg, tri_avg = [], [], []
    bi_slor, tri_slor = [], []

    for i in range(len(texts)):
        n = lengths[i] if lengths[i] > 0 else 1
        uni_avg.append(uni_logprobs[i] / n)
        bi_avg.append(bi_logprobs[i] / n)
        tri_avg.append(tri_logprobs[i] / n)
        bi_slor.append(bi_logprobs[i] - uni_logprobs[i])
        tri_slor.append(tri_logprobs[i] - uni_logprobs[i])

    # ==========================================
    # PART 1 OUTPUT: Show the actual computed scores
    # ==========================================
    print("\n" + "="*80)
    print("PART 1: COMPUTED MODEL SCORES (Sample of first 5 sentences)")
    print("="*80)
    print(f"{'Sent #':<8} | {'Model':<10} | {'Total Log-Prob':<15} | {'Avg Log-Prob':<15} | {'SLOR':<10}")
    print("-" * 80)
    
    for i in range(5):
        print(f"Sentence {i+1}: '{texts[i][:40]}...' (Length: {lengths[i]})")
        print(f"{'':<8} | {'Unigram':<10} | {uni_logprobs[i]:<15.4f} | {uni_avg[i]:<15.4f} | {'N/A':<10}")
        print(f"{'':<8} | {'Bigram':<10} | {bi_logprobs[i]:<15.4f} | {bi_avg[i]:<15.4f} | {bi_slor[i]:<10.4f}")
        print(f"{'':<8} | {'Trigram':<10} | {tri_logprobs[i]:<15.4f} | {tri_avg[i]:<15.4f} | {tri_slor[i]:<10.4f}")
        print("-" * 80)

    # ==========================================
    # PART 2 OUTPUT: Real Spearman Correlations
    # ==========================================
    print("\n" + "="*50)
    print("PART 2: SPEARMAN CORRELATION RESULTS")
    print("="*50)
    
    metrics = {
        "Unigram (Total Log-Prob)": uni_logprobs,
        "Bigram (Total Log-Prob)": bi_logprobs,
        "Trigram (Total Log-Prob)": tri_logprobs,
        "Unigram (Avg Log-Prob)": uni_avg,
        "Bigram (Avg Log-Prob)": bi_avg,
        "Trigram (Avg Log-Prob)": tri_avg,
        "Bigram (SLOR)": bi_slor,
        "Trigram (SLOR)": tri_slor
    }

    for name, data in metrics.items():
        rho, p_val = spearmanr(data, ratings)
        print(f"{name: <25} | ρ: {rho: >7.4f} | p-value: {p_val:.4e}")

if __name__ == "__main__":
    main()