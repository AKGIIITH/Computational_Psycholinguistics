#!/usr/bin/env python3
import subprocess
import os
import csv
import tempfile
import re
import math
import numpy as np
from collections import Counter
import matplotlib.pyplot as plt
from scipy.stats import spearmanr, linregress, pearsonr

# Neural Model Imports
import torch
from transformers import GPT2LMHeadModel, GPT2Tokenizer

# Configuration paths
SRILM_DIR = "./srilm"  
WORK_DIR = "./"
UNIGRAM_LM = os.path.join(SRILM_DIR, "unigram.lm")
BNC_CSV = os.path.join(WORK_DIR, "bnc.csv")

def batch_score_sentences_srilm(model_file, sentences_list):
    """Scores a batch of sentences using SRILM (Used for the Unigram baseline for SLOR)."""
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
        
        for line in output.split('\n'):
            if 'logprob=' in line and 'ppl=' in line:
                parts = line.split()
                for i, part in enumerate(parts):
                    if part.startswith('logprob='):
                        try:
                            if len(part) > 8:
                                scores.append(float(part.split('=')[1]))
                            else:
                                scores.append(float(parts[i+1]))
                        except (ValueError, IndexError):
                            pass
                            
        if len(scores) < len(sentences_list):
            print(f"\n[CRITICAL ERROR] SRILM failed to extract unigram scores.")
            return None
            
        return scores[:len(sentences_list)]
        
    finally:
        if os.path.exists(temp_file):
            os.remove(temp_file)

def batch_score_sentences_gpt2(sentences_list):
    """Scores sentences using the pre-trained GPT-2 Neural Language Model."""
    print("  Loading GPT-2 model & tokenizer... (this may take a moment)")
    tokenizer = GPT2Tokenizer.from_pretrained('gpt2')
    model = GPT2LMHeadModel.from_pretrained('gpt2')
    model.eval() # Set to evaluation mode

    scores = []
    print("  Scoring sentences with GPT-2...")
    for i, text in enumerate(sentences_list):
        if i % 1000 == 0 and i > 0:
            print(f"    Processed {i}/{len(sentences_list)} sentences...")
            
        # Tokenize the sentence
        inputs = tokenizer(text, return_tensors='pt')
        
        # Handle extremely short/empty sentences safely
        if inputs['input_ids'].shape[1] < 2:
            scores.append(0.0)
            continue
            
        with torch.no_grad():
            outputs = model(**inputs, labels=inputs["input_ids"])
            loss = outputs.loss
            
            # GPT-2 loss is CrossEntropy (Natural Log base 'e').
            # We multiply by sequence length to get total log prob.
            # CRITICAL: We divide by np.log(10) to convert from base 'e' to base '10' 
            # so it matches SRILM's mathematical scale for calculating SLOR!
            total_log_prob_e = -loss.item() * inputs["input_ids"].shape[1]
            total_log_prob_10 = total_log_prob_e / np.log(10)
            
            scores.append(total_log_prob_10)
            
    return scores

def run_parts_1_and_2():
    print("Loading BNC corpus...")
    texts, ratings, lengths = [], [], []
    
    with open(BNC_CSV, 'r', encoding='utf-8') as f:
        reader = csv.DictReader(f, delimiter='\t')
        for row in reader:
            texts.append(row['text'].strip())
            ratings.append(float(row['mean_rating']))
            lengths.append(int(row['length']))

    print("\nScoring Baseline Unigram (SRILM)...")
    uni_logprobs = batch_score_sentences_srilm(UNIGRAM_LM, texts)
    
    print("\nScoring Neural Model (GPT-2)...")
    gpt2_logprobs = batch_score_sentences_gpt2(texts)

    uni_avg, gpt2_avg = [], []
    gpt2_slor = []

    for i in range(len(texts)):
        n = lengths[i] if lengths[i] > 0 else 1
        
        uni_avg.append(uni_logprobs[i] / n)
        gpt2_avg.append(gpt2_logprobs[i] / n)
        gpt2_slor.append(gpt2_logprobs[i] - uni_logprobs[i])

    # ==========================================
    # PART 1 OUTPUT
    # ==========================================
    print("\n" + "="*80)
    print("PART 1: COMPUTED NEURAL MODEL SCORES (Sample of first 5 sentences)")
    print("="*80)
    print(f"{'Sent #':<8} | {'Model':<10} | {'Total Log-Prob':<15} | {'Avg Log-Prob':<15} | {'SLOR':<10}")
    print("-" * 80)
    
    for i in range(5):
        print(f"Sentence {i+1}: '{texts[i][:40]}...' (Length: {lengths[i]})")
        print(f"{'':<8} | {'Unigram':<10} | {uni_logprobs[i]:<15.4f} | {uni_avg[i]:<15.4f} | {'N/A':<10}")
        print(f"{'':<8} | {'GPT-2':<10} | {gpt2_logprobs[i]:<15.4f} | {gpt2_avg[i]:<15.4f} | {gpt2_slor[i]:<10.4f}")
        print("-" * 80)

    # ==========================================
    # PART 2 OUTPUT
    # ==========================================
    print("\n" + "="*50)
    print("PART 2: SPEARMAN CORRELATION RESULTS")
    print("="*50)
    
    metrics = {
        "Unigram (Total Log-Prob)": uni_logprobs,
        "GPT-2 (Total Log-Prob)": gpt2_logprobs,
        "Unigram (Avg Log-Prob)": uni_avg,
        "GPT-2 (Avg Log-Prob)": gpt2_avg,
        "GPT-2 (SLOR)": gpt2_slor
    }

    for name, data in metrics.items():
        rho, p_val = spearmanr(data, ratings)
        print(f"{name: <25} | ρ: {rho: >7.4f} | p-value: {p_val:.4e}")

def main():
    run_parts_1_and_2()

if __name__ == "__main__":
    main()