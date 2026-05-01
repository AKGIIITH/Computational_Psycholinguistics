#!/usr/bin/env python3
import csv
import re
import numpy as np
from collections import Counter
import matplotlib.pyplot as plt
from scipy.stats import linregress, pearsonr

def main():
    print("Part 3: Word Length & Information Content Analysis\n")
    
    # ---------------------------------------------------------
    # DATA LOADING & CLEANING
    # ---------------------------------------------------------
    print("Loading and cleaning datasets...")
    raw_words = []
    
    # Load BNC
    try:
        with open('bnc.csv', 'r', encoding='utf-8') as f:
            reader = csv.DictReader(f, delimiter='\t')
            for row in reader:
                raw_words.extend(re.findall(r'\b[a-z]+\b', row['text'].lower()))
    except FileNotFoundError:
        pass

    # Load Brown
    for file in ['srilm/browndata/brown-train.txt', 'srilm/browndata/brown-test.txt']:
        try:
            with open(file, 'r', encoding='utf-8') as f:
                raw_words.extend(re.findall(r'\b[a-z]+\b', f.read().lower()))
        except FileNotFoundError:
            try:
                with open(file.split('/')[-1], 'r', encoding='utf-8') as f:
                    raw_words.extend(re.findall(r'\b[a-z]+\b', f.read().lower()))
            except FileNotFoundError:
                pass

    # Clean Data: Remove 1-letter noise (keep only legitimate words)
    valid_single_letters = {'a', 'i', 'o'}
    clean_words = [w for w in raw_words if len(w) > 1 or w in valid_single_letters]

    total_tokens = len(clean_words)
    word_freq = Counter(clean_words)
    unique_words = list(word_freq.keys())
    
    print(f"Total words (tokens) after cleaning: {total_tokens:,}")
    print(f"Unique words (types): {len(unique_words):,}\n")

    # ---------------------------------------------------------
    # CALCULATIONS
    # ---------------------------------------------------------
    lengths = np.array([len(w) for w in unique_words])
    frequencies = np.array([word_freq[w] for w in unique_words])
    probabilities = frequencies / total_tokens
    
    # Information Content = -log2(Probability)
    information_content = -np.log2(probabilities)
    
    log10_lengths = np.log10(lengths)
    log10_frequencies = np.log10(frequencies)

    # ---------------------------------------------------------
    # TASK 1 & 2: Length Data and Histogram
    # ---------------------------------------------------------
    print("Task 1 & 2: Word Length Distribution")
    length_counts = Counter(lengths)
    x_lengths = sorted(length_counts.keys())
    
    # Print the requested numeric data for the report
    print("Number of unique words at each length (Top 10):")
    for length in x_lengths[:10]:
        print(f"  Length {length} letters: {length_counts[length]:,} words")
    print("  ...")
    
    # Plot
    plt.figure(figsize=(10, 6))
    y_counts = [length_counts[l] for l in x_lengths]
    plt.bar(x_lengths, y_counts, color='skyblue', edgecolor='black')
    plt.title('Histogram of Word Lengths (Unique Words)')
    plt.xlabel('Word Length (Number of Letters)')
    plt.ylabel('Number of Unique Words')
    plt.grid(axis='y', alpha=0.3)
    plt.savefig('task2_length_histogram.png')
    plt.close()

    # ---------------------------------------------------------
    # HYPOTHESIS TESTING
    # ---------------------------------------------------------
    print("\nTask 2.1: Hypothesis Testing (Information Content vs Frequency)")
    
    # Model A: Raw Frequency
    slope_f, int_f, r_f, p_f, std_f = linregress(frequencies, lengths)
    r2_freq = r_f**2
    
    # Model B: Information Content
    slope_ic, int_ic, r_ic, p_ic, std_ic = linregress(information_content, lengths)
    r2_ic = r_ic**2
    
    print(f"Model A (Predictor: Raw Frequency): R^2 = {r2_freq:.4f}")
    print(f"Model B (Predictor: Information Content): R^2 = {r2_ic:.4f}")

    # ---------------------------------------------------------
    # SHORTEST WORDS
    # ---------------------------------------------------------
    print("\nTask 2.2: Shortest Words in Dataset")
    min_length = min(lengths)
    shortest_words = [(w, word_freq[w]) for w, l in zip(unique_words, lengths) if l == min_length]
    
    # Sort by frequency descending
    shortest_words.sort(key=lambda x: x[1], reverse=True)
    
    print(f"The shortest words are {min_length} letter(s) long:")
    for word, freq in shortest_words:
        print(f"  '{word}' (Frequency: {freq:,})")
        
    """Explanation: These are highly frequent function words. Because they appear constantly,
    they carry very little unpredictable information (low Information Content). Language naturally
    optimizes them to be as short as possible to maximize communicative efficiency."""

    # ---------------------------------------------------------
    # PEARSON CORRELATION
    # ---------------------------------------------------------
    print("\nTask 2.3: Pearson Correlation")
    pearson_r, pearson_p = pearsonr(lengths, frequencies)
    print(f"Pearson's coefficient of correlation (Length vs Frequency): {pearson_r:.4f}")
    """This negative correlation mathematically confirms Zipf's law of abbreviation: shorter words tend to be used more frequently."""

    # ---------------------------------------------------------
    # SCATTER PLOTS
    # ---------------------------------------------------------
    print("\nGenerating scatter plots (Tasks 2.4, 2.5, 2.6)...")
    
    # 4. Plot: Length vs Frequency
    plt.figure(figsize=(10, 6))
    plt.scatter(lengths, frequencies, alpha=0.3, color='purple')
    plt.title('Length vs. Frequency (Zoomed to bottom 2,000)')
    plt.xlabel('Length (Letters)')
    plt.ylabel('Frequency')
    plt.ylim(0, 2000) 
    
    plt.savefig('task4_length_vs_freq.png')
    plt.close()

    # 5. Plot: log10(Length) vs log10(Frequency)
    plt.figure(figsize=(10, 6))
    plt.scatter(log10_lengths, log10_frequencies, alpha=0.3, color='blue')
    plt.title('log10(Length) vs. log10(Frequency)')
    plt.xlabel('log10(Word Length)')
    plt.ylabel('log10(Frequency)')
    plt.savefig('task5_log10Length_vs_log10Freq.png')
    plt.close()

    # 6. Plot: log10(Length) vs Information Content
    plt.figure(figsize=(10, 6))
    plt.scatter(log10_lengths, information_content, alpha=0.3, color='green')
    plt.title('log10(Length) vs. Information Content')
    plt.xlabel('log10(Word Length)')
    plt.ylabel('Information Content (-log2(P))')
    plt.savefig('task6_log10Length_vs_IC.png')
    plt.close()

if __name__ == "__main__":
    main()