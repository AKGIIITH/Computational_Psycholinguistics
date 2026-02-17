import pandas as pd 
import matplotlib.pyplot as plt 
import seaborn as sns 
from scipy.stats import pearsonr 
import numpy as np

# Set style for better-looking plots
sns.set_style("whitegrid")
plt.rcParams['figure.figsize'] = (10, 6)

# Load the data
print("Loading data...")
rt_data = pd.read_csv('naturalstories/naturalstories_RTS/processed_RTs.tsv', sep='\t')
freq_1gram = pd.read_csv('naturalstories/freqs/freqs-1.tsv', sep='\t', header=None,
                         names=['token_code', 'ngram_order', 'word', 'freq', 'context_freq'])

print(f"Loaded {len(rt_data)} RT records and {len(freq_1gram)} frequency records\n")

# Task 1: Compute mean RT per word
print("TASK 1: Computing mean RT per word")

mean_rt_per_word = rt_data.groupby(['item', 'zone', 'word'])['RT'].mean().reset_index()
mean_rt_per_word.columns = ['item', 'zone', 'word', 'mean_RT']

print(f"Computed mean RT for {len(mean_rt_per_word)} unique word instances")
print("\nSample of mean RT per word:")
print(mean_rt_per_word.head(10))
print()

# Task 2: Add word length (in characters)
print("TASK 2: Adding word length")

mean_rt_per_word['word_length'] = mean_rt_per_word['word'].str.len()
print(f"Word length range: {mean_rt_per_word['word_length'].min()} to {mean_rt_per_word['word_length'].max()} characters")
print()

# Task 3: Merge with frequency data
print("TASK 3: Merging with frequency data")

# Extract word from token_code for matching
freq_1gram['item'] = freq_1gram['token_code'].str.split('.').str[0].astype(int)
freq_1gram['zone'] = freq_1gram['token_code'].str.split('.').str[1].astype(int)

# Merge RT data with frequency data
merged_data = mean_rt_per_word.merge(
    freq_1gram[['item', 'zone', 'freq']], 
    on=['item', 'zone'], 
    how='left'
)

# Handle words with no frequency data (assign a small frequency)
merged_data['freq'] = pd.to_numeric(merged_data['freq'], errors='coerce')
merged_data = merged_data.dropna(subset=['freq'])

# Convert frequency to log scale (common in psycholinguistics)
merged_data['log_freq'] = np.log10(merged_data['freq'] + 1)

print(f"Successfully merged {len(merged_data)} records with frequency data")
print(f"Frequency range: {merged_data['freq'].min():.0f} to {merged_data['freq'].max():.0f}")
print(f"Log frequency range: {merged_data['log_freq'].min():.2f} to {merged_data['log_freq'].max():.2f}")
print()

# Task 4: Plot word length vs mean RT
print("TASK 4: Plotting word length vs mean RT")

plt.figure(figsize=(10, 6))
plt.scatter(merged_data['word_length'], merged_data['mean_RT'], alpha=0.5, s=30)
plt.xlabel('Word Length (characters)', fontsize=12)
plt.ylabel('Mean RT (ms)', fontsize=12)
plt.title('Word Length vs Mean Reading Time', fontsize=14, fontweight='bold')

# Add trend line
z = np.polyfit(merged_data['word_length'], merged_data['mean_RT'], 1)
p = np.poly1d(z)
plt.plot(merged_data['word_length'].unique(), 
         p(merged_data['word_length'].unique()), 
         "r--", alpha=0.8, linewidth=2, label='Trend line')
plt.legend()
plt.tight_layout()
plt.savefig('word_length_vs_rt.png', dpi=300)
print("Plot saved as 'word_length_vs_rt.png'")
plt.close()

# Task 5: Plot word frequency vs mean RT
print("TASK 5: Plotting word frequency vs mean RT")

plt.figure(figsize=(10, 6))
plt.scatter(merged_data['log_freq'], merged_data['mean_RT'], alpha=0.5, s=30)
plt.xlabel('Log Word Frequency (log10)', fontsize=12)
plt.ylabel('Mean RT (ms)', fontsize=12)
plt.title('Word Frequency vs Mean Reading Time', fontsize=14, fontweight='bold')

# Add trend line
z = np.polyfit(merged_data['log_freq'], merged_data['mean_RT'], 1)
p = np.poly1d(z)
plt.plot(sorted(merged_data['log_freq'].unique()), 
         p(sorted(merged_data['log_freq'].unique())), 
         "r--", alpha=0.8, linewidth=2, label='Trend line')
plt.legend()
plt.tight_layout()
plt.savefig('word_frequency_vs_rt.png', dpi=300)
print("Plot saved as 'word_frequency_vs_rt.png'")
plt.close()

# Task 6: Compute Pearson correlations
print("TASK 6: Computing Pearson Correlations")

# Correlation 1: Length vs Frequency
corr_len_freq, p_len_freq = pearsonr(merged_data['word_length'], merged_data['log_freq'])
print(f"\n1. Word Length vs Log Frequency:")
print(f"   Pearson's r = {corr_len_freq:.4f}")
print(f"   p-value = {p_len_freq:.4e}")
print(f"   Interpretation: {'Significant' if p_len_freq < 0.05 else 'Not significant'} correlation")

# Correlation 2: Length vs Mean RT
corr_len_rt, p_len_rt = pearsonr(merged_data['word_length'], merged_data['mean_RT'])
print(f"\n2. Word Length vs Mean RT:")
print(f"   Pearson's r = {corr_len_rt:.4f}")
print(f"   p-value = {p_len_rt:.4e}")
print(f"   Interpretation: {'Significant' if p_len_rt < 0.05 else 'Not significant'} correlation")

# Correlation 3: Frequency vs Mean RT
corr_freq_rt, p_freq_rt = pearsonr(merged_data['log_freq'], merged_data['mean_RT'])
print(f"\n3. Log Frequency vs Mean RT:")
print(f"   Pearson's r = {corr_freq_rt:.4f}")
print(f"   p-value = {p_freq_rt:.4e}")
print(f"   Interpretation: {'Significant' if p_freq_rt < 0.05 else 'Not significant'} correlation")

# Task 7: Create a comprehensive visualization
print("Creating comprehensive visualization...")

fig, axes = plt.subplots(2, 2, figsize=(14, 12))

# Plot 1: Length vs RT
axes[0, 0].scatter(merged_data['word_length'], merged_data['mean_RT'], alpha=0.5, s=30)
axes[0, 0].set_xlabel('Word Length (characters)', fontsize=11)
axes[0, 0].set_ylabel('Mean RT (ms)', fontsize=11)
axes[0, 0].set_title(f'Length vs RT (r={corr_len_rt:.3f}, p={p_len_rt:.2e})', 
                     fontsize=12, fontweight='bold')
z = np.polyfit(merged_data['word_length'], merged_data['mean_RT'], 1)
p = np.poly1d(z)
axes[0, 0].plot(merged_data['word_length'].unique(), 
                p(merged_data['word_length'].unique()), "r--", alpha=0.8)

# Plot 2: Frequency vs RT
axes[0, 1].scatter(merged_data['log_freq'], merged_data['mean_RT'], alpha=0.5, s=30)
axes[0, 1].set_xlabel('Log Frequency', fontsize=11)
axes[0, 1].set_ylabel('Mean RT (ms)', fontsize=11)
axes[0, 1].set_title(f'Frequency vs RT (r={corr_freq_rt:.3f}, p={p_freq_rt:.2e})', 
                     fontsize=12, fontweight='bold')
z = np.polyfit(merged_data['log_freq'], merged_data['mean_RT'], 1)
p = np.poly1d(z)
axes[0, 1].plot(sorted(merged_data['log_freq'].unique()), 
                p(sorted(merged_data['log_freq'].unique())), "r--", alpha=0.8)

# Plot 3: Length vs Frequency
axes[1, 0].scatter(merged_data['word_length'], merged_data['log_freq'], alpha=0.5, s=30)
axes[1, 0].set_xlabel('Word Length (characters)', fontsize=11)
axes[1, 0].set_ylabel('Log Frequency', fontsize=11)
axes[1, 0].set_title(f'Length vs Frequency (r={corr_len_freq:.3f}, p={p_len_freq:.2e})', 
                     fontsize=12, fontweight='bold')
z = np.polyfit(merged_data['word_length'], merged_data['log_freq'], 1)
p = np.poly1d(z)
axes[1, 0].plot(merged_data['word_length'].unique(), 
                p(merged_data['word_length'].unique()), "r--", alpha=0.8)

# Plot 4: Correlation heatmap
corr_matrix = merged_data[['word_length', 'log_freq', 'mean_RT']].corr()
sns.heatmap(corr_matrix, annot=True, fmt='.3f', cmap='coolwarm', center=0,
            square=True, ax=axes[1, 1], cbar_kws={"shrink": 0.8})
axes[1, 1].set_title('Correlation Matrix', fontsize=12, fontweight='bold')

plt.tight_layout()
plt.savefig('comprehensive_analysis.png', dpi=300)
print("Comprehensive plot saved as 'comprehensive_analysis.png'")
plt.close()

# Summary statistics
print("SUMMARY STATISTICS")

print("\nDescriptive Statistics:")
print(merged_data[['word_length', 'freq', 'log_freq', 'mean_RT']].describe())