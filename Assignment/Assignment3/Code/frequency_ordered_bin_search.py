import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.stats import pearsonr, ttest_ind
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error
import statsmodels.api as sm
from statsmodels.formula.api import ols
import nltk
from nltk.stem import WordNetLemmatizer
from nltk.corpus import wordnet
from collections import Counter, defaultdict

# Download required NLTK data
try:
    nltk.data.find('corpora/wordnet')
except LookupError:
    nltk.download('wordnet')
    nltk.download('averaged_perceptron_tagger')
    nltk.download('omw-1.4')

sns.set_style("whitegrid")
plt.rcParams['figure.figsize'] = (14, 10)

print("=" * 80)
print("FREQUENCY ORDERED BIN SEARCH (FOBS) MODEL")
print("=" * 80)

# Load data
print("\nLoading data...")
rt_data = pd.read_csv('naturalstories/naturalstories_RTS/processed_RTs.tsv', sep='\t')
freq_1gram = pd.read_csv('naturalstories/freqs/freqs-1.tsv', sep='\t', header=None,
                         names=['token_code', 'ngram_order', 'word', 'freq', 'context_freq'])

mean_rt_per_word = rt_data.groupby(['item', 'zone', 'word'])['RT'].mean().reset_index()
mean_rt_per_word.columns = ['item', 'zone', 'word', 'mean_RT']
mean_rt_per_word['word_length'] = mean_rt_per_word['word'].str.len()

freq_1gram['item'] = freq_1gram['token_code'].str.split('.').str[0].astype(int)
freq_1gram['zone'] = freq_1gram['token_code'].str.split('.').str[1].astype(int)

merged_data = mean_rt_per_word.merge(
    freq_1gram[['item', 'zone', 'freq']], 
    on=['item', 'zone'], 
    how='left'
)

merged_data['freq'] = pd.to_numeric(merged_data['freq'], errors='coerce')
merged_data = merged_data.dropna(subset=['freq'])
merged_data['log_freq'] = np.log10(merged_data['freq'] + 1)

print(f"Loaded {len(merged_data)} word instances")

# Lemmatization
print("\n" + "=" * 80)
print("PART 1: LEMMATIZATION AND FREQUENCY ORDERED BIN SEARCH")
print("=" * 80)

lemmatizer = WordNetLemmatizer()

def get_wordnet_pos(word):
    try:
        tag = nltk.pos_tag([word])[0][1][0].upper()
        tag_dict = {"J": wordnet.ADJ, "N": wordnet.NOUN, "V": wordnet.VERB, "R": wordnet.ADV}
        return tag_dict.get(tag, wordnet.NOUN)
    except:
        return wordnet.NOUN

def lemmatize_word(word):
    word_clean = word.lower().strip('.,!?;:"\'')
    pos = get_wordnet_pos(word_clean)
    return lemmatizer.lemmatize(word_clean, pos)

print("\nLemmatizing words...")
merged_data['word_clean'] = merged_data['word'].str.lower().str.strip('.,!?;:"\'')
merged_data['lemma'] = merged_data['word_clean'].apply(lemmatize_word)
merged_data['lemma_length'] = merged_data['lemma'].str.len()

print(f"Unique surface forms: {merged_data['word_clean'].nunique()}")
print(f"Unique lemmas: {merged_data['lemma'].nunique()}")

# Compute lemma frequencies
print("\nComputing lemma frequencies...")
lemma_freq_dict = {}

for lemma in merged_data['lemma'].unique():
    surface_forms = merged_data[merged_data['lemma'] == lemma]['word_clean'].unique()
    total_freq = 0
    for surface in surface_forms:
        surface_data = freq_1gram[freq_1gram['word'].str.lower().str.strip('.,!?;:"\'') == surface]
        if len(surface_data) > 0:
            total_freq += surface_data['freq'].sum()
    lemma_freq_dict[lemma] = total_freq

merged_data['lemma_freq'] = merged_data['lemma'].map(lemma_freq_dict)
merged_data['log_lemma_freq'] = np.log10(merged_data['lemma_freq'] + 1)

print(f"Computed frequencies for {len(lemma_freq_dict)} lemmas")

# FOBS Structure
print("\n" + "=" * 80)
print("CONSTRUCTING FOBS STRUCTURE")
print("=" * 80)

class FOBSMemory:
    def __init__(self, word_freq_dict, lemma_freq_dict):
        self.word_freq_dict = word_freq_dict
        self.lemma_freq_dict = lemma_freq_dict
        self.word_bins = self._create_bins(word_freq_dict)
        self.lemma_bins = self._create_bins(lemma_freq_dict)
    
    def _create_bins(self, freq_dict):
        sorted_items = sorted(freq_dict.items(), key=lambda x: x[1], reverse=True)
        bins = defaultdict(list)
        for word, freq in sorted_items:
            bin_key = 0 if freq == 0 else int(np.log10(freq + 1))
            bins[bin_key].append((word, freq))
        return dict(bins)
    
    def search_depth(self, word, bin_type='word'):
        bins = self.word_bins if bin_type == 'word' else self.lemma_bins
        freq_dict = self.word_freq_dict if bin_type == 'word' else self.lemma_freq_dict
        
        if word not in freq_dict:
            return None
        
        freq = freq_dict[word]
        bin_key = 0 if freq == 0 else int(np.log10(freq + 1))
        
        if bin_key not in bins:
            return None
        
        position = next((i for i, (w, f) in enumerate(bins[bin_key]) if w == word), None)
        if position is None:
            return None
        
        depth = sum(len(bins[k]) for k in bins if k > bin_key) + position + 1
        return depth
    
    def get_bin_statistics(self):
        word_bin_stats = {k: len(v) for k, v in self.word_bins.items()}
        lemma_bin_stats = {k: len(v) for k, v in self.lemma_bins.items()}
        return word_bin_stats, lemma_bin_stats

word_freq_dict = dict(zip(merged_data['word_clean'], merged_data['freq']))
fobs = FOBSMemory(word_freq_dict, lemma_freq_dict)

merged_data['word_search_depth'] = merged_data['word_clean'].apply(
    lambda x: fobs.search_depth(x, 'word')
)
merged_data['lemma_search_depth'] = merged_data['lemma'].apply(
    lambda x: fobs.search_depth(x, 'lemma')
)

word_bin_stats, lemma_bin_stats = fobs.get_bin_statistics()

print("\nFOBS Bin Statistics:")
print("-" * 60)
print(f"Word bins: {len(word_bin_stats)}")
print(f"Lemma bins: {len(lemma_bin_stats)}")
print("\nBin distribution (Word):")
for bin_key in sorted(word_bin_stats.keys(), reverse=True):
    print(f"  Bin {bin_key} (freq 10^{bin_key}): {word_bin_stats[bin_key]} words")

print("\nBin distribution (Lemma):")
for bin_key in sorted(lemma_bin_stats.keys(), reverse=True):
    print(f"  Bin {bin_key} (freq 10^{bin_key}): {lemma_bin_stats[bin_key]} lemmas")

# Hypothesis 1: Root vs Surface Frequency
print("\n" + "=" * 80)
print("HYPOTHESIS 1: Root Frequency vs Surface Frequency")
print("=" * 80)

model_data = merged_data.dropna(subset=['log_freq', 'log_lemma_freq', 'word_length', 'lemma_length'])
print(f"\nModeling dataset: {len(model_data)} words")

# Model 1: Surface
X1 = model_data[['log_freq', 'word_length']].values
y = model_data['mean_RT'].values

model1 = LinearRegression()
model1.fit(X1, y)
y_pred1 = model1.predict(X1)

r2_1 = r2_score(y, y_pred1)
rmse_1 = np.sqrt(mean_squared_error(y, y_pred1))
mae_1 = mean_absolute_error(y, y_pred1)

formula1 = 'mean_RT ~ log_freq + word_length'
sm_model1 = ols(formula1, data=model_data).fit()

print("\nMODEL 1: Mean RT ~ surface_freq + surface_length")
print("-" * 60)
print(f"R² Score: {r2_1:.4f}")
print(f"RMSE: {rmse_1:.4f}")
print(f"MAE: {mae_1:.4f}")
print(f"AIC: {sm_model1.aic:.4f}")
print(f"BIC: {sm_model1.bic:.4f}")

# Model 2: Lemma
X2 = model_data[['log_lemma_freq', 'lemma_length']].values

model2 = LinearRegression()
model2.fit(X2, y)
y_pred2 = model2.predict(X2)

r2_2 = r2_score(y, y_pred2)
rmse_2 = np.sqrt(mean_squared_error(y, y_pred2))
mae_2 = mean_absolute_error(y, y_pred2)

formula2 = 'mean_RT ~ log_lemma_freq + lemma_length'
sm_model2 = ols(formula2, data=model_data).fit()

print("\nMODEL 2: Mean RT ~ lemma_freq + lemma_length")
print("-" * 60)
print(f"R² Score: {r2_2:.4f}")
print(f"RMSE: {rmse_2:.4f}")
print(f"MAE: {mae_2:.4f}")
print(f"AIC: {sm_model2.aic:.4f}")
print(f"BIC: {sm_model2.bic:.4f}")

better = "Lemma Frequency" if r2_2 > r2_1 else "Surface Frequency"
print(f"\nCONCLUSION: {better} is a better predictor of reading time")

# Hypothesis 2: Pseudo vs Real Affixes - FIXED VERSION
print("\n" + "=" * 80)
print("HYPOTHESIS 2: Pseudo-Affixes vs Real Affixes (FREQUENCY MATCHED)")
print("=" * 80)

# STEP 1: Find all -er words in corpus with their stats
print("\nSTEP 1: Finding all -er suffix words in corpus...")
er_candidates = merged_data[merged_data['word_clean'].str.endswith('er')].copy()

# Aggregate stats per unique word
er_stats = er_candidates.groupby('word_clean').agg({
    'mean_RT': 'mean',
    'freq': 'first',
    'word_length': 'first'
}).reset_index()

print(f"Found {len(er_stats)} unique -er words in corpus")
print(f"Frequency range: {er_stats['freq'].min():.0f} to {er_stats['freq'].max():.0f}")
print(f"Length range: {er_stats['word_length'].min()} to {er_stats['word_length'].max()}")

# STEP 2: Define target frequency and length ranges
TARGET_FREQ_MIN = 500_000      # 500k
TARGET_FREQ_MAX = 10_000_000   # 10M
TARGET_LENGTH_MIN = 5
TARGET_LENGTH_MAX = 7

print(f"\nSTEP 2: Filtering for matched words:")
print(f"  Target frequency: {TARGET_FREQ_MIN:,} - {TARGET_FREQ_MAX:,}")
print(f"  Target length: {TARGET_LENGTH_MIN} - {TARGET_LENGTH_MAX} characters")

er_filtered = er_stats[
    (er_stats['freq'] >= TARGET_FREQ_MIN) &
    (er_stats['freq'] <= TARGET_FREQ_MAX) &
    (er_stats['word_length'] >= TARGET_LENGTH_MIN) &
    (er_stats['word_length'] <= TARGET_LENGTH_MAX)
].copy()

print(f"\n{len(er_filtered)} words match the criteria:")
print(er_filtered[['word_clean', 'freq', 'word_length', 'mean_RT']].to_string(index=False))

# STEP 3: Manually classify available words
print("\n" + "-" * 60)
print("STEP 3: Manual classification of available words")
print("-" * 60)

# Pseudo-affixed: -er is NOT a suffix (non-decomposable)
pseudo_candidates = ['finger', 'under', 'never', 'corner', 'manner', 'umber', 'inner', 'outer', 
                     'dinner', 'winter', 'summer', 'silver', 'proper', 'river', 'offer']

# Real affixed: -er IS a suffix (verb -> noun conversion)
real_candidates = ['teacher', 'farmer', 'worker', 'writer', 'speaker', 'maker', 'player',
                   'owner', 'buyer', 'seller', 'runner', 'helper', 'lover', 'killer']

# Find which candidates exist in our filtered dataset
available_words = set(er_filtered['word_clean'].tolist())

pseudo_available = [w for w in pseudo_candidates if w in available_words]
real_available = [w for w in real_candidates if w in available_words]

print(f"\nPseudo-affixed words available: {pseudo_available}")
print(f"Real affixed words available: {real_available}")

# If we have at least 3 of each, proceed
if len(pseudo_available) >= 3 and len(real_available) >= 3:
    # Further match frequencies between groups
    pseudo_df = er_filtered[er_filtered['word_clean'].isin(pseudo_available)]
    real_df = er_filtered[er_filtered['word_clean'].isin(real_available)]
    
    # Find frequency overlap
    pseudo_freq_range = (pseudo_df['freq'].min(), pseudo_df['freq'].max())
    real_freq_range = (real_df['freq'].min(), real_df['freq'].max())
    
    overlap_min = max(pseudo_freq_range[0], real_freq_range[0])
    overlap_max = min(pseudo_freq_range[1], real_freq_range[1])
    
    print(f"\nFrequency overlap: {overlap_min:.0f} - {overlap_max:.0f}")
    
    # Select words within overlap
    pseudo_final = pseudo_df[
        (pseudo_df['freq'] >= overlap_min) & (pseudo_df['freq'] <= overlap_max)
    ].copy()
    real_final = real_df[
        (real_df['freq'] >= overlap_min) & (real_df['freq'] <= overlap_max)
    ].copy()
    
    print(f"\nFinal selection:")
    print(f"  Pseudo: {len(pseudo_final)} words")
    print(f"  Real: {len(real_final)} words")
    
    if len(pseudo_final) >= 2 and len(real_final) >= 2:
        # Create classification dictionary
        pseudo_affixed = {row['word_clean']: {'base': row['word_clean'][:-2], 'affix': 'er', 'type': 'pseudo'}
                          for _, row in pseudo_final.iterrows()}
        real_affixed = {row['word_clean']: {'base': row['word_clean'][:-2], 'affix': 'er', 'type': 'real'}
                        for _, row in real_final.iterrows()}
        
        test_words = list(pseudo_affixed.keys()) + list(real_affixed.keys())
        
        # Get RT data for these words
        test_data = merged_data[merged_data['word_clean'].isin(test_words)].copy()
        
        def classify_affix(word):
            if word in pseudo_affixed:
                return 'pseudo'
            elif word in real_affixed:
                return 'real'
            return None
        
        test_data['affix_type'] = test_data['word_clean'].apply(classify_affix)
        test_data = test_data.dropna(subset=['affix_type'])
        
        print("\n" + "=" * 80)
        print("DETAILED STATISTICS")
        print("=" * 80)
        
        # Aggregate by word
        pseudo_stats = test_data[test_data['affix_type'] == 'pseudo'].groupby('word_clean').agg({
            'mean_RT': ['mean', 'std', 'count'],
            'word_length': 'first',
            'freq': 'first'
        }).reset_index()
        pseudo_stats.columns = ['word', 'mean_RT', 'std_RT', 'count', 'length', 'freq']
        
        real_stats = test_data[test_data['affix_type'] == 'real'].groupby('word_clean').agg({
            'mean_RT': ['mean', 'std', 'count'],
            'word_length': 'first',
            'freq': 'first'
        }).reset_index()
        real_stats.columns = ['word', 'mean_RT', 'std_RT', 'count', 'length', 'freq']
        
        print("\nPseudo-Affixed Words:")
        print("-" * 80)
        print(f"{'Word':<15} {'Mean RT':<12} {'Std RT':<12} {'Count':<8} {'Length':<8} {'Frequency':<12}")
        print("-" * 80)
        for _, row in pseudo_stats.iterrows():
            print(f"{row['word']:<15} {row['mean_RT']:<12.2f} {row['std_RT']:<12.2f} "
                  f"{int(row['count']):<8} {int(row['length']):<8} {int(row['freq']):<12}")
        
        print(f"\nMean: {pseudo_stats['mean_RT'].mean():.2f} ms")
        print(f"Frequency range: {pseudo_stats['freq'].min():.0f} - {pseudo_stats['freq'].max():.0f}")
        
        print("\nReal Affixed Words:")
        print("-" * 80)
        print(f"{'Word':<15} {'Mean RT':<12} {'Std RT':<12} {'Count':<8} {'Length':<8} {'Frequency':<12}")
        print("-" * 80)
        for _, row in real_stats.iterrows():
            print(f"{row['word']:<15} {row['mean_RT']:<12.2f} {row['std_RT']:<12.2f} "
                  f"{int(row['count']):<8} {int(row['length']):<8} {int(row['freq']):<12}")
        
        print(f"\nMean: {real_stats['mean_RT'].mean():.2f} ms")
        print(f"Frequency range: {real_stats['freq'].min():.0f} - {real_stats['freq'].max():.0f}")
        
        # Statistical test
        pseudo_rts = test_data[test_data['affix_type'] == 'pseudo']['mean_RT'].values
        real_rts = test_data[test_data['affix_type'] == 'real']['mean_RT'].values
        
        t_stat, p_value = ttest_ind(pseudo_rts, real_rts)
        
        print("\n" + "=" * 80)
        print("STATISTICAL TEST")
        print("=" * 80)
        print(f"Pseudo-affixed: n={len(pseudo_rts)}, mean={np.mean(pseudo_rts):.2f} ms, SD={np.std(pseudo_rts):.2f}")
        print(f"Real affixed:   n={len(real_rts)}, mean={np.mean(real_rts):.2f} ms, SD={np.std(real_rts):.2f}")
        print(f"\nt-statistic: {t_stat:.4f}")
        print(f"p-value: {p_value:.4f}")
        print(f"\nResult: {'SIGNIFICANT' if p_value < 0.05 else 'NOT SIGNIFICANT'} at α=0.05")
        
        if p_value < 0.05:
            if np.mean(pseudo_rts) > np.mean(real_rts):
                print("Conclusion: Pseudo-affixed words take LONGER to process")
            else:
                print("Conclusion: Real affixed words take LONGER to process")
        else:
            print("Conclusion: No significant difference in processing time")
        
        # Visualization
        fig, axes = plt.subplots(2, 2, figsize=(16, 12))
        
        # Box plot
        axes[0, 0].boxplot([pseudo_rts, real_rts], labels=['Pseudo-Affixed', 'Real Affixed'])
        axes[0, 0].set_ylabel('Mean RT (ms)', fontsize=11)
        axes[0, 0].set_title(f'Reading Time Comparison\nPseudo vs Real Affixes\np = {p_value:.4f}', 
                             fontsize=12, fontweight='bold')
        axes[0, 0].grid(True, alpha=0.3)
        
        # Violin plot
        parts = axes[0, 1].violinplot([pseudo_rts, real_rts], positions=[1, 2], 
                                      showmeans=True, showmedians=True)
        axes[0, 1].set_xticks([1, 2])
        axes[0, 1].set_xticklabels(['Pseudo-Affixed', 'Real Affixed'])
        axes[0, 1].set_ylabel('Mean RT (ms)', fontsize=11)
        axes[0, 1].set_title('Distribution Comparison', fontsize=12, fontweight='bold')
        axes[0, 1].grid(True, alpha=0.3)
        
        # Individual word RTs
        x_pseudo = list(range(len(pseudo_stats)))
        x_real = list(range(len(real_stats)))
        
        axes[1, 0].bar(x_pseudo, pseudo_stats['mean_RT'], color='coral', alpha=0.7, 
                       label=f'Pseudo mean: {pseudo_stats["mean_RT"].mean():.1f}ms')
        axes[1, 0].bar([x + len(pseudo_stats) + 1 for x in x_real], real_stats['mean_RT'], 
                       color='skyblue', alpha=0.7, label=f'Real mean: {real_stats["mean_RT"].mean():.1f}ms')
        axes[1, 0].axhline(pseudo_stats['mean_RT'].mean(), color='red', linestyle='--', alpha=0.7)
        axes[1, 0].axhline(real_stats['mean_RT'].mean(), color='blue', linestyle='--', alpha=0.7)
        axes[1, 0].set_xticks(list(range(len(pseudo_stats))) + 
                              [x + len(pseudo_stats) + 1 for x in range(len(real_stats))])
        axes[1, 0].set_xticklabels(list(pseudo_stats['word']) + list(real_stats['word']), 
                                   rotation=45, ha='right')
        axes[1, 0].set_ylabel('Mean RT (ms)', fontsize=11)
        axes[1, 0].set_title('Individual Word Reading Times', fontsize=12, fontweight='bold')
        axes[1, 0].legend()
        axes[1, 0].grid(True, alpha=0.3, axis='y')
        
        # Frequency effect
        axes[1, 1].scatter(np.log10(pseudo_stats['freq']), pseudo_stats['mean_RT'], 
                          color='coral', s=100, alpha=0.7, label='Pseudo-Affixed')
        axes[1, 1].scatter(np.log10(real_stats['freq']), real_stats['mean_RT'], 
                          color='skyblue', s=100, alpha=0.7, label='Real Affixed')
        axes[1, 1].set_xlabel('Log Frequency', fontsize=11)
        axes[1, 1].set_ylabel('Mean RT (ms)', fontsize=11)
        axes[1, 1].set_title('Frequency Effect by Affix Type', fontsize=12, fontweight='bold')
        axes[1, 1].legend()
        axes[1, 1].grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig('fobs_hypothesis2_affixes.png', dpi=300, bbox_inches='tight')
        print("\nVisualization saved as 'fobs_hypothesis2_affixes.png'")
        plt.close()
        
    else:
        print("\n✗ Insufficient frequency-matched words found")
else:
    print("\n✗ Insufficient test words available in corpus")
    print("This is a limitation of using naturalistic corpus data.")

# FOBS Hypothesis 1 visualization
fig, axes = plt.subplots(2, 2, figsize=(16, 12))

# Surface frequency vs RT
axes[0, 0].scatter(model_data['log_freq'], model_data['mean_RT'], alpha=0.5, s=20)
z = np.polyfit(model_data['log_freq'], model_data['mean_RT'], 1)
p = np.poly1d(z)
axes[0, 0].plot(sorted(model_data['log_freq']), p(sorted(model_data['log_freq'])), "r--", lw=2)
axes[0, 0].set_xlabel('Log Surface Frequency', fontsize=11)
axes[0, 0].set_ylabel('Mean RT (ms)', fontsize=11)
axes[0, 0].set_title('Surface Frequency vs RT', fontsize=12, fontweight='bold')
axes[0, 0].grid(True, alpha=0.3)

# Lemma frequency vs RT
axes[0, 1].scatter(model_data['log_lemma_freq'], model_data['mean_RT'], alpha=0.5, s=20)
z = np.polyfit(model_data['log_lemma_freq'], model_data['mean_RT'], 1)
p = np.poly1d(z)
axes[0, 1].plot(sorted(model_data['log_lemma_freq']), p(sorted(model_data['log_lemma_freq'])), "r--", lw=2)
axes[0, 1].set_xlabel('Log Lemma Frequency', fontsize=11)
axes[0, 1].set_ylabel('Mean RT (ms)', fontsize=11)
axes[0, 1].set_title('Lemma Frequency vs RT', fontsize=12, fontweight='bold')
axes[0, 1].grid(True, alpha=0.3)

# Model comparison
models = ['Surface', 'Lemma']
r2_scores = [r2_1, r2_2]
rmse_scores = [rmse_1, rmse_2]
mae_scores = [mae_1, mae_2]

x_pos = np.arange(len(models))
width = 0.25

axes[1, 0].bar(x_pos - width, r2_scores, width, label='R²', color='steelblue')
axes[1, 0].bar(x_pos, [r/10 for r in rmse_scores], width, label='RMSE/10', color='coral')
axes[1, 0].bar(x_pos + width, [m/10 for m in mae_scores], width, label='MAE/10', color='lightgreen')
axes[1, 0].set_ylabel('Score', fontsize=11)
axes[1, 0].set_title('Model Comparison', fontsize=12, fontweight='bold')
axes[1, 0].set_xticks(x_pos)
axes[1, 0].set_xticklabels(models)
axes[1, 0].legend()
axes[1, 0].grid(True, alpha=0.3, axis='y')

# FOBS search depth correlation
depth_data = model_data.dropna(subset=['word_search_depth', 'lemma_search_depth'])

if len(depth_data) > 0:
    corr_word_depth = pearsonr(depth_data['word_search_depth'], depth_data['mean_RT'])
    corr_lemma_depth = pearsonr(depth_data['lemma_search_depth'], depth_data['mean_RT'])
    
    x_pos = [0, 1]
    correlations = [abs(corr_word_depth[0]), abs(corr_lemma_depth[0])]
    p_values = [corr_word_depth[1], corr_lemma_depth[1]]
    
    bars = axes[1, 1].bar(x_pos, correlations, color=['steelblue', 'coral'])
    axes[1, 1].set_ylabel('|Correlation with RT|', fontsize=11)
    axes[1, 1].set_title('FOBS Search Depth Correlation', fontsize=12, fontweight='bold')
    axes[1, 1].set_xticks(x_pos)
    axes[1, 1].set_xticklabels(['Word\nSearch Depth', 'Lemma\nSearch Depth'])
    axes[1, 1].grid(True, alpha=0.3, axis='y')
    
    for i, (bar, corr, pval) in enumerate(zip(bars, correlations, p_values)):
        height = bar.get_height()
        axes[1, 1].text(bar.get_x() + bar.get_width()/2., height,
                       f'{corr:.3f}\n(p={pval:.0e})',
                       ha='center', va='bottom', fontsize=9, fontweight='bold')

plt.tight_layout()
plt.savefig('fobs_hypothesis1.png', dpi=300, bbox_inches='tight')
print("\nFOBS Hypothesis 1 visualization saved")
plt.close()

print("\n" + "=" * 80)
print("FOBS ANALYSIS COMPLETE")
print("=" * 80)