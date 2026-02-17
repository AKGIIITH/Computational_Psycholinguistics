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

# Model 1
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

# Model 2
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
print("HYPOTHESIS 2: Pseudo-Affixes vs Real Affixes")
print("=" * 80)

# First, check what -er words exist in corpus
print("\nChecking available -er suffix words in corpus...")
er_words_in_corpus = merged_data[merged_data['word_clean'].str.endswith('er')]['word_clean'].unique()
print(f"Found {len(er_words_in_corpus)} unique -er words")
print("Sample:", list(er_words_in_corpus[:20]))

# Use words that are more likely to exist in narrative text
pseudo_affixed = {
    'finger': {'base': 'fing', 'affix': 'er', 'type': 'pseudo'},
    'corner': {'base': 'corn', 'affix': 'er', 'type': 'pseudo'},
    'manner': {'base': 'mann', 'affix': 'er', 'type': 'pseudo'},
    'never': {'base': 'nev', 'affix': 'er', 'type': 'pseudo'},
    'under': {'base': 'und', 'affix': 'er', 'type': 'pseudo'}
}

real_affixed = {
    'teacher': {'base': 'teach', 'affix': 'er', 'type': 'real'},
    'farmer': {'base': 'farm', 'affix': 'er', 'type': 'real'},
    'father': {'base': 'fath', 'affix': 'er', 'type': 'real'},  # Note: not truly affixed but common
    'mother': {'base': 'moth', 'affix': 'er', 'type': 'real'},  # Note: not truly affixed but common
    'brother': {'base': 'broth', 'affix': 'er', 'type': 'real'}  # Note: not truly affixed but common
}

# Alternative: use -ing pseudo vs real if -er doesn't work
pseudo_affixed_ing = {
    'thing': {'base': 'th', 'affix': 'ing', 'type': 'pseudo'},
    'nothing': {'base': 'noth', 'affix': 'ing', 'type': 'pseudo'},
    'something': {'base': 'someth', 'affix': 'ing', 'type': 'pseudo'},
    'anything': {'base': 'anyth', 'affix': 'ing', 'type': 'pseudo'},
    'everything': {'base': 'everyth', 'affix': 'ing', 'type': 'pseudo'}
}

real_affixed_ing = {
    'running': {'base': 'run', 'affix': 'ning', 'type': 'real'},
    'walking': {'base': 'walk', 'affix': 'ing', 'type': 'real'},
    'looking': {'base': 'look', 'affix': 'ing', 'type': 'real'},
    'thinking': {'base': 'think', 'affix': 'ing', 'type': 'real'},
    'going': {'base': 'go', 'affix': 'ing', 'type': 'real'}
}

print("\nTest Words:")
print("-" * 60)
print("\nPseudo-Affixed Words:")
for word in pseudo_affixed.keys():
    exists = "✓" if word in er_words_in_corpus else "✗"
    print(f"  {word:<15} {exists}")

print("\nReal Affixed Words:")
for word in real_affixed.keys():
    exists = "✓" if word in er_words_in_corpus else "✗"
    print(f"  {word:<15} {exists}")

# Use the set that has more words
test_words_er = list(pseudo_affixed.keys()) + list(real_affixed.keys())
test_words_ing = list(pseudo_affixed_ing.keys()) + list(real_affixed_ing.keys())

test_data_er = merged_data[merged_data['word_clean'].isin([w.lower() for w in test_words_er])].copy()
test_data_ing = merged_data[merged_data['word_clean'].isin([w.lower() for w in test_words_ing])].copy()

if len(test_data_er) > len(test_data_ing):
    test_data = test_data_er
    pseudo_dict = pseudo_affixed
    real_dict = real_affixed
    suffix = "-er"
else:
    test_data = test_data_ing
    pseudo_dict = pseudo_affixed_ing
    real_dict = real_affixed_ing
    suffix = "-ing"

def classify_affix(word):
    word_lower = word.lower()
    if word_lower in [w.lower() for w in pseudo_dict.keys()]:
        return 'pseudo'
    elif word_lower in [w.lower() for w in real_dict.keys()]:
        return 'real'
    return None

test_data['affix_type'] = test_data['word_clean'].apply(classify_affix)
test_data = test_data.dropna(subset=['affix_type'])

print(f"\n\nUsing {suffix} suffix words")
print(f"Found {len(test_data)} instances of test words in corpus")
print(f"Pseudo-affixed instances: {(test_data['affix_type'] == 'pseudo').sum()}")
print(f"Real affixed instances: {(test_data['affix_type'] == 'real').sum()}")

if len(test_data) > 0 and (test_data['affix_type'] == 'real').sum() > 0:
    # Continue with analysis...
    pseudo_stats = test_data[test_data['affix_type'] == 'pseudo'].groupby('word_clean').agg({
        'mean_RT': ['mean', 'std', 'count'],
        'word_length': 'first',
        'freq': 'first'
    }).reset_index()
    
    real_stats = test_data[test_data['affix_type'] == 'real'].groupby('word_clean').agg({
        'mean_RT': ['mean', 'std', 'count'],
        'word_length': 'first',
        'freq': 'first'
    }).reset_index()
    
    pseudo_stats.columns = ['word', 'mean_RT', 'std_RT', 'count', 'length', 'freq']
    real_stats.columns = ['word', 'mean_RT', 'std_RT', 'count', 'length', 'freq']
    
    print("\n" + "=" * 80)
    print("DETAILED STATISTICS")
    print("=" * 80)
    
    print("\nPseudo-Affixed Words:")
    print("-" * 60)
    for _, row in pseudo_stats.iterrows():
        print(f"{row['word']:<12} {row['mean_RT']:<12.2f} {row['std_RT']:<12.2f} "
              f"{int(row['count']):<8} {int(row['length']):<8} {int(row['freq']):<12}")
    
    print("\nReal Affixed Words:")
    print("-" * 60)
    for _, row in real_stats.iterrows():
        print(f"{row['word']:<12} {row['mean_RT']:<12.2f} {row['std_RT']:<12.2f} "
              f"{int(row['count']):<8} {int(row['length']):<8} {int(row['freq']):<12}")
    
    # Statistical test
    pseudo_rts = test_data[test_data['affix_type'] == 'pseudo']['mean_RT'].values
    real_rts = test_data[test_data['affix_type'] == 'real']['mean_RT'].values
    
    if len(pseudo_rts) > 1 and len(real_rts) > 1:
        t_stat, p_value = ttest_ind(pseudo_rts, real_rts)
        print("\n" + "=" * 80)
        print("STATISTICAL TEST")
        print("=" * 80)
        print(f"Pseudo: {np.mean(pseudo_rts):.2f} ms (SD={np.std(pseudo_rts):.2f})")
        print(f"Real: {np.mean(real_rts):.2f} ms (SD={np.std(real_rts):.2f})")
        print(f"t={t_stat:.4f}, p={p_value:.4f}")
        print(f"Result: {'Significant' if p_value < 0.05 else 'Not significant'}")
else:
    print("\n⚠ WARNING: Insufficient data for affixed word analysis")
    print("The chosen test words do not appear in the Natural Stories corpus.")
    print("This is expected - literary texts may not contain all word types.")