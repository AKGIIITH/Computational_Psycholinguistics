# Probabilistic Earley Parser

## Quick Start

```bash
python parse.py grammar.gr sentences.sen
```

No external libraries needed — only Python 3.6+ standard library.

---

## Files Included

| File | Purpose |
|------|---------|
| `parse.py` | Main parser (submit this) |
| `./time/time.gr` | Figure 1 grammar |
| `./time/time.sen` | "time flies like an arrow" |
| `./soldier/soldier.gr` | Grammar designed for Q3 |
| `./soldier/soldier.sen` | "the man shot the soldier with a gun" |

---

## Step-by-Step Setup

### 1 — Prerequisites
- Python 3.6 or newer (check with `python --version`)
- No `pip install` required

### 2 — Download Files
Download the grammar and sentence files.

### 3 — Run

```bash
python parse.py grammar.gr sentences.sen
```

---

## Grammar File Format

One rule per line:
```
probability   LHS   RHS_sym_1   [RHS_sym_2 ...]
```

**Note:** No `->` separator. The RHS symbols start immediately after the LHS.

Example from Figure 1:
```
1.0  S  NP  VP
0.25 NP  N   N
0.4  NP  D   N
0.35 NP  N
0.6  VP  V   NP
0.4  VP  V   ADVP
1.0  ADVP  ADV NP
0.4  N   time
0.2  N   flies
0.4  N   arrow
1.0  D   an
1.0  V   like
1.0  ADV  like
```

Rules:
- Case-sensitive (`The` ≠ `the`)
- At least one RHS symbol (no epsilon rules)
- Probabilities per LHS already sum to 1.0
- Comments start with `//`

---

## Output Format

For each sentence the program prints:

1. **The Earley Chart** — every column (Chart[0] … Chart[n]), each showing  
   `[done] [i,j]  LHS -> matched • remaining   logp=…`  
   ✓ marks complete items (dot at end of rule).

2. **Parse Trees** — every distinct parse tree found, with:
   - `Probability` = exp(sum of log rule probabilities)
   - `Log-prob`    = sum of log(rule_prob) for every rule in the tree
   - Bracketed tree in `(LHS child1 child2 …)` notation

---

## Implementation Details (Q4 Answer)

### Correctness — tracking best derivation and multiple derivations

Every chart entry `chart[j][(ridx, dot, origin)]` stores two things:

```python
{
    'w':     best_log_probability,   # Viterbi weight
    'backs': [back1, back2, ...]     # ALL backpointers (all derivations)
}
```

**When a new item is first inserted** its log-weight and a single backpointer
are stored.

**When the same item arrives again via a different derivation** two things
happen independently:
- If the new log-weight is higher → `entry['w']` is updated (Viterbi update).
- The new backpointer is appended to `entry['backs']` if it is not already
  present. This keeps every distinct derivation for exhaustive tree extraction.

This means the `w` field always holds the weight of the *best* derivation seen
so far, while `backs` retains *all* derivations so every parse tree can be
enumerated in the post-parse step.

### Efficiency — O(n²) space and O(n³) time

**O(n²) space**  
`chart` contains n+1 columns. Each item key is `(rule_idx, dot, origin)`.
For a fixed grammar G:
- `rule_idx ∈ {0 … |G|-1}` (constant)
- `dot ∈ {0 … max_rhs_len}` (constant)
- `origin ∈ {0 … j}` (at most n+1 values per column)

So each column holds at most O(|G| · n) distinct items → total O(n²) items.

**O(n³) time**  
The three operations:
- **PREDICT**: For each item at column j that expects non-terminal Y, add
  all rules for Y. O(|G|) work per item, dominated by other steps.
- **SCAN**: O(1) per item — just add one item to the next column.
- **COMPLETE**: For each complete item `(Y → γ •, i, j)`, scan `chart[i]`
  for items waiting for Y. `chart[i]` has O(|G| · i) ≤ O(|G| · n) items.
  Over all (i, j) pairs — O(n²) pairs — this gives O(n³ |G|) = **O(n³)**.

**O(1) push (including duplicate check)**  
`chart[col]` is a Python `dict`. Both `key not in chart[col]` (duplicate
check) and `chart[col][key] = …` (insert) run in **O(1) amortised** time
because Python dicts use hash tables.

Without this O(1) check, the same item could be added to `agenda[col]`
multiple times. Each re-processed item triggers the COMPLETE loop again,
potentially adding more duplicates, leading to worst-case exponential blowup.
The dict-based deduplication ensures each item is *processed* at most once.

---

## Q3 — Grammar Design for PP-Attachment Ambiguity

The sentence *"the man shot the soldier with a gun"* is structurally
ambiguous. Two parse trees arise depending on where the PP *"with a gun"*
attaches:

**Parse 1 — NP attachment** (the soldier *had* a gun):
```
(S
  (NP (DT the) (N man))
  (VP (V shot)
    (NP (DT the) (N soldier)
      (PP (P with) (NP (DT a) (N gun))))))
```
Rule used: `NP  DT N PP`  (PP inside the object NP)

**Parse 2 — VP attachment** (the man *used* a gun):
```
(S
  (NP (DT the) (N man))
  (VP (V shot)
    (NP (DT the) (N soldier))
    (PP (P with) (NP (DT a) (N gun)))))
```
Rule used: `VP  V NP PP`  (PP inside the VP)

Both trees have equal probability (0.000122) under the symmetric grammar
provided because `P(NP→DT N PP) = P(VP→V NP PP) = 0.5`.

---

## Verification of Q2 Probabilities

For *"time flies like an arrow"* the two parses and their probabilities:

**Tree 1**: `[time flies]_NP  [like [an arrow]_NP]_VP`  
`= P(S  NP VP) × P(NP  N N) × P(N  time) × P(N  flies)`  
`× P(VP  V NP) × P(V  like) × P(NP  D N) × P(D  an) × P(N  arrow)`  
`= 1 × 0.25 × 0.4 × 0.2 × 0.6 × 0.5 × 0.4 × 1 × 0.4`  
`= **0.00096**`

**Tree 2**: `[time]_NP  [flies [like [an arrow]_NP]_ADVP]_VP`  
`= P(S  NP VP) × P(NP  N) × P(N  time)`  
`× P(VP  V ADVP) × P(V  flies) × P(ADVP  ADV NP)`  
`× P(ADV  like) × P(NP  D N) × P(D  an) × P(N  arrow)`  
`= 1 × 0.35 × 0.4 × 0.4 × 0.5 × 1 × 1 × 0.4 × 1 × 0.4`  
`= **0.00448**`

Tree 2 is ~4.67× more probable under this grammar.