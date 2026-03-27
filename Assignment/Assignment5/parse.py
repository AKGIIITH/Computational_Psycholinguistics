"""
Probabilistic Earley Parser
Usage:  python parse.py grammar.gr sentences.sen

Outputs the exact format requested:
- Parse tree (Lisp-style S-expression)
- Log2 cost (negative log base 2)
- "NONE" if no parse is found
"""

import sys
import math
from collections import defaultdict

# Load Grammar
def load_grammar(path):
    """
    Read a PCFG and convert probabilities to base-2 costs.
    cost = -log2(probability)
    """
    rules = []
    by_lhs = defaultdict(list)
    nts = set()
    rule_map = {}

    with open(path) as fh:
        for raw in fh:
            raw = raw.strip()
            if not raw or raw.startswith('//'):
                continue
            
            parts = raw.split()
            if len(parts) < 3:
                continue
            
            try:
                p = float(parts[0])
                # Convert probability to base-2 cost (lower is better)
                cost = -math.log2(p) if p > 0 else float('inf')
            except ValueError:
                continue

            lhs = parts[1]
            rhs = tuple(parts[2:])

            idx = len(rules)
            by_lhs[lhs].append(idx)
            nts.add(lhs)
            rule_map[(lhs, rhs)] = cost
            rules.append((cost, lhs, rhs))

    return rules, by_lhs, nts, rule_map


# Earley Algorithm
def earley(words, rules, by_lhs, nts, start='ROOT'):
    """
    Core Earley recognizer. Tracks minimum cost and all backpointers.
    """
    n = len(words)
    chart = [dict() for _ in range(n + 1)]
    agenda = [[] for _ in range(n + 1)]

    def add(col, ridx, dot, origin, cost, back=None):
        key = (ridx, dot, origin)
        if key not in chart[col]:
            chart[col][key] = {
                'cost': cost,
                'backs': [] if back is None else [back]
            }
            agenda[col].append(key)
        else:
            entry = chart[col][key]
            # Viterbi update: keep the lowest cost
            if cost < entry['cost']:
                entry['cost'] = cost
            # Exhaustive tracking: keep all distinct backpointers
            if back is not None and back not in entry['backs']:
                entry['backs'].append(back)

    # Initialize agenda with start symbol
    for ridx in by_lhs.get(start, []):
        add(0, ridx, 0, 0, rules[ridx][0])

    # Process chart column by column
    for j in range(n + 1):
        ptr = 0
        while ptr < len(agenda[j]):
            key = agenda[j][ptr]
            ptr += 1

            ridx, dot, origin = key
            _, lhs, rhs = rules[ridx]
            current_cost = chart[j][key]['cost']

            # COMPLETE
            if dot == len(rhs):
                for wkey, we in list(chart[origin].items()):
                    wridx, wdot, worigin = wkey
                    _, _, wrhs = rules[wridx]
                    if wdot < len(wrhs) and wrhs[wdot] == lhs:
                        new_cost = we['cost'] + current_cost
                        new_back = ('complete', wkey, key, origin)
                        add(j, wridx, wdot + 1, worigin, new_cost, new_back)

            else:
                sym = rhs[dot]

                # PREDICT
                if sym in nts:
                    for ridx2 in by_lhs[sym]:
                        add(j, ridx2, 0, j, rules[ridx2][0])

                # SCAN
                elif j < n and words[j] == sym:
                    add(j + 1, ridx, dot + 1, origin, current_cost, ('scan', key))

    return chart


# Print Tree
def all_trees(key, col, chart, rules, words, path=None):
    """Recursively yield all distinct parse trees, with cycle detection."""
    # Initialize the path for cycle detection
    if path is None:
        path = frozenset()
        
    state = (key, col)
    
    # CYCLE DETECTED: If we've already visited this exact item at this exact 
    # column in our current branch, we are in a unary loop. Break immediately.
    if state in path:
        return
        
    # Add current state to the path for downstream recursive calls
    new_path = path | {state}

    ridx, dot, origin = key
    _, lhs, _ = rules[ridx]

    entry = chart[col].get(key)
    if entry is None:
        return

    backs = entry['backs']

    if not backs:
        yield (lhs, [])
        return

    for back in backs:
        kind = back[0]

        if kind == 'scan':
            _, prev_key = back
            terminal = words[col - 1]
            # Pass new_path down
            for partial in all_trees(prev_key, col - 1, chart, rules, words, new_path):
                yield (partial[0], partial[1] + [terminal])

        elif kind == 'complete':
            _, left_key, right_key, mid = back
            # Pass new_path down to both left and right branches
            for partial in all_trees(left_key, mid, chart, rules, words, new_path):
                for sub in all_trees(right_key, col, chart, rules, words, new_path):
                    yield (partial[0], partial[1] + [sub])


def tree_cost(tree, rule_map):
    """Compute the total base-2 cost of a specific parse tree."""
    lhs, children = tree
    if not children:
        return 0.0

    child_syms = tuple(c if isinstance(c, str) else c[0] for c in children)
    cost = rule_map.get((lhs, child_syms), float('inf'))

    for child in children:
        if isinstance(child, tuple):
            cost += tree_cost(child, rule_map)

    return cost


def tree_to_str(tree, current_indent=0):
    """Convert parse tree to a pretty-printed S-expression string."""
    lhs, children = tree
    
    # Base case: no children
    if not children:
        return f"({lhs})"

    # Base case: pre-terminal (e.g., (Num 3) or (N time))
    # Keep these together on a single line
    if len(children) == 1 and isinstance(children[0], str):
        return f"({lhs} {children[0]})"

    # The prefix for this level, e.g., "(TERM "
    prefix = f"({lhs} "
    
    # The depth of the next child is the current depth + the length of the prefix
    child_indent = current_indent + len(prefix)

    lines = []
    for i, child in enumerate(children):
        # Format the child (recursive step)
        if isinstance(child, str):
            child_str = child
        else:
            child_str = tree_to_str(child, child_indent)
        
        # Place the first child right next to the prefix
        if i == 0:
            lines.append(prefix + child_str)
        # Place subsequent children on new lines, indented to match the first child
        else:
            lines.append((" " * child_indent) + child_str)
            
    # Accumulate the closing parenthesis on the very last line
    lines[-1] += ")"
    
    return "\n".join(lines).rstrip('\n')

def main():
    if len(sys.argv) != 3:
        print(__doc__)
        sys.exit("Error: expected exactly two arguments: grammar.gr sentences.sen")

    grammar_file = sys.argv[1]
    sentences_file = sys.argv[2]

    # Load PCFG
    rules, by_lhs, nts, rule_map = load_grammar(grammar_file)
    
    # Auto-detect root symbol
    start_sym = 'ROOT' if 'ROOT' in by_lhs else ('S' if 'S' in by_lhs else list(by_lhs.keys())[0])

    # Read sentences
    with open(sentences_file) as fh:
        sentences = [line.split() for line in fh if line.strip()]

    # Parse and Output
    for words in sentences:
        chart = earley(words, rules, by_lhs, nts, start=start_sym)
        n = len(words)

        # Look for completed start symbols covering the entire sentence
        complete_roots = [
            (ridx, len(rules[ridx][2]), 0)
            for ridx in by_lhs.get(start_sym, [])
            if (ridx, len(rules[ridx][2]), 0) in chart[n]
        ]

        if not complete_roots:
            print("NONE")
            continue

        valid_parses = []
        seen = set()

        # Enumerate trees
        for root_key in complete_roots:
            for tree in all_trees(root_key, n, chart, rules, words):
                ts = tree_to_str(tree)
                # Ensure no duplicates from alternative paths generating exact same structure
                if ts not in seen:
                    seen.add(ts)
                    tc = tree_cost(tree, rule_map)
                    valid_parses.append((tc, ts))
        
        # Sort by cost ascending (best trees first)
        valid_parses.sort(key=lambda x: x[0])

        # Print outputs
        for cost, ts in valid_parses:
            print(ts)
            print(cost)

if __name__ == '__main__':
    main()