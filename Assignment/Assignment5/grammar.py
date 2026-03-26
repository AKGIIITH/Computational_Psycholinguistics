"""
Grammar Loading Module
Loads probabilistic context-free grammar (PCFG) from file.

Format:
    probability   LHS   sym1  [sym2 ...]

Example:
    1.0  S  NP  VP
    0.4  N  time
"""

import math
from collections import defaultdict


def load_grammar(path):
    """
    Read a grammar file and return four objects:

    Args:
        path: Path to grammar file

    Returns:
        rules    : list of (log_prob, lhs, rhs_tuple)
        by_lhs   : dict  lhs  ->  [rule_idx, ...]
        nts      : set of non-terminal symbols
        rule_map : dict  (lhs, rhs_tuple)  ->  log_prob
    """
    rules    = []
    by_lhs   = defaultdict(list)
    nts      = set()
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
                lp = math.log(float(parts[0]))   # log probability
            except ValueError:
                continue

            lhs = parts[1]
            rhs = tuple(parts[2:])

            idx = len(rules)
            by_lhs[lhs].append(idx)
            nts.add(lhs)
            rule_map[(lhs, rhs)] = lp
            rules.append((lp, lhs, rhs))

    return rules, by_lhs, nts, rule_map