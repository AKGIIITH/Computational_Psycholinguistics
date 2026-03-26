"""
Earley Algorithm Module
Core parsing algorithm with PREDICT, SCAN, COMPLETE operations.

Complexity:
    Space:  O(n²)
    Time:   O(n³)
"""

from collections import defaultdict


def earley(words, rules, by_lhs, nts, start='S'):
    """
    Run the probabilistic Earley algorithm on `words`.

    An Earley item is a triple (rule_idx, dot, origin) meaning:
        dot    = how far we have matched into RHS (0 … len(RHS))
        origin = input position where this rule application began

    Args:
        words: List of input tokens
        rules: Grammar rules from load_grammar()
        by_lhs: LHS index from load_grammar()
        nts: Non-terminals set from load_grammar()
        start: Start symbol (default 'S')

    Returns:
        chart: Dictionary of Earley items indexed by position
    """
    n      = len(words)
    chart  = [dict() for _ in range(n + 1)]
    agenda = [[]     for _ in range(n + 1)]

    def add(col, ridx, dot, origin, lw, back=None):
        """Add item to chart with log-weight and backpointer."""
        key = (ridx, dot, origin)
        if key not in chart[col]:
            chart[col][key] = {
                'w':     lw,
                'backs': [] if back is None else [back]
            }
            agenda[col].append(key)
        else:
            entry = chart[col][key]
            if lw > entry['w']:
                entry['w'] = lw
            if back is not None and back not in entry['backs']:
                entry['backs'].append(back)

    # Seed: predict every S-rule at position 0
    for ridx in by_lhs.get(start, []):
        add(0, ridx, 0, 0, rules[ridx][0])

    for j in range(n + 1):
        ptr = 0
        while ptr < len(agenda[j]):
            key = agenda[j][ptr]
            ptr += 1

            ridx, dot, origin = key
            lp_rule, lhs, rhs = rules[ridx]
            lw = chart[j][key]['w']

            # COMPLETE: item is fully matched
            if dot == len(rhs):
                for wkey, we in list(chart[origin].items()):
                    wridx, wdot, worigin = wkey
                    _, _, wrhs = rules[wridx]
                    if wdot < len(wrhs) and wrhs[wdot] == lhs:
                        new_lw   = we['w'] + lw
                        new_back = ('complete', wkey, key, origin)
                        add(j, wridx, wdot + 1, worigin, new_lw, new_back)

            else:
                sym = rhs[dot]

                # PREDICT: next symbol is non-terminal
                if sym in nts:
                    for ridx2 in by_lhs[sym]:
                        add(j, ridx2, 0, j, rules[ridx2][0])

                # SCAN: next symbol matches input word
                elif j < n and words[j] == sym:
                    add(j + 1, ridx, dot + 1, origin, lw, ('scan', key))

    return chart