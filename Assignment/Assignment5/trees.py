"""
Parse Tree Module
Tree enumeration, probability computation, and pretty-printing.
"""


def all_trees(key, col, chart, rules, words):
    """
    Generator: yield every distinct parse tree for item `key` in chart[col].

    Each tree is a nested tuple (lhs, [child, ...]) where leaves are
    plain strings (terminal words).

    Args:
        key: Earley item key (ridx, dot, origin)
        col: Chart column index
        chart: Earley chart
        rules: Grammar rules
        words: Input sentence

    Yields:
        Parse trees as nested tuples
    """
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
            for partial in all_trees(prev_key, col - 1, chart, rules, words):
                yield (partial[0], partial[1] + [terminal])

        elif kind == 'complete':
            _, left_key, right_key, mid = back
            for partial in all_trees(left_key, mid, chart, rules, words):
                for sub in all_trees(right_key, col, chart, rules, words):
                    yield (partial[0], partial[1] + [sub])


def tree_logprob(tree, rule_map):
    """
    Recursively compute log probability of a parse tree.

    Args:
        tree: Parse tree tuple
        rule_map: Rule probability map from load_grammar()

    Returns:
        Log probability of the tree
    """
    lhs, children = tree
    if not children:
        return 0.0

    child_syms = tuple(c if isinstance(c, str) else c[0] for c in children)
    lp = rule_map.get((lhs, child_syms), float('-inf'))

    for child in children:
        if isinstance(child, tuple):
            lp += tree_logprob(child, rule_map)

    return lp


def tree_to_str(tree, indent=0):
    """
    Pretty-print a parse tree as bracketed string.

    Example:
        (S
          (NP (N time))
          (VP (V flies)))

    Args:
        tree: Parse tree tuple
        indent: Current indentation level

    Returns:
        Formatted tree string
    """
    lhs, children = tree
    if not children:
        return lhs

    if all(isinstance(c, str) for c in children):
        return f"({lhs} {' '.join(children)})"

    pad = ' ' * (indent + 2)
    parts = []
    for c in children:
        if isinstance(c, str):
            parts.append(pad + c)
        else:
            parts.append(pad + tree_to_str(c, indent + 2))
    inner = '\n'.join(parts)
    return f"({lhs}\n{inner})"