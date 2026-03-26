"""
Probabilistic Earley Parser
Usage:  python parse.py grammar.gr sentences.sen

Main entry point for the parser. Coordinates grammar loading, parsing, and output.
"""

import sys
from grammar import load_grammar
from earley import earley
from trees import all_trees, tree_logprob, tree_to_str
from output import print_chart


def main():
    if len(sys.argv) != 3:
        print(__doc__)
        sys.exit("Error: expected exactly two arguments: grammar.gr  sentences.sen")

    grammar_file   = sys.argv[1]
    sentences_file = sys.argv[2]

    rules, by_lhs, nts, rule_map = load_grammar(grammar_file)

    # Read all non-blank sentences
    with open(sentences_file) as fh:
        sentences = [line.split() for line in fh if line.strip()]

    for words in sentences:
        print(f"  Sentence: {' '.join(words)}")

        chart = earley(words, rules, by_lhs, nts)

        # Print chart
        print_chart(chart, rules, words)

        # Find complete S items spanning the whole sentence
        n = len(words)
        complete_s = [
            (ridx, len(rules[ridx][2]), 0)
            for ridx in by_lhs.get('S', [])
            if (ridx, len(rules[ridx][2]), 0) in chart[n]
        ]

        if not complete_s:
            print("\n  ✗  Sentence NOT accepted by this grammar.\n")
            continue

        # Enumerate and print all parse trees
        print()
        print("=" * 72)
        print("  PARSE TREES")
        print("=" * 72)

        seen  = set()
        count = 0

        for sk in complete_s:
            for tree in all_trees(sk, n, chart, rules, words):
                ts = tree_to_str(tree)
                if ts in seen:
                    continue
                seen.add(ts)
                count += 1

                lp   = tree_logprob(tree, rule_map)
                prob = 2.71828 ** lp if lp > float('-inf') else 0.0

                print(f"\n  ── Tree {count} ────────────────────────────────────")
                print(f"  Probability : {prob:.10f}")
                print(f"  Log-prob    : {lp:.6f}")
                print()
                for line in ts.splitlines():
                    print(f"    {line}")

        print()
        print(f"  Total: {count} parse tree(s) found.")
        print()


if __name__ == '__main__':
    main()