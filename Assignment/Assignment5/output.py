"""
Output Formatting Module
Chart printing and formatting utilities.
"""


def print_chart(chart, rules, words):
    """
    Print every column of the Earley chart.

    Columns indexed by input position. Shows:
        [done] [i,j]  LHS -> matched • remaining   logp=…

    Args:
        chart: Earley chart
        rules: Grammar rules
        words: Input sentence
    """
    n = len(words)
    print()
    print("=" * 72)
    print("  EARLEY CHART")
    print("=" * 72)

    for j in range(n + 1):
        if j == 0:
            label = "Chart[0]  (before reading any word)"
        else:
            label = f"Chart[{j}]  (after reading '{words[j-1]}')"
        print(f"\n  ── {label} {'─'*(50-len(label))}")

        if not chart[j]:
            print("     (empty)")
            continue

        for key in chart[j]:
            ridx, dot, origin = key
            _, lhs, rhs = rules[ridx]

            before   = ' '.join(rhs[:dot])
            after    = ' '.join(rhs[dot:])
            rule_str = (f"{lhs} -> {before} • {after}" if before
                        else f"{lhs} -> • {after}"     if after
                        else f"{lhs} -> {' '.join(rhs)} •")

            w       = chart[j][key]['w']
            done    = '✓' if dot == len(rhs) else ' '
            print(f"  {done}  [{origin},{j}]  {rule_str:<44}  logp={w:8.4f}")