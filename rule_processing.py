from copy import deepcopy
from typing import List, Dict, Tuple
from my_types import ProposedRule, expressions_equivalent

#Makes this way easier to read
domain = List[Tuple[int, int]]
Conditions = Dict[str, domain]

"""=== Identical Rule Merging ==="""

"""Merges proposed initial rules if they are 'identical'. Rules are identical if they
possess an identical expression"""
def conduct_identical_merge(rules: List[ProposedRule]) -> List[ProposedRule]:
    changed = True
    current = rules

    while changed:
        changed = False
        output: List[ProposedRule] = []

        for rule in rules:
            merged = False

            for item in output:
                if expressions_equivalent(rule.expression, item.expression):
                    item.conditions = merge_conditions(item.conditions, rule.conditions)
                    item.contributing_vars |= rule.contributing_vars
                    merged = True
                    changed = True
                    break

            if not merged:
                output.append(deepcopy(rule))

        return output

def _determine_varied_range(rule: ProposedRule, var_domains: dict[str, tuple[int, int]]):
    merged = dict(rule.conditions)

    v = rule.varied_var
    if isinstance(v, str) and v not in merged:
        low, high = var_domains[v]
        merged[v] = [(low, high)]

    return merged

def merge_conditions(
    condition_a: Conditions,
    condition_b: Conditions,
) -> Conditions:
    output: Conditions = {}
    all_vars = condition_a.keys() | condition_b.keys()

    for variable in all_vars:
        combined_ranges = []
        if variable in condition_a:
            combined_ranges.extend(condition_a[variable])
        if variable in condition_b:
            combined_ranges.extend(condition_b[variable])

        output[variable] = _merge_intervals(combined_ranges)

    return output

def _merge_intervals(intervals: list[tuple[int, int]]) -> list[tuple[int, int]]:
    intervals = sorted(intervals, key=lambda x: x[0])
    saved = []
    current = intervals[0]

    for new_item in intervals[1:]:
        if new_item[0] <= current[1] + 1:
            current = (current[0], max(current[1], new_item[1]))
        else:
            saved.append(current)
            current = new_item

    saved.append(current)
    return saved

# def _determine_domain_compatibility(first_range: domain, second_range: domain) -> bool:
#     for first_start, first_end in first_range:
#         for second_start, second_end in second_range:
#             if second_start <= first_end + 1 and first_start <= second_end + 1:
#                 return True
#     return False
#
# def _determine_conditional_compatibility(condition1: Conditions, condition2: Conditions) -> bool:
#     for variable in condition1.keys() & condition2.keys():
#         compatible = _determine_domain_compatibility(condition1[variable], condition2[variable])
#         if not compatible:
#             return False
#     return True