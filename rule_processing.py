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
    output: List[ProposedRule] = []

    for rule in rules:
        merged = False

        for item in output:
            if expressions_equivalent(rule.expression, item.expression):
                item.conditions = _merge_conditions(item.conditions, rule.conditions)
                merged = True
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

def _determine_domain_compatibility(first_range: domain, second_range: domain) -> bool:
    for first_start, first_end in first_range:
        for second_start, second_end in second_range:
            if second_start <= first_end + 1 and first_start <= second_end + 1:
                return True
    return False

def _determine_conditional_compatibility(condition1: Conditions, condition2: Conditions) -> bool:
    for variable in condition1.keys() & condition2.keys():
        compatible = _determine_domain_compatibility(condition1[variable], condition2[variable])
        if not compatible:
            return False
    return True

def _merge_ranges(domains: domain) -> domain:
    if not domain:
        return []

    domains = sorted(domains)
    merged_domains = [domains[0]]

    for first_start, first_end in domains[1:]:
        last_start, last_end = merged_domains[-1]
        if first_start <= last_end + 1:
            merged_domains[-1] = (last_start, max(last_end, first_end))
        else:
            merged_domains.append((first_start, first_end))

    return merged_domains

def _merge_conditions(condition1: Conditions, condition2: Conditions) -> Conditions:
    merged = {}

    #Union of conditional variables (keys)
    all_variables = condition1.keys() | condition2.keys()
    for variable in all_variables:
        ranges = []
        #Extend syntax avoids for loops
        if variable in condition1:
            ranges.extend(condition1[variable])
        if variable in condition2:
            ranges.extend(condition2[variable])

        merged[variable] = _merge_ranges(ranges)

    return merged