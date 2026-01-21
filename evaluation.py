import my_types as mt
from typing import List
import random
import printer as printer
from enum import Enum, auto

N_SAMPLES = 1000 #Number of samples for validation

#Behaviour influences the result of a observation not matching exactly with a proposed rule
class EvaluationEnum(Enum):
    EXPLICIT = auto()
    EXTRAPOLATE = auto()

def evaluate(training_results):
    explicit_accuracies = []
    extrapolated_accuracies = []
    inclusive_uniform_accuracies = []
    inclusive_proportional_accuracies = []
    environment_satisfactions = [] #0-1
    rule_matches = [] #0-1
    environment_rules = []

    for i, training_result in enumerate(training_results):
        print(f"=== Testing cycle {i} ===")
        environment = training_result.environment
        rules = training_result.rules

        observations = generate_random_observations(environment, N_SAMPLES)
        inclusive_uniform_observations = generate_inclusive_uniform(environment, rules, N_SAMPLES//len(environment.env_variables))
        inclusive_proportional_observations = generate_inclusive_proportional(environment, rules, N_SAMPLES)

        ground_truths = [observation.output for observation in observations]
        explicit_predicted = evaluate_rules(rules, observations, EvaluationEnum.EXPLICIT)
        extrapolated_predicted = evaluate_rules(rules, observations, EvaluationEnum.EXTRAPOLATE)

        inclusive_uniform_predicted = evaluate_rules(rules, inclusive_uniform_observations, EvaluationEnum.EXPLICIT)
        inclusive_uniform_ground_truths = [observation.output for observation in inclusive_uniform_observations]

        inclusive_proportional_predicted = evaluate_rules(rules, inclusive_proportional_observations, EvaluationEnum.EXPLICIT)
        inclusive_proportional_ground_truths = [observation.output for observation in inclusive_proportional_observations]

        explicit_accuracy = compute_accuracy(explicit_predicted, ground_truths)
        explicit_accuracies.append(explicit_accuracy)
        print(f"Explicit accuracy: {explicit_accuracy * 100:.3f}%")

        extrapolated_accuracy = compute_accuracy(extrapolated_predicted, ground_truths)
        extrapolated_accuracies.append(extrapolated_accuracy)
        print(f"Extrapolated accuracy: {extrapolated_accuracy * 100:.3f}%")

        inclusive_uniform_accuracy = compute_accuracy(inclusive_uniform_predicted, inclusive_uniform_ground_truths)
        inclusive_uniform_accuracies.append(inclusive_uniform_accuracy)
        print(f"Inclusive, uniform accuracy: {inclusive_uniform_accuracy * 100:.3f}%")

        inclusive_proportional_accuracy = compute_accuracy(inclusive_proportional_predicted, inclusive_proportional_ground_truths)
        inclusive_proportional_accuracies.append(inclusive_proportional_accuracy)
        print(f"Inclusive, proportional accuracy: {inclusive_proportional_accuracy * 100:.3f}%")

        num_agent_rules, num_env_rules, num_env_satisfied, matched_agent_count = algebraic_equivalence_metrics(environment, rules)
        print(f"Proposed rules: {num_agent_rules}")
        print(f"Environmental rules: {num_env_rules}")
        print(f"Environmental rules satisfied: {num_env_satisfied} / {num_env_rules}")
        print(f"Proposed rules matched: {matched_agent_count} / {num_agent_rules}")
        environment_satisfactions.append(num_env_satisfied / num_env_rules)
        rule_matches.append(matched_agent_count / num_agent_rules)
        environment_rules.append(num_env_rules)

        print("")

    # Please refer to begin_cycle within printer.py
    print("===🖥️ TOTAL SUMMARY 🖥️===")
    print(f"Mean explicit accuracy: {sum(explicit_accuracies)/len(extrapolated_accuracies) * 100:.6f}%")
    print(f"Mean extrapolated accuracy: {sum(extrapolated_accuracies)/len(extrapolated_accuracies) * 100:.6f}%")
    print(f"Mean inclusive, uniform accuracy: {sum(inclusive_uniform_accuracies)/len(inclusive_uniform_accuracies) * 100:.6f}%")
    print(f"Mean inclusive, proportional accuracy: {sum(inclusive_proportional_accuracies)/len(inclusive_proportional_accuracies) * 100:.6f}%")
    print(f"Mean environment variable satisfactions: {sum(environment_satisfactions) / len(environment_satisfactions) * 100:.6f}%")
    print(f"Mean proposed rule matches: {sum(rule_matches) / len(rule_matches) * 100:.6f}%")
    print(f"Mean environmental rules: {sum(environment_rules) / len(environment_rules):.6f}")

def compute_accuracy(predicted, ground_truths) -> float:
    correct = 0
    for predict, ground_truth in zip(predicted, ground_truths):
        if predict == ground_truth:
            correct += 1

    return correct / len(ground_truths)

"""Generate observations evenly across each rule inclusive only to the proposed rules.
Adapted from agent.py, conduct controlled."""
def generate_inclusive_uniform(environment, rules: list[mt.ProposedRule], samples_per_rule: int):
    observations: List[mt.Observation] = []

    for rule in rules:
        for _ in range(samples_per_rule):
            values = {}

            for variable in environment.env_variables:
                name = variable.name
                if name in rule.conditions:
                    values[name] = _samples_from_ranges(rule.conditions[name])
                else:
                    values[name] = random.randint(variable.min_val, variable.max_val)

            guess = mt.Guess(values=values)
            output = environment.evaluate(guess)
            observations.append(
                mt.Observation(inputs=values, output=output)
            )
    return observations

"""Generates observations depending on the size of the rule and inclusive only to the proposed rules.
Adapted from agent.py, conduct controlled."""
def generate_inclusive_proportional(environment, rules: list[mt.ProposedRule], n_samples: int):
    observations: list[mt.Observation] = []

    if not rules:
        return observations

    for _ in range(n_samples):
        rule = random.choice(rules)
        values = {}

        for variable in environment.env_variables:
            name = variable.name

            if name in rule.conditions:
                values[name] = _samples_from_ranges(rule.conditions[name])
            else:
                values[name] = random.randint(variable.min_val, variable.max_val)

        guess = mt.Guess(values=values)
        output = environment.evaluate(guess)

        observations.append(
            mt.Observation(inputs=values, output=output)
        )

    return observations

"""A0 - Generates random observations under an environment"""
def generate_random_observations(environment, n_samples) -> List[mt.Observation]:
    observations: List[mt.Observation] = []

    for _ in range(n_samples):
        values = {}
        for var in environment.env_variables:
            values[var.name] = random.randint(var.min_val, var.max_val)

        guess = mt.Guess(values=values)
        output = environment.evaluate(guess)

        observations.append(
            mt.Observation(inputs=values, output=output)
        )

    return observations

"""A1 - Takes the list of generated rules and the samples and returns a list of outputs to be compared against ground truths"""
def evaluate_rules(rules: List[mt.ProposedRule], observations: list[mt.Observation], mode: EvaluationEnum) -> list[int]:
    outputs = []

    for observation in observations:
        output = _evaluate_rules_on_inputs(rules, observation.inputs, mode)
        outputs.append(output)

    return outputs

"""Determines if a rule applies to a sample. - Used in A2"""
def _determine_rule_applies(rule: mt.ProposedRule, inputs: dict[str, int]) -> bool:
    for var, ranges in rule.conditions.items():
        value = inputs[var]
        if not any(low <= value <= high for low, high in ranges):
            return False
    return True

"""Helper to break down ProposedRule lists."""
def _any_rule_applies(rules: list[mt.ProposedRule], inputs: dict[str, int]) -> bool:
    for rule in rules:
        if _determine_rule_applies(rule, inputs):
            return True
    return False

"""A2 - Takes the list of generated rules and the sample and determines if a rule applied. - Used in A1
#Mode determines whether to default for unmatched rules or to extrapolate"""
def _evaluate_rules_on_inputs(rules: list[mt.ProposedRule], inputs: dict[str, int], mode: EvaluationEnum) -> int:
    guess = mt.Guess(values=inputs)
    for rule in rules:
        if _determine_rule_applies(rule, inputs):
            return mt.evaluate_expression(rule.expression, guess)

    """Default for now if no rule applies and mode is EXPLICIT, return -1"""
    if mode == EvaluationEnum.EXPLICIT:
        return -1
    elif mode == EvaluationEnum.EXTRAPOLATE:
        rule = _determine_extrapolate_rule(rules, inputs)
        if rule is None:
            return -1
        return mt.evaluate_expression(rule.expression, guess)

    raise ValueError("Unhandled observation!")

"""A3 - Changes rule variable domains to a range between their lowest-highest variable values. - Used in A1"""
def _bound_rules(rule: mt.ProposedRule) -> dict[str, tuple[int, int]]:
    filled_bounds = {}

    for var, ranges in rule.conditions.items():
        low = min(ran[0] for ran in ranges)
        high = max(ran[1] for ran in ranges)
        filled_bounds[var] = (low, high)

    return filled_bounds

"""A4 - Determines if an observation is within some bounds. - Used in A6"""
def _determine_within_bounds(bounds: dict[str, tuple[int, int]], inputs: dict[str, int]) -> bool:
    for var, (low, high) in bounds.items():
        value = inputs[var]
        if value < low or value > high:
            return False
    return True

"""A5 - Computes rule size by its bounds, the sum of the length of its boundaries/domains for each variable. - Used in A6
Rules of the smallest size are selected in EXTRAPOLATION mode when multiple rules apply."""
def _determine_rule_size(bounds: dict[str, tuple[int, int]]) -> int:
    size = 0
    for low, high in bounds.values():
        size += (high - low + 1)
    return size

"""A6 - Expands the domains/boundaries of each rule and determines if the rule applies to the input.
For each applicable rule, chooses the smallest one, or the first in a tie. If no rule applies, triggers
the default case of -1. - Used in A2"""
def _determine_extrapolate_rule(rules: list[mt.ProposedRule], inputs: dict[str, int]) -> mt.ProposedRule | None:
    candidates = []

    for rule in rules:
        bounds = _bound_rules(rule)

        if _determine_within_bounds(bounds, inputs):
            size = _determine_rule_size(bounds)
            candidates.append((size, rule))

    #None triggers the default case in A2
    if not candidates:
        return None

    #Sort candidates by size. Chooses the top sorted candidate in a tie
    candidates.sort(key=lambda x: x[0])
    return candidates[0][1]

"""As from validation, samples within a range."""
def _samples_from_ranges(ranges: List[tuple[int, int]]) -> int:
    total = sum(end - start + 1 for start, end in ranges)
    r = random.randint(1, total)

    for start, end in ranges:
        width = end - start + 1
        if r <= width:
            return start + r - 1
        r -= width

"""Polynomial normalization and algebraic equivalnce functions.
The following functions were creating with assistance from an AI (ChatGPT).
All functions are hardwritten and checked. AI was used to assist with all
parts of the process."""

"""B1
Checks, for every agent rule, if at least one environmental rule is equivalent.
Returns a metrics tuple."""
def algebraic_equivalence_metrics(environment, agent_rules):
    """
    Computes algebraic equivalence metrics between agent and environment RHS rules.

    Prints:
    - number of proposed (agent) rules
    - number of environmental rules
    - number of environmental rules satisfied (unique env matches)
    - number of proposed rules that match (agent-side coverage)

    Last two are printed as X / Y.
    """

    # Canonicalize environment RHS expressions
    env_polys = [
        rhs_to_poly(rule.rhs_expr)
        for rule in environment.rules
    ]

    # Canonicalize agent RHS expressions
    agent_polys = [
        rhs_to_poly(rule.expression)
        for rule in agent_rules
    ]

    # Track matches
    matched_env_indices = set()
    matched_agent_count = 0

    for agent_poly in agent_polys:
        matches = [
            i for i, env_poly in enumerate(env_polys)
            if agent_poly == env_poly
        ]

        if matches:
            matched_agent_count += 1
            matched_env_indices.update(matches)

    num_env_rules = len(env_polys)
    num_agent_rules = len(agent_polys)
    num_env_satisfied = len(matched_env_indices)

    return (
        num_agent_rules,
        num_env_rules,
        num_env_satisfied,
        matched_agent_count,
    )
"""B2
"""
def _merge_powers(a: dict[str, int], b: dict[str, int]) -> dict[str, int]:
    output = dict(a)
    for var, expression in b.items():
        output[var] = output.get(var, 0) + expression
    return output

"""B3
"""
def _normalize_expression(expression):
    if isinstance(expression, mt.Const):
        return [(expression.value, {})]
    if isinstance(expression, mt.VariableReference):
        return [(1, {expression.var.name: 1})]
    if isinstance(expression, mt.Add):
        return _normalize_expression(expression.left) + _normalize_expression(expression.right)
    if isinstance(expression, mt.Mul):
        output = []
        left_terms = _normalize_expression(expression.left)
        right_terms = _normalize_expression(expression.right)

        for ca, pa in left_terms:
            for cb, pb in right_terms:
                output.append((ca * cb, _merge_powers(pa, pb)))
        return output
    if isinstance(expression, mt.Pow):
        if expression.exponent == 0:
            return [(1, {})]

        base_terms = _normalize_expression(expression.base)
        output = [(1, {})]

        for _ in range(expression.exponent - 1):
            output = [
                (ca * cb, _merge_powers(pa, pb))
                for ca, pa in output
                for cb, pb in base_terms
            ]
        return output

"""B4
"""
def _canonicalize(terms):
    combined = {}

    for coefficient, powers in terms:
        if coefficient == 0:
            continue

        key = tuple(sorted(powers.items()))
        combined[key] = combined.get(key, 0) + coefficient

    return tuple(sorted(
        (key, coefficient)
        for key, coefficient in combined.items()
        if coefficient != 0
    ))

def rhs_to_poly(expr):
    return _canonicalize(_normalize_expression(expr))

def pretty_polynomial(poly):
    if not poly:
        return "0"

    parts = []
    for powers, coeff in poly:
        if powers:
            vars_part = []
            for var, exp in powers:
                vars_part.append(var if exp == 1 else f"{var}^{exp}")
            term = " * ".join(vars_part)
            if coeff != 1:
                term = f"{coeff} * {term}"
        else:
            term = str(coeff)
        parts.append(term)

    return " + ".join(parts)