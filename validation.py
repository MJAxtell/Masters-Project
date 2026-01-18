from collections import defaultdict
from typing import List, Dict, Tuple
from my_types import ProposedRule, Guess, evaluate_expression, expressions_equivalent, BehaviourSignature
from environment import Environment
from itertools import combinations
from rule_processing import merge_conditions
from random import random
from copy import deepcopy
from printer import MainPrinter

import random

class Validator:
    def __init__(self, environment: Environment):
        self.environment = environment
        self.printer = MainPrinter()

    #Note slightly different signature - packs both rules and stats for printing into a list
    def validate_initials(self, rules: List[ProposedRule], validation_samples: int, validation_threshold: float) -> List[List[ProposedRule]]:
        output: List[ProposedRule] = []
        stats = []

        for rule in rules:
            successful = 0

            for _ in range(validation_samples):
                guess = self._conduct_rule_guess(rule)
                truth = self.environment.evaluate(guess)
                obtained = evaluate_expression(rule.expression, guess)

                if obtained == truth:
                    successful += 1

            accuracy = successful / validation_samples
            success = accuracy >= validation_threshold
            stats.append((success, accuracy))

            if success:
                output.append(rule)

        return [output, stats]

    def _conduct_rule_guess(self, rule: ProposedRule) -> Guess:
        values: Dict[str, int] = {}

        for var in self.environment.env_variables:
            name = var.name
            if name in rule.conditions:
                low, high = random.choice(rule.conditions[name])
                values[name] = random.randint(low, high)
            else:
                values[name] = random.randint(var.min_val, var.max_val)

        return Guess(values=values)

    """Global rule merging and validation"""
    def conduct_rule_merges(self, rules: List[ProposedRule], n_samples: int, validation_threshold: float) -> List[ProposedRule]:
        output = []
        used: set[int] = set()
        pairs = self._determine_potential_pairs(rules)
        for rulea, ruleb in pairs:
            #Skip rules included in a pair
            if id(rulea) in used or id(ruleb) in used:
                continue

            merged_conditional = merge_conditions(rulea.conditions, ruleb.conditions)
            a_samples = self.conditional_samples(rulea, n_samples)
            b_samples = self.conditional_samples(ruleb, n_samples)
            correct_a, correct_b = self.determine_correct(a_samples, b_samples, rulea.expression, ruleb.expression)

            a_accuracy = correct_a / len(a_samples)
            b_accuracy = correct_b / len(b_samples)

            """To win, a rule must:
            Exceed the accuracy of the other, meet or exceed the validation threshold.
            If no rule succeeds, both singular rules are added back to the list."""
            #Rule A is the winner:
            if a_accuracy > b_accuracy and a_accuracy >= validation_threshold:
                new_rule = deepcopy(rulea)
                new_rule.conditions = merged_conditional
                new_rule.contributing_vars |= ruleb.contributing_vars
                output.append(new_rule)
                used |= {id(rulea), id(ruleb)}

            #Rule B is the winner:
            elif b_accuracy > a_accuracy and b_accuracy >= validation_threshold:
                new_rule = deepcopy(ruleb)
                new_rule.conditions = merged_conditional
                new_rule.contributing_vars |= rulea.contributing_vars
                output.append(new_rule)
                used |= {id(rulea), id(ruleb)}

        #Remember to add rules that never got paired!
        for rule in rules:
            if id(rule) not in used:
                output.append(rule)

        return output

    def conditional_samples(self, rule: ProposedRule, n_samples: int) -> List[Dict[str, int]]:
        samples: List[Dict[str, int]] = []

        for _ in range(n_samples):
            values: Dict[str, int] = {}

            for var in self.environment.env_variables:
                name = var.name

                if name in rule.conditions:
                    values[name] = self._samples_from_ranges(rule.conditions[name])
                else:
                    values[name] = random.randint(var.min_val, var.max_val)
            samples.append(values)
        return samples

    #Does not yet compute accuracy, only number of right
    def determine_correct(self, samples_a, samples_b, expression_a, expression_b):
        correct_a = 0
        correct_b = 0

        for sample in samples_a:
            guess = Guess(values=sample)

            truth = self.environment.evaluate(guess)
            evaluated = evaluate_expression(expression_b, guess)
            if truth == evaluated:
                correct_b += 1

        for sample in samples_b:
            guess = Guess(values=sample)

            truth = self.environment.evaluate(guess)
            evaluated = evaluate_expression(expression_a, guess)
            if truth == evaluated:
                correct_a += 1

        return correct_a, correct_b

    def _determine_potential_pairs(self, rules: List[ProposedRule],) -> List[Tuple[ProposedRule, ProposedRule]]:
        signature_dict: dict[BehaviourSignature, list[ProposedRule]] = defaultdict(list)
        for rule in rules:
            signature_dict[rule.signature].append(rule)

        potential_pairs: list[tuple[ProposedRule, ProposedRule]] = []

        for group in signature_dict.values():
            if len(group) < 2:
                continue
            #Itertools!
            potential_pairs.extend(combinations(group, 2))

        return potential_pairs

    def _samples_from_ranges(self, ranges: List[tuple[int, int]]) -> int:
        total = sum(end - start + 1 for start, end in ranges)
        r = random.randint(1, total)

        for start, end in ranges:
            width = end - start + 1
            if r <= width:
                return start + r - 1
            r -= width
