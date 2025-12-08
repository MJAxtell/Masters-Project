import random
from dataclasses import dataclass
from typing import List, Callable, Set

import my_types as mt

#Rule class, lhs conditional and rhs function
@dataclass
class Rule:
    condition: Callable[[mt.Guess], bool]
    rhs_expr: mt.Expression

#Takes a dictionary of names/min-max domain limits, list of rules
class Environment:
    def __init__(self, names_domains: dict[str, tuple[int, int]], rules: List[Rule] | None = None):
        #List of Variable objects populated according to name_domains
        self.env_variables: List[mt.Variable] = [
            mt.Variable(name, min_val, max_val)
            for name, (min_val, max_val) in names_domains.items()
        ]

        if rules is None:
            self.rules: List[Rule] = self._hardcoded_rule_builder()
        else:
            self.rules: List[Rule] = rules

    # == Helpers ==
    def _var_from_name(self, name: str) -> mt.Variable:
        for var in self.env_variables:
            if var.name == name:
                return var
        raise KeyError(f"Provided variable name not found: {name}")

    def _hardcoded_rule_builder(self) -> List[Rule]:
        #Temp references to environment's own variables
        a_var = self._var_from_name("A")
        b_var = self._var_from_name("B")
        c_var = self._var_from_name("C")

        #Conditional function: A >= 5
        def hard_condition(guess: mt.Guess) -> bool:
            return guess[a_var.name] >= 5

        #Function component: B + C
        rhs_expr: mt.Expression = mt.Add(
            left=mt.VariableReference(var=b_var),
            right=mt.VariableReference(var=c_var),
        )

        #Rule: Conditional - Function
        hard_rule = Rule(hard_condition, rhs_expr)

        return [hard_rule]

    """Iterates through the rules, the first one who's conditional is True when passed guess,
    evaluate the rhs_expr and return 0 by default"""
    def environment_runner(self, guess: mt.Guess) -> int:
        for rule in self.rules:
            if rule.condition(guess):
                return mt.evaluate_expression(rule.rhs_expr, guess)
        return 0

    #Debugging check that guess contains exactly the environment variables
    def evaluate(self, guess: mt.Guess) -> int:
        environments: Set[str] = {var.name for var in self.env_variables}
        guesses: Set[str] = set(guess.values.keys())

        if environments != guesses:
            missing = environments - guesses
            additional = guesses - environments
            if missing:
                raise KeyError(f"Guess has missed the following variables: {missing}")
            if additional:
                raise KeyError(f"Guess has the following extra variables: {additional}")

        return self.environment_runner(guess)