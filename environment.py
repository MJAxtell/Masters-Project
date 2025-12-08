import random
from dataclasses import dataclass
from typing import List, Callable, Set

import my_types as mt

MAX_DEPTH = 2

#Rule class, lhs conditional and rhs function
@dataclass
class Rule:
    condition: Callable[[mt.Guess], bool]
    rhs_expr: mt.Expression

#Takes a dictionary of names/min-max domain limits, list of rules (if provided manually)
class Environment:
    def __init__(self, names_domains: dict[str, tuple[int, int]], rules: List[Rule] | None = None, seed: int | None = None):
        #List of Variable objects populated according to name_domains
        self.env_variables: List[mt.Variable] = [
            mt.Variable(name, min_val, max_val)
            for name, (min_val, max_val) in names_domains.items()
        ]

        """Populates rules with a single rule with a random rhs expression and always true conditional"""
        if rules is None:
            rand = random.Random(seed)

            def condition(guess: mt.Guess) -> bool:
                return True

            rhs_expr = self.random_expression(max_depth=MAX_DEPTH, rand=rand)

            self.rules: List[Rule] = [Rule(condition, rhs_expr)]

        else:
            self.rules = rules

        """Populates rules with old hardcoded rule"""
        # if rules is None:
        #     self.rules: List[Rule] = self._hardcoded_rule_builder()
        # else:
        #     self.rules: List[Rule] = rules

    """Random Expression Generation V1"""

    #Currently, no external hyperparameters, hardcoded magic numbers / reasonable limits
    #Generates a random leaf, either a variable or constant
    def random_leaf(self, rand: random.Random) -> mt.Expression:
        choice = rand.choice(["const", "var"])
        if choice == "const":
            return mt.Const(value=rand.randint(1,10))
        else:
            var = rand.choice(self.env_variables)
            return mt.VariableReference(var)

    #Builds a random rhs expression tree with custom max depth
    #Uses hardcoded magic numbers and reasonable limits
    def random_expression(self, max_depth: int, rand: random.Random) -> mt.Expression:
        def inner(depth: int, pow_allowed: bool) -> mt.Expression:
            if depth <= 0:
                return self.random_leaf(rand)

            if rand.random() < 0.25:
                return self.random_leaf(rand)

            if pow_allowed:
                operator = rand.choice(["Add", "Mul", "Pow"])
            else:
                operator = rand.choice(["Add", "Mul"])

            if operator == "Add":
                return mt.Add(
                    left = inner(depth - 1, pow_allowed),
                    right = inner(depth - 1, pow_allowed),
                )

            if operator == "Mul":
                return mt.Mul(
                    left = inner(depth - 1, pow_allowed),
                    right = inner(depth - 1, pow_allowed),
                )

            #Exponent is never nested, always 2-4
            if operator == "Pow":
                return mt.Pow(
                    base = inner(depth - 1, pow_allowed=False),
                    exponent = rand.randint(2,4)
                )

            raise RuntimeError(f"Operator Unknown{operator}")

        return inner(depth=max_depth, pow_allowed=True)

    # == Helpers ==
    # Redundant hardcoded singular function examples
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