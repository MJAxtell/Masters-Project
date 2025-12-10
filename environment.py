import random
from dataclasses import dataclass
from typing import List, Callable, Set

import my_types as mt

MAX_DEPTH = 2
MIN_CLAUSES = 1
MAX_CLAUSES = 3

@dataclass
class Rule:
    condition: mt.Conditional
    rhs_expr: mt.Expression

class Environment:
    def __init__(self, names_domains: dict[str, tuple[int, int]], rules: List[Rule] | None = None, seed: int | None = None):
        #List of Variable objects populated according to name_domains
        self.env_variables: List[mt.Variable] = [
            mt.Variable(name, min_val, max_val)
            for name, (min_val, max_val) in names_domains.items()
        ]

        #If not handed explicit rules, populates environment with a single random rule
        if rules is None:
            rand = random.Random(seed)
            lhs_cond = self.random_conditional(min_clauses=MIN_CLAUSES, max_clauses=MAX_CLAUSES, rand=rand)
            rhs_expr = self.random_expression(max_depth=MAX_DEPTH, rand=rand)
            self.rules: List[Rule] = [Rule(lhs_cond, rhs_expr)]
        else:
            self.rules = rules

    """Random Comparison Generation V1"""
    def random_comparison(self, rand: random.Random) -> mt.Conditional:
        var = rand.choice(self.env_variables)
        choice = rand.choice(["==", "!=", "<", "<=", ">", ">="])

        #Magic number 50% chance to compare between variables vs constants
        if rand.random() < 0.5:
            value = rand.randint(var.min_val, var.max_val)
        else:
            other_variables = [v for v in self.env_variables if v is not var]
            #Fallback to constant if env_variables is only len() == 1
            if not other_variables:
                value = rand.randint(var.min_val, var.max_val)
            else:
                value = rand.choice(other_variables)

        return mt.Comp(var=var, op=choice, value=value)

    def random_conditional(self, min_clauses: int, max_clauses: int, rand: random.Random) -> mt.Conditional:
        if min_clauses > max_clauses: raise RuntimeError("Minimum items greater than maximum items")
        if min_clauses <= 0: raise RuntimeError("Minimum items less than or equal to 0")
        if len(self.env_variables) < max_clauses: raise RuntimeError("Max items greater than number of env variables")

        num_clauses = rand.randint(min_clauses, max_clauses)
        condition: mt.Conditional = self.random_comparison(rand)

        for _ in range(1, num_clauses):
            next_clause = self.random_comparison(rand)
            connector = rand.choice(["AndCond", "OrCond"])
            if connector == "AndCond":
                condition = mt.AndCond(left=condition, right=next_clause)
            else:
                condition = mt.OrCond(left=condition, right=next_clause)

        return condition

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
                    exponent = rand.randint(2,3)
                )

            raise RuntimeError(f"Operator Unknown{operator}")

        return inner(depth=max_depth, pow_allowed=True)

    #Iterates through the rules and return 0 by default
    def environment_runner(self, guess: mt.Guess) -> int:
        for rule in self.rules:
            if mt.evaluate_conditional(rule.condition, guess):
                return mt.evaluate_expression(rule.rhs_expr, guess)
        return 0

    #Debugging check that guess contains exactly the environment variables + runs guess through environment
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