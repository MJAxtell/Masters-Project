import random
from dataclasses import dataclass
from typing import List, Callable, Set, Tuple, Optional

import my_types as mt

""""Hyperparameters - edit note, should let main pass these as arguments on init"""

"""MIN_CONST and MAX_CONST values MUST match symreg.py MIN_EPH and MAX_EPH values respectively.
Project inductive bias is that we use integers only, no floats, 1-10.
You may arrive at larger integers through composition."""
MIN_CONST = 1 #Minimum value for constants - default 1
MAX_CONST = 10 #Maximum value for constants - default 10

"""LHS expression values"""
MAX_VARS = 5 #Maximum number of variables that may appear in a single rule's conditional

"""RHS expression values"""
MIN_POW = 2 #Minimum value for POW operator
MAX_POW = 2 #Maximum value for POW operators
MAX_DEPTH = 2 #Maximum branching depth for expressions
LEAF_BIAS = 0.8 #Strength of recursion preference for termination during depth descent, tune to MAX_DEPTH
ROOT_LEAF_PROB = 0.25 #Probability for root to be a leaf (bare variable or scalar)

"""Bias for internal advanced rule generation to add additional slices.
0 = one slice, 1 = maximum slices."""
SPLIT_BIAS = 0.5

"""Bias for internal advanced rule generation add more included variables.
0 = one variable, 1 = all variables."""
VAR_BIAS = 0.5

@dataclass
class Rule:
    condition: mt.Conditional
    rhs_expr: mt.Expression

"""Advanced function generation classes"""
@dataclass
class LeafRegion:
    ranges: dict[str, tuple[int, int]]

    def constrained_count(self) -> int:
        return len(self.ranges)

class Environment:
    def __init__(self,
                 names_domains: dict[str, tuple[int, int]],
                 rules: List[Rule] | None = None,
                 num_rules: int = 1,
                 seed: int | None = None):
        #List of Variable objects populated according to name_domains
        self.env_variables: List[mt.Variable] = [
            mt.Variable(name, min_val, max_val)
            for name, (min_val, max_val) in names_domains.items()
        ]

        #If not handed explicit rules, populates environment with a single random rule
        if rules is None:
            rand = random.Random(seed)
            self.rules: List[Rule] = self.random_split_rules(rand, num_rules)
        else:
            self.rules: List[Rule] = rules

    """Advanced Random Rule Generation V2.0
    Produces contiguous ranges for included variables."""
    def random_split_rules(self, rand: random.Random, num_rules: int) -> List[Rule]:
        leaves: List[LeafRegion] = [LeafRegion(ranges={}) for _ in range(num_rules)]

        """ ~ Variable selection ~ """
        chosen_vars = [var for var in self.env_variables if rand.random() < VAR_BIAS]
        if not chosen_vars:
            chosen_vars = [rand.choice(self.env_variables)]

        single_choice = (len(chosen_vars) == 1)

        for variable in chosen_vars:
            growable = [i for i, leaf in enumerate(leaves) if leaf.constrained_count() < MAX_VARS]

            if not growable:
                continue

            #If single choice, need one slice per variable
            if single_choice:
                num_splits = min(len(growable), num_rules)
            else:
                num_splits = self._num_slices_generator(len(growable), num_rules, rand)

            #Generate contiguous domain slices
            slices = self._slice_domain(low=variable.min_val, high=variable.max_val, num_splits=num_splits, rand=rand)

            #Assign a maximum of one slice to each rule
            chosen_rules = rand.sample(growable, num_splits)
            for i, (low, high) in zip(chosen_rules, slices):
                leaves[i].ranges[variable.name] = (low, high)

        return [Rule(self.region_to_condition(leaf), self.random_expression(rand)) for leaf in leaves]

    """Translates distinct domain values into conditional representation"""
    def region_to_condition(self, leaf: LeafRegion) -> mt.Conditional:
        conditional: Optional[mt.Conditional] = None

        for var in sorted(self.env_variables, key=lambda v: v.name):
            if var.name not in leaf.ranges:
                continue

            low, high = leaf.ranges[var.name]
            clause = mt.RangeCond(var=var, low=low, high=high)

            if conditional is None:
                conditional = clause
            else:
                conditional = mt.AndCond(left=conditional, right=clause)

        if conditional is None:
            v = self.env_variables[0]
            conditional = mt.RangeCond(var=v, low=v.min_val, high=v.max_val)

        return conditional

    """Advanced rule generation helpers"""
    #Determines number of slices
    def _num_slices_generator(self, max_slices: int, num_rules: int, rand: random.Random) -> int:
        true_max = min(max_slices, num_rules) #No more splits that one per rule or max slices
        num_slices = 1

        while num_slices < true_max and rand.random() < SPLIT_BIAS:
            num_slices += 1
        return num_slices

    #Slices a provided domain into num_splits contiguous intervals
    def _slice_domain(self, low:int, high:int, num_splits:int, rand: random.Random) -> List[tuple[int, int]]:
        intervals = []

        if num_splits <= 0:
            return []

        length = high - low + 1
        if num_splits > length: #This guard should never be hit! Too big!
            num_splits = length

        slices = sorted(rand.sample(range(low, high), num_splits - 1))
        start = low
        for slice_point in slices:
            intervals.append((start, slice_point))
            start = slice_point + 1

        intervals.append((start, high))
        return intervals

    """LHS Expression Generation Helpers"""
    def _is_constant(self, expression: mt.Expression) -> bool:
        match expression:
            case mt.Const():
                return True
            case mt.Variable():
                return False
            case mt.Add(left=left, right=right):
                return self._is_constant(left) and self._is_constant(right)
            case mt.Mul(left=left, right=right):
                return self._is_constant(left) and self._is_constant(right)
            case mt.Pow(base=base):
                return self._is_constant(base)
        return False

    def _eval_constant(self, expression: mt.Expression) -> int:
        match expression:
            case mt.Const(value=value):
                return value
            case mt.Add(left=left, right=right):
                return self._eval_constant(left) + self._eval_constant(right)
            case mt.Mul(left=left, right=right):
                return self._eval_constant(left) * self._eval_constant(right)
            case mt.Pow(base=base, exponent=exponent):
                return self._eval_constant(base) ** exponent

    """LHS Random Expression Generation V1"""
    #Generates a random leaf, either a variable or constant
    def random_leaf(self, rand: random.Random) -> mt.Expression:
        choice = rand.choice(["const", "var"])
        if choice == "const":
            return mt.Const(value=rand.randint(MIN_CONST, MAX_CONST))
        else:
            var = rand.choice(self.env_variables)
            return mt.VariableReference(var)

    #Builds a random rhs expression tree with custom max depth
    #Uses hardcoded magic numbers and reasonable limits - FIX
    def random_expression(self, rand: random.Random) -> mt.Expression:
        def inner(depth: int, pow_allowed: bool) -> mt.Expression:
            if depth <= 0:
                return self.random_leaf(rand)

            if rand.random() <= ROOT_LEAF_PROB:
                return self.random_leaf(rand)

            if rand.random() < self._stop_probability(depth):
                return self.random_leaf(rand)

            if pow_allowed:
                operator = rand.choice(["Add", "Mul", "Pow"])
            else:
                operator = rand.choice(["Add", "Mul"])

            if operator == "Add":
                expression = mt.Add(
                    left = inner(depth - 1, pow_allowed),
                    right = inner(depth - 1, pow_allowed),
                )
                return mt.Const(self._eval_constant(expression)) if self._is_constant(expression) else expression

            if operator == "Mul":
                expression = mt.Mul(
                    left = inner(depth - 1, pow_allowed),
                    right = inner(depth - 1, pow_allowed),
                )
                return mt.Const(self._eval_constant(expression)) if self._is_constant(expression) else expression

            #Exponent is never nested
            if operator == "Pow":
                expression = mt.Pow(
                    base = inner(depth - 1, pow_allowed=False),
                    exponent = rand.randint(MIN_POW, MAX_POW)
                )
                return mt.Const(self._eval_constant(expression)) if self._is_constant(expression) else expression

            raise RuntimeError(f"Operator Unknown{operator}")

        return inner(depth=MAX_DEPTH, pow_allowed=True)


    #Simple probability function that provides decent results
    def _stop_probability(self, depth: int) -> float:
        return LEAF_BIAS * (MAX_DEPTH - depth) / MAX_DEPTH

    """Environmental Functions"""
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