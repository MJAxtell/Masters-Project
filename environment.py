import random
from dataclasses import dataclass
from typing import List, Callable, Set, Tuple

import my_types as mt

""""Hyperparameters - edit note, should let main pass these as arguments on init"""
MIN_POW = 2 #Minimum value for POW operator
MAX_POW = 2 #Maximum value for POW operators
MAX_DEPTH = 3 #Maximum branching depth for a rules
MIN_CLAUSES = 1 #Minimum number of independent clauses for a rule - disabled for advanced rule generation
MAX_CLAUSES = 3 #Maximum number of independent clauses for a rule
MAX_BRANCHES = 4 #Maximum number of branches for internal advanced rule generation domain splitting

SPLIT_BIAS = 0.5 #Bias for internal advanced rule gen to split after obtaining sufficient branches, [0,1]
VAR_BIAS = 0.5 #Bias for internal advanced rule gen to prefer adding new variables, [0,1]

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

    """Simple Single Random Rule Generation"""
    def random_rule(self, rand: random.Random) -> Rule:
        lhs_cond = self.random_conditional(min_clauses=MIN_CLAUSES, max_clauses=MAX_CLAUSES, rand=rand)
        rhs_expr = self.random_expression(max_depth=MAX_DEPTH, rand=rand)
        return Rule(lhs_cond, rhs_expr)

    """Advanced Random Rule Generation."""
    """AI USE DISCLAIMER - Generative AI (ChatGPT) has been used to assist in the creation
    of this function and by extension its helpers. AI was used to assist with flow logic.
    All code is handwritten, edited, and reviewed."""
    def random_split_rules(self, rand: random.Random, num_rules: int) -> List[Rule]:
        leaves: List[LeafRegion] = [LeafRegion(ranges={})]
        MAX_TOTAL_SPLITS = num_rules * MAX_CLAUSES
        splits_done = 0

        """Split bias implementation
        0 - stop when len(leaves) is sufficient.
        1 - stop when absolutely necessary"""
        while (
            len(leaves) < num_rules or (
                SPLIT_BIAS > 0
                and splits_done < MAX_TOTAL_SPLITS
                and self._can_split(leaves)
                and rand.random() < SPLIT_BIAS
            )
        ):
            rand.shuffle(leaves)
            finished_split = False

            for i, leaf in enumerate(leaves):
                can_add = leaf.constrained_count() < MAX_CLAUSES

                """Choose splittable variables.
                Will reuse already split on variables.
                Will not use new variables if out of MAX_CLAUSES budget."""

                """Variable bias implementation
                0 - never split on a variable unless necessary.
                1 - always split on a variable unless impossible.
                """
                used = [v for v in self.env_variables if v.name in leaf.ranges]
                new = [v for v in self.env_variables if v.name not in leaf.ranges]

                if VAR_BIAS == 0:
                    if can_add:
                        options = used + new
                    else:
                        options = used
                    rand.shuffle(options)

                elif can_add and new and rand.random() < VAR_BIAS:
                    rand.shuffle(new)
                    rand.shuffle(used)
                    options = new + used
                else:
                    rand.shuffle(used)
                    rand.shuffle(new)
                    options = used + new

                """For every variable chosen"""
                for var in options:
                    low, high = self._get_range(leaf, var)
                    if low == high:
                        continue

                    if var.name in leaf.ranges and leaf.constrained_count() >= MAX_CLAUSES:
                        continue
                    max_branches_here = min(MAX_BRANCHES, high - low + 1)
                    if max_branches_here < 2:
                        continue

                    branches = rand.randint(2, max_branches_here)


                    intervals = self._split_interval(low, high, branches, rand)
                    children = [self._set_range(leaf, var, a, b) for (a, b) in intervals]

                    leaves.pop(i)
                    leaves.extend(children)
                    splits_done += 1
                    finished_split = True
                    break

                if finished_split:
                    break

            if not finished_split:
                break

        if len(leaves) > num_rules:
            leaves = rand.sample(leaves, num_rules)

        #Return as human-readable conditional representation
        return [Rule(self.region_to_condition(leaf),
                     self.random_expression(max_depth=MAX_DEPTH, rand=rand)) for leaf in leaves]

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
    def _full_range(self, var: mt.Variable) -> Tuple[int, int]:
        return var.min_val, var.max_val

    def _get_range(self, leaf: LeafRegion, var: mt.Variable) -> Tuple[int, int]:
        return leaf.ranges.get(var.name, self._full_range(var))

    def _set_range(self, leaf: LeafRegion, var: mt.Variable, low: int, high: int) -> LeafRegion:
        new_ranges = dict(leaf.ranges)
        new_ranges[var.name] = (low, high)
        return LeafRegion(new_ranges)

    #Leaf splitting legality
    def _can_split(self, leaves: List[LeafRegion]) -> bool:
        for leaf in leaves:
            if leaf.constrained_count() >= MAX_CLAUSES:
                continue

            for var in self.env_variables:
                low, high = self._get_range(leaf, var)
                if high > low:
                    return True

        return False

    """Takes a low - high range and splits it into consistent (branch number of) intervals"""
    def _split_interval(self, low: int, high: int, branches: int, rand: random.Random) -> List[Tuple[int, int]]:
        length = high - low + 1
        if branches < 2:
            raise ValueError("Branches too small! Must be greater than 2")
        elif branches > length:
            raise ValueError(f"Branches too large! Must be less than length. Branches: {branches}, Length: {length}")

        cuts = sorted(rand.sample(range(low, high), branches - 1))
        intervals: List[Tuple[int, int]] = []
        start = low
        for cut in cuts:
            intervals.append((start, cut))
            start = cut + 1
        intervals.append((start, high))
        return intervals

    """Antiquated Functions"""
    # def practice_partition_two_rules(self, rand: random.Random) -> List[Rule]:
    #     var = rand.choice(self.env_variables)
    #     split = rand.randint(var.min_val, var.max_val - 1)
    #
    #     left_cond = mt.RangeCond(var=var, low=var.min_val, high=split)
    #     right_cond = mt.RangeCond(var=var, low=split+1, high=var.max_val)
    #
    #     left_rule = Rule(left_cond, self.random_expression(MAX_DEPTH, rand))
    #     right_rule = Rule(right_cond, self.random_expression(MAX_DEPTH, rand))
    #
    #     return [left_rule, right_rule]

    # """Random Comparison Generation V1"""
    # def random_comparison(self, rand: random.Random) -> mt.Conditional:
    #     var = rand.choice(self.env_variables)
    #     choice = rand.choice(["==", "!=", "<", "<=", ">", ">="])
    #
    #     #Magic number 50% chance to compare between variables vs constants
    #     if rand.random() < 0.5:
    #         value = rand.randint(var.min_val, var.max_val)
    #     else:
    #         other_variables = [v for v in self.env_variables if v is not var]
    #         #Fallback to constant if env_variables is only len() == 1
    #         if not other_variables:
    #             value = rand.randint(var.min_val, var.max_val)
    #         else:
    #             value = rand.choice(other_variables)
    #
    #     return mt.Comp(var=var, op=choice, value=value)

    """RHS Conditional Creator"""
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

        return inner(depth=max_depth, pow_allowed=True)

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

