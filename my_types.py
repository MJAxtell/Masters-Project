from dataclasses import dataclass
from typing import Dict, Union

"""Critical types"""
@dataclass
class Variable:
    name: str
    min_val: int = 1
    max_val: int = 10

    def __repr__(self):
        return f"Variable:{self.name}, domain=[{self.min_val} {self.max_val}]"

"""LHS Conditional Types"""

#Op attribute comparisons included are listed in the evaluation function
@dataclass
class Comp:
    var: Variable
    op: str
    value: Union[int, Variable]

@dataclass
class AndCond:
    left: "Conditional"
    right: "Conditional"

@dataclass
class OrCond:
    left: "Conditional"
    right: "Conditional"

Conditional = Union[Comp, AndCond, OrCond]

"""RHS Expression Types"""
@dataclass
class Const:
    value: int

@dataclass
class VariableReference:
    var: Variable

@dataclass
class Add:
    left: "Expression"
    right: "Expression"

@dataclass
class Mul:
    left: "Expression"
    right: "Expression"

@dataclass
class Pow:
    base: "Expression"
    exponent: int

Expression = Union[Const, VariableReference, Add, Mul, Pow]

#Edit note: Rename to query?
#A dictionary of variable name/value pairs proposed by an agent
@dataclass
class Guess:
    values: Dict[str, int]

    def __getitem__(self, name: str) -> int:
        return self.values[name]

#A dictionary of variable name/value pairs (guess) with associated output from environment
@dataclass
class Observation:
    inputs: Dict[str, int]
    output: int

#== Expression Evaluator ==
#Editing note: Handles mul, add, pow for now
def evaluate_expression(expression: Expression, guess: Guess) -> int:
    if isinstance(expression, Const):
        return expression.value

    if isinstance(expression, VariableReference):
        return guess[expression.var.name]

    if isinstance(expression, Add):
        left_expression = evaluate_expression(expression.left, guess)
        right_expression = evaluate_expression(expression.right, guess)
        return left_expression + right_expression

    if isinstance(expression, Mul):
        left_expression = evaluate_expression(expression.left, guess)
        right_expression = evaluate_expression(expression.right, guess)
        return left_expression * right_expression

    if isinstance(expression, Pow):
        base = evaluate_expression(expression.base, guess)
        return base ** expression.exponent

    raise TypeError(f"Unrecognized node, the cuprit looked like: {expression}")

#Comparisons included: ==, !=, <, <=, >, >=
def evaluate_conditional(conditional: Conditional, guess: Guess) -> bool:
    if isinstance(conditional, Comp):
        left = guess[conditional.var.name]

        if isinstance(conditional.value, int):
            right = conditional.value
        else:
            right = guess[conditional.value.name]

        if conditional.op == "==":
            return left == right
        if conditional.op == "!=":
            return left == right
        if conditional.op == "<":
            return left < right
        if conditional.op == "<=":
            return left <= right
        if conditional.op == ">":
            return left > right
        if conditional.op == ">=":
            return left >= right
        raise ValueError("Unrecognized comparison operator, the culprit looked like: {conditional.op}")

    if isinstance(conditional, AndCond):
        return evaluate_conditional(conditional.left, guess) and evaluate_conditional(conditional.right, guess)

    if isinstance(conditional, OrCond):
        return evaluate_conditional(conditional.left, guess) or evaluate_conditional(conditional.right, guess)

    raise TypeError(f"Unrecognized node, the culprit looked like: {conditional}")

#Printers
#Editing note: To be refactored to dedicated file
def rule_printer(conditional: Conditional, expression: Expression) -> str:
    lhs = lhs_conditional_printer(conditional)
    rhs = rhs_expression_printer(expression)
    return f"IF {lhs} THEN {rhs}"

def rhs_expression_printer(expression: Expression) -> str:
    if isinstance(expression, Const):
        return str(expression.value)

    if isinstance(expression, VariableReference):
        return expression.var.name

    if isinstance(expression, Add):
        return f"({rhs_expression_printer(expression.left)} + {rhs_expression_printer(expression.right)})"

    if isinstance(expression, Mul):
        return f"({rhs_expression_printer(expression.left)} * {rhs_expression_printer(expression.right)})"

    if isinstance(expression, Pow):
        return f"({rhs_expression_printer(expression.base)} ** {expression.exponent})"

    raise TypeError(f"Expression is unknown! Was: {type(expression)}")

def lhs_conditional_printer(conditional: Conditional) -> str:
    if isinstance(conditional, Comp):
        left = conditional.var.name

        if isinstance(conditional.value, int):
            right = str(conditional.value)
        else:
            right = str(conditional.value.name)

        if conditional.op == "==":
            return f"{left} == {right}"
        if conditional.op == "!=":
            return f"{left} != {right}"
        if conditional.op == "<":
            return f"{left} < {right}"
        if conditional.op == "<=":
            return f"{left} <= {right}"
        if conditional.op == ">":
            return f"{left} > {right}"
        if conditional.op == ">=":
            return f"{left} >= {right}"

    if isinstance(conditional, AndCond):
        return f"{lhs_conditional_printer(conditional.left)} and {lhs_conditional_printer(conditional.right)}"

    if isinstance(conditional, OrCond):
        return f"{lhs_conditional_printer(conditional.left)} or {lhs_conditional_printer(conditional.right)}"\

    raise TypeError(f"Condition is unknown! Was: {type(conditional)}")