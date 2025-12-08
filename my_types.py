from dataclasses import dataclass
from typing import Dict, Union

"""Critical types"""
#Name, minimum and maximum domain values
@dataclass
class Variable:
    name: str
    #Hardcoded minval
    min_val: int = 1
    #Hardcoded maxval
    max_val: int = 10

    #Representation
    def __repr__(self):
        return f"Variable:{self.name}, domain=[{self.min_val} {self.max_val}]"

@dataclass
class Const:
    value: int

@dataclass
class VariableReference:
    var: Variable

"""== Operators =="""
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

#Types of expressions
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

#Matthew's printer
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