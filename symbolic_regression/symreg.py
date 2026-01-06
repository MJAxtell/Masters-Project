from typing import List, Dict
from my_types import Observation, Expression, Variable, VariableReference, Const, Add, Mul
from functools import partial

import operator
import random
import numpy as np
import math

#Deap setup
from deap import base, creator, tools, gp, algorithms

"""AI USE DISCLAIMER - Generative AI (ChatGPT) has been used to advise on this snippet
to solve a global registries issue. All code is handwritten, edited, and reviewed."""
if not hasattr(creator, 'FitnessMin'):
    creator.create("FitnessMin", base.Fitness, weights=(-1.0,))
if not hasattr(creator, 'Individual'):
    creator.create("Individual", gp.PrimitiveTree, fitness=creator.FitnessMin)

"""MIN_EPH and MAX_EPH values MUST match symreg.py MIN_CONST and MAX_CONST values respectively.
Project inductive bias is that we use integers only, no floats, 1-10"""
MIN_EPH = 1 #Minimum ephemeral constant integer range - default 1
MAX_EPH = 10 #Maximum ephemeral constant integer range - default 10

POPULATION_SIZE = 300 #Number of candidates explored per generation
CROSSOVER_PROBABILITY = 0.5
MUTATION_PROBABILITY = 0.2
NUM_GENERATIONS = 40

ALPHA = 0.2
GEN_MIN = 1 #Minimum tree depth
GEN_MAX = 3 #Maximum tree depth
BLOAT_MAX = 10 #Tree node cap through mutation / crossover
MAX_DEPTH = 4 #Maximum tree depth

#Tournament selection = competition of tournsize expressions for best fitness to move on to next generation
TOURN_SIZE = 3 #Number of competing expressions in tournament selection

class SymbolicRegressor:
    def __init__(self, var_by_name: Dict[str, Variable]):
        self.var_by_name = var_by_name

    def regress(self, observations: List[Observation], names: List[str]) -> Expression:
        X, y = self._parameter_builder(observations, names)
        primitives = self._build_primitives(names)

        toolbox = self._build_toolbox(primitives, X, y)
        population = toolbox.population(n=POPULATION_SIZE)

        #DEAP standard evolutionary loop
        algorithms.eaSimple(
            population,
            toolbox,
            cxpb=CROSSOVER_PROBABILITY,
            mutpb=MUTATION_PROBABILITY,
            ngen=NUM_GENERATIONS,
            verbose=False,
        )

        champion = tools.selBest(population, k=1)[0]
        return self._tree_to_expression(champion, self.var_by_name)

    #Runs the tree and returns our Expression class
    def _tree_to_expression(self, tree: gp.PrimitiveTree, var_by_name: Dict[str, Variable]) -> Expression:
        def inner(i: int):
            node = tree[i]

            if isinstance(node, gp.Primitive):
                if node.name == "add":
                    left, next_i = inner(i + 1)
                    right, next_i = inner(next_i)
                    return Add(left, right), next_i

                if node.name == "mul":
                    left, next_i = inner(i + 1)
                    right, next_i = inner(next_i)
                    return Mul(left, right), next_i

                raise ValueError(f"Primitive type unknown! Type is {type(node.name)}")

            if isinstance(node, gp.Terminal):
                value = node.value

                if isinstance(value, str):
                    return VariableReference(var_by_name[value]), i + 1

                return Const(value), i + 1

            raise TypeError(f"Node type unknown! Type is {type(node)}")

        # Expression capture only
        expression, _ = inner(0)
        return expression

    def _parameter_builder(self, observations: List[Observation], names: List[str]):
        """X[i][j] indexes observation by i and variable name by j
        Y[i] indexes observation output
        Returns list of observations values and list of observation outputs"""
        X: List[List[int]] = []
        y: List[float] = []

        for observation in observations:
            obs_item = [observation.inputs[name] for name in names]
            X.append(obs_item)
            y.append(observation.output)

        return X, y

    def _build_primitives(self, names: List[str]) -> gp.PrimitiveSet:
        primitives = gp.PrimitiveSet("AddMul", len(names))

        primitives.addPrimitive(operator.add, 2)
        primitives.addPrimitive(operator.mul, 2)

        #Renames to actual names for convenience
        for i, name in enumerate(names):
            #Dict unpacking trick
            primitives.renameArguments(**{f"ARG{i}": name})

        primitives.addEphemeralConstant("EC", partial(random.randint, MIN_EPH, MAX_EPH))

        return primitives

    """Protects integers. Infinite floats become 0.
    Negative floats become zero. Otherwise transform to int."""
    def _integer_safety(self, x):
        if not math.isfinite(x):
            return 0
        if x < 0:
            return 0
        return int(x)

    def _fitness(self, tree, pset, X, y, alpha):
        tree_function = gp.compile(expr=tree, pset=pset)
        Y_hat = np.array([self._integer_safety(tree_function(*observation)) for observation in X])

        #Regular mean squared error
        mse = np.mean((Y_hat - y) ** 2)

        #Parsimony penalized mean squared error
        return (mse + alpha * len(tree),)

    """Allows for creation, scoring, and evolution of expressions"""
    def _build_toolbox(self, primitives, X, y):
        toolbox = base.Toolbox()

        """https://deap.readthedocs.io/en/master/api/base.html
        register(alias, function, argument/arguments)"""

        #Defines expressions
        toolbox.register(
            "expression",
            gp.genHalfAndHalf,  #Mix of full and grown (unfilled) tree shapes, generates diverse initial trees
            pset=primitives,
            min_ = GEN_MIN,
            max_ = GEN_MAX,
        )

        #Defines a candidate solution
        toolbox.register(
            "single",
            tools.initIterate,
            creator.Individual,
            toolbox.expression,
        )

        #Defines population of candidate solutions
        toolbox.register(
            "population",
            tools.initRepeat,
            list,
            toolbox.single,
        )

        """Following toolbox registrations are strictly named"""
        #Objective function
        toolbox.register(
            "evaluate",
            self._fitness,
            pset = primitives,
            X = X,
            y = y,
            alpha = ALPHA,
        )

        #Selection method, tournament selection is GP default
        toolbox.register(
            "select",
            tools.selTournament,
            tournsize=3,
        )

        #Recombination capabilities
        toolbox.register(
            "mate",
            gp.cxOnePoint, #Swap random subtrees in parents, produce two children
        )

        #Mutation/noise capabilities
        toolbox.register(
            "mutate",
            gp.mutUniform, #Replace random subtree with new subtree
            expr=toolbox.expression,
            pset=primitives,
        )

        # Protection against bloating issues by limiting tree size
        # toolbox.decorate("mate", gp.staticLimit(key=len, max_value=BLOAT_MAX))
        # toolbox.decorate("mutate", gp.staticLimit(key=len, max_value=BLOAT_MAX))

        # Limitation of maximum tree depth
        toolbox.decorate(
            "mate",
            gp.staticLimit(key=operator.attrgetter("height"), max_value=MAX_DEPTH),
        )

        toolbox.decorate(
            "mutate",
            gp.staticLimit(key=operator.attrgetter("height"), max_value=MAX_DEPTH),
        )

        return toolbox