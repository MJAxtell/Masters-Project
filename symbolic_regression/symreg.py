from typing import List
from my_types import Observation, Expression
from functools import partial

import operator
import random
import numpy as np

#Deap setup
from deap import base, creator, tools, gp, algorithms

"""AI USE DISCLAIMER - Generative AI (ChatGPT) has been used to advise on this snippet
to solve a global registries issue. All code is handwritten, edited, and reviewed."""
if not hasattr(creator, 'FitnessMin'):
    creator.create("FitnessMin", base.Fitness, weights=(-1.0,))
if not hasattr(creator, 'Individual'):
    creator.create("Individual", gp.PrimitiveTree, fitness=creator.FitnessMin)

POPULATION_SIZE = 300 #Number of candidates explored per generation
CROSSOVER_PROBABILITY = 0.5
MUTATION_PROBABILITY = 0.2
NUM_GENERATIONS = 40

ALPHA = 0.01
GEN_MIN = 1 #Minimum tree depth
GEN_MAX = 3 #Maximum tree depth
BLOAT_MAX = 10 #Maximum tree length through mutation / crossover

#Tournament selection = competition of tournsize expressions for best fitness to move on to next generation
TOURN_SIZE = 3 #Number of competing expressions in tournament selection

class SymbolicRegressor:
    def regress(self, observations: List[Observation], names: List[str]) -> Expression:
        X, Y = self._parameter_builder(observations, names)
        primitives = self._build_primitives(names)

        toolbox = self._build_toolbox(X, Y, primitives)
        population = toolbox.multi(n=POPULATION_SIZE)

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
        return self._tree_to_expression(champion, names)

    def _parameter_builder(self, observations: List[Observation], names: List[str]):
        """X[i][j] indexes observation by i and variable name by j
        Y[i] indexes observation output
        Returns list of observations values and list of observation outputs"""
        X: List[List[int]] = []
        Y: List[float] = []

        for observation in observations:
            obs_item = [observation.inputs[name] for name in names]
            X.append(obs_item)
            Y.append(observation.output)

        return X, Y

    def _build_primitives(self, names: List[str]) -> gp.PrimitiveSet:
        primitives = gp.PrimitiveSet("AddMul", len(names))

        primitives.addPrimitive(operator.add, 2)
        primitives.addPrimitive(operator.mul, 2)

        #Renames to actual names for convenience
        for i, name in enumerate(names):
            #Dict unpacking trick
            primitives.renameArguments(**{f"ARG{i}": name})

        primitives.addEphemeralConstant("EC", partial(random.uniform, -1.0, 1.0))

        return primitives

    def _fitness(self, tree, primitives, X, Y):
        tree_function = gp.compile(exprn=tree, pset=primitives)
        Y_hat = np.array([tree_function(*observation) for observation in X])

        #Regular mean squared error
        mse = np.mean((Y_hat - Y) ** 2)

        #Parsimony penalized mean squared error
        return (mse + ALPHA * len(tree),)

    """Allows for creation, scoring, and evolution of expressions"""
    def _build_toolbox(self, primitives, X, Y):
        toolbox = base.Toolbox()

        """https://deap.readthedocs.io/en/master/api/base.html
        register(alias, function, argument/arguments)"""

        #Defines expressions
        toolbox.register(
            "expression",
            gp.genHalfAndHalf,  #Mix of full and grown (unfilled) tree shapes, generates diverse initial trees
            primitives=primitives,
            min_ = GEN_MIN,
            max = GEN_MAX,
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
            "multi",
            tools.initRepeat,
            list,
            toolbox.single,
        )

        """Following toolbox registrations are strictly named"""
        #Objective function
        toolbox.register(
            "evaluate",
            self._fitness,
            primitives = primitives,
            X = X,
            Y = Y,
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
            primitives=primitives,
        )

        #Protection against bloating issues by limiting tree size
        toolbox.decorate("mate", gp.staticLimit(key=len, max_value=BLOAT_MAX))
        toolbox.decorate("mutate", gp.staticLimit(key=len, max_value=BLOAT_MAX))

        return toolbox