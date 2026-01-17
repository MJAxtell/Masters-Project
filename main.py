import my_types as mt
from environment import Environment
from agent import Agent
import random
from printer import MainPrinter
from copy import deepcopy

"""NUM_RULES must be at least 2 as a rule.
A rule must never have an entire domain as its range nor the rule set leave ambiguous cases."""
NUM_RULES = 10 #Maximum number of rules for the environment

NUM_INITIAL = 20 #Number of initial, random observations
DEPTH_CONTROLLED = 5 #Depth of controlled expansion on initial observations
FOCUSED_THRESHOLD = 0.01 #Threshold to exceed for entropy measurements to qualify a group for focusing
#FOCUSED_RANDS = 5 #Number of random observation contexts to mix into selected focused contexts
FOCUSED_RANDS = 6 #The number of random contexts to generate per varied variable for rule boundary detection
FOCUSED_DEPTH = 20 #Depth of exploration per focused context

"""Minimum number of observations in proposed rule fragment before regression will be performed
Making this number too large will exclude certain rule with small conditional domains."""
REGRESSION_THRESHOLD = 1

"""The largest variable domain in a proposed rule must exceed this value
or it is considered weak."""
WEAK_THRESHOLD = 3

"""Rule validation / Postprocessing"""
VALIDATION_SAMPLES = 5 #Number of samples to take throughout the validation process
VALIDATION_THRESHOLD = 0.6

"""Main/Train specific"""
NAME_DOMAINS = {
        "A": (1, 50),
        "B": (1, 50),
        "C": (1, 50),
        "D": (1, 50),
        "E": (1, 50),
    }

NUM_CYCLES = 10 #Number of cycles per test
NUM_TESTS = 10 #Number of tests

def main():
    #Currently, directly passes names_domains and a seed for random generation of a single rule
    #Later, will hand hyperparameters which include variable names and domains
    input("Enter to begin...")

    training_results: [mt.TrainingResult] = []

    """Train Agent"""
    """Testing cycles - different environments"""
    for _ in range(NUM_TESTS):
        env = Environment(NAME_DOMAINS, num_rules=NUM_RULES, seed=random.randint(1, 100))
        printer = MainPrinter()
        agent = Agent(env, printer=printer)

        """Training cycles - safe environment"""
        for i, rule in enumerate(env.rules): print(f"Rule {i}:", mt.rule_printer(rule.condition, rule.rhs_expr))

        for _ in range(NUM_CYCLES):
            agent.train(
                num_initial=NUM_INITIAL,
                depth_controlled=DEPTH_CONTROLLED,
                focused_threshold=FOCUSED_THRESHOLD,
                focused_rands=FOCUSED_RANDS,
                focused_depth=FOCUSED_DEPTH,
                regression_threshold=REGRESSION_THRESHOLD,
                weak_threshold=WEAK_THRESHOLD,
                validation_samples=VALIDATION_SAMPLES,
                validation_threshold=VALIDATION_THRESHOLD,
            )
        training_results.append(mt.TrainingResult(
            rules = deepcopy(agent.agent_rules),
            environment_rules = env.rules,
        ))

if __name__ == "__main__":
    main()
