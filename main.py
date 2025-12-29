import my_types as mt
from environment import Environment
from agent import Agent
import random
from printer import MainPrinter

NUM_INITIAL = 10
NUM_CONTROLLED = 3
DEPTH_CONTROLLED = 5
NAME_DOMAINS = {
        "A": (1, 20),
        "B": (1, 20),
        "C": (1, 20),
        "D": (1, 20),
        "E": (1, 20),
        "F": (1, 20),
    }

def main():
    #Currently, directly passes names_domains and a seed for random generation of a single rule
    #Later, will hand hyperparameters which include variable names and domains
    input("Enter to begin...")

    env = Environment(NAME_DOMAINS, seed=random.randint(1,100))

    printer = MainPrinter()
    agent = Agent(env, printer=printer)

    for i, rule in enumerate(env.rules): print(f"Rule {i}:", mt.rule_printer(rule.condition, rule.rhs_expr))

    """Run Agent"""
    agent.train(
        num_initial=NUM_INITIAL,
        num_controlled=NUM_CONTROLLED,
        depth_controlled=DEPTH_CONTROLLED,
    )

    # for hist in agent.history:
    #     print(hist)

if __name__ == "__main__":
    main()