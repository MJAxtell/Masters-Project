import my_types as mt
from environment import Environment
from agent import Agent
import random
from printer import MainPrinter

NUM_INITIAL = 20
DEPTH_CONTROLLED = 10
FOCUSED_THRESHOLD = 0.01
FOCUSED_RANDS = 5
FOCUSED_DEPTH = 15

NAME_DOMAINS = {
        "A": (1, 50),
        "B": (1, 50),
        "C": (1, 50),
        "D": (1, 50),
        "E": (1, 50),
    }

def main():
    #Currently, directly passes names_domains and a seed for random generation of a single rule
    #Later, will hand hyperparameters which include variable names and domains
    input("Enter to begin...")

    env = Environment(NAME_DOMAINS, seed=random.randint(1,100))

    printer = MainPrinter()
    agent = Agent(env, printer=printer)

    for i, rule in enumerate(env.rules): print(f"Rule {i}:", mt.rule_printer(rule.condition, rule.rhs_expr))

    """Train Agent"""
    agent.train(
        num_initial=NUM_INITIAL,
        depth_controlled=DEPTH_CONTROLLED,
        focused_threshold=FOCUSED_THRESHOLD,
        focused_rands=FOCUSED_RANDS,
        focused_depth=FOCUSED_DEPTH,
    )

    # for hist in agent.history:
    #     print(hist)

if __name__ == "__main__":
    main()