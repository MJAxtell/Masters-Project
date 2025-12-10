import my_types as mt
from environment import Environment
from agent import Agent
import random

def main():

    name_domains = {
        "A": (1, 10),
        "B": (2, 10),
        "C": (3, 10),
        "D": (4, 9),
        "E": (0, 5),
        "F": (6, 7),
    }

    #Currently, directly passes names_domains and a seed for random generation of a single rule
    #Later, will hand hyperparameters which include variable names and domains
    print("\n" * 10)
    env = Environment(name_domains, seed=random.randint(1,100))
    for i, rule in enumerate(env.rules): print(f"Rule {i}:", mt.rule_printer(rule.condition, rule.rhs_expr))

    agent = Agent(env)

    num_experiments = 100
    for i in range(num_experiments):
        obs = agent.conduct_experiment()
        print(f"Experiment {i}: inputs {obs.inputs}, outputs {obs.output}\n")

if __name__ == "__main__":
    main()