import my_types as mt
from environment import Environment
from agent import Agent

def main():
    #Needs to include at least A, B, and C for hardcoded environment
    name_domains = {
        "A": (1, 10),
        "B": (2, 10),
        "C": (3, 10),
        "D": (4, 9),
        "E": (15, 88),
        "F": (6, 7),
    }

    #Currently, directly passes names_domains for hardcoded, handles without explicit rules
    #Later, will hand hyperparameters which include variable names and domains
    env = Environment(name_domains)
    agent = Agent(env)

    num_experiments = 20
    for i in range(num_experiments):
        obs = agent.conduct_experiment()
        print(f"Experiment {i}: inputs {obs.inputs}, outputs {obs.output}\n")

if __name__ == "__main__":
    main()