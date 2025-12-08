from typing import List, Dict
import random

from my_types import Guess, Observation
from environment import Environment

class Agent:
    def __init__(self, env: Environment):
        self.environment = env
        self.history: List[Observation] = []

    def make_guess(self) -> Guess:
        values: Dict[str, int] = {}
        for var in self.environment.env_variables:
            values[var.name] = random.randint(var.min_val, var.max_val)
        return Guess(values=values)

    def conduct_experiment(self) -> Observation:
        guess = self.make_guess()
        output = self.environment.evaluate(guess)
        observation = Observation(inputs=guess.values, output=output)
        self.history.append(observation)
        return observation