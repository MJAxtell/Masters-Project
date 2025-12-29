from typing import List, Dict, Optional
from dataclasses import dataclass
from printer import MainPrinter
import random

from my_types import Guess, Observation, ControlledGroup
from environment import Environment

@dataclass
class TrainingCycle:
    num_initial: int
    num_controlled: int
    depth_controlled: int

    initial_observations: Optional[List[Observation]] = None
    controlled_groups: Optional[List[ControlledGroup]] = None

class Agent:
    def __init__(self, env: Environment, printer: Optional[MainPrinter] = None):
        self.environment = env
        self.printer = printer or MainPrinter()
        self.var_by_name = {v.name: v for v in self.environment.env_variables}

        #Persistent attributes
        self.history: List[Observation] = []

        #Active training cycle
        self.cycle: TrainingCycle | None = None

    def train(self, num_initial: int, num_controlled: int, depth_controlled: int):
        self.begin_cycle(num_initial, num_controlled, depth_controlled)

        """Safety Checks"""
        #Safety 1 - depth of controlled  is greater than number of controlled
        for var in self.environment.env_variables:
            domain_size = (var.max_val - var.min_val) + 1
            if (domain_size/2) < self.cycle.depth_controlled:
                raise ValueError(f"Cannot train! Domain of variable: {var.name} is too small! "
                                 f"Domain needs to be at least {self.cycle.depth_controlled * 2}",
                                 f"Domain size == {domain_size}, Depth == {self.cycle.depth_controlled}")

        self.printer.print_newline()
        self.printer.begin_cycle(self.cycle)

        self.cycle.initial_observations = self.conduct_initials()
        self.cycle.controlled_groups = self.conduct_controlled(self.cycle.initial_observations)

    def begin_cycle(self, num_initial: int, num_controlled: int, depth_controlled: int):
        self.cycle = TrainingCycle(
            num_initial=num_initial,
            num_controlled=num_controlled,
            depth_controlled=depth_controlled,
        )

    def make_guess(self) -> Guess:
        values: Dict[str, int] = {}
        for var in self.environment.env_variables:
            values[var.name] = random.randint(var.min_val, var.max_val)
        return Guess(values=values)

    def make_controlled_guess(self, base_obs: Observation, varied_var: str, used_values: set) -> Guess:
        values: Dict[str, int] = {}
        for var in self.environment.env_variables:
            if var.name == varied_var:
                while True:
                    candidate = random.randint(var.min_val, var.max_val)
                    if candidate not in used_values:
                        used_values.add(candidate)
                        values[var.name] = candidate
                        break

            else:
                values[var.name] = base_obs.inputs[var.name]

        return Guess(values=values)

    """=== Conduction initials/controlled functions have dedicated printers ==="""

    def conduct_initials(self) -> List[Observation]:
        initials: List[Observation] = []
        for i in range(self.cycle.num_initial):
            guess = self.make_guess()
            output = self.environment.evaluate(guess)
            observation = Observation(inputs=guess.values, output=output)
            initials.append(observation)
            self.history.append(observation)

            """Printer call"""
            self.printer.print_initials(i, observation)

        return initials

    """Chooses num_controlled initial observations, limited by number of initial observations, then
    produces depth_controlled observations with a random variable varied"""
    def conduct_controlled(self, initials: List[Observation]) -> List[ControlledGroup]:
        controlled: List[ControlledGroup] = []

        #Limit number of controlled by number of initials
        num_controlled = min(self.cycle.num_controlled, len(initials))

        #Shuffle initials for iteration
        initial_shuffled = initials[:]
        random.shuffle(initial_shuffled)

        #Shuffle environmental variables for iteration
        env_vars_names: List[str] = [var.name for var in self.environment.env_variables]
        random.shuffle(env_vars_names)

        num_vars = len(env_vars_names)

        self.printer.print_emptyline()

        for i in range(num_controlled):
            controlled_initial = initial_shuffled[i]
            varied_var = env_vars_names[i % num_vars]

            used_values = {controlled_initial.inputs[varied_var]}
            batch_observations: List[Observation] = []

            """Printer call"""
            self.printer.print_controlled_header(i, varied_var)

            for j in range(self.cycle.depth_controlled):
                guess = self.make_controlled_guess(controlled_initial, varied_var, used_values)
                output = self.environment.evaluate(guess)

                observation = Observation(inputs=guess.values, output=output)
                batch_observations.append(observation)
                self.history.append(observation)

                """Printer Call"""
                self.printer.print_controlled(i=i, depth=j, obs=observation)

            controlled.append(
                ControlledGroup(varied_var=varied_var,
                                observations=batch_observations,)
            )

            self.printer.print_emptyline()

        return controlled

    def generate_clusters(self):
        return