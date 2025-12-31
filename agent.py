from typing import List, Dict, Optional
from dataclasses import dataclass
from printer import MainPrinter
from collections import Counter
import random
import math

from my_types import Guess, Observation, ControlledGroup
from environment import Environment

@dataclass
class TrainingCycle:
    num_initial: int
    depth_controlled: int
    focused_threshold: float
    focused_rands: int
    focused_depth: int

    initial_observations: Optional[List[Observation]] = None
    controlled_groups: Optional[List[ControlledGroup]] = None
    entropy_scores: List[float] = None

class Agent:
    def __init__(self, env: Environment, printer: Optional[MainPrinter] = None):
        self.environment = env
        self.printer = printer or MainPrinter()
        self.var_by_name = {v.name: v for v in self.environment.env_variables}

        #Persistent attributes
        self.history: List[Observation] = []

        #Active training cycle
        self.cycle: TrainingCycle | None = None

    def train(self, num_initial: int, depth_controlled: int, focused_threshold: float, focused_rands: int, focused_depth: int):
        self.begin_cycle(num_initial, depth_controlled, focused_threshold, focused_rands, focused_depth)

        """Safety Checks"""
        #Safety 1 - depth of controlled  is greater than number of controlled
        for var in self.environment.env_variables:
            domain_size = (var.max_val - var.min_val) + 1
            if (domain_size/2) < self.cycle.depth_controlled:
                raise ValueError(f"Cannot train! Domain of variable: {var.name} is too small! "
                                 f"Domain needs to be at least {self.cycle.depth_controlled * 2}",
                                 f"Domain size == {domain_size}, Depth == {self.cycle.depth_controlled}")

        self.printer.print_emptyline()
        self.printer.begin_cycle(self.cycle)

        self.cycle.initial_observations = self.conduct_initials()
        self.cycle.controlled_groups = self.conduct_controlled(self.cycle.initial_observations)
        self.cycle.entropy_scores = self.conduct_entropy_scores(self.cycle.controlled_groups)
        self.cycle.focused_experiments = self.conduct_focused()

    def begin_cycle(self, num_initial: int,
                    depth_controlled: int,
                    focused_threshold: float,
                    focused_rands: int,
                    focused_depth: int):
        self.cycle = TrainingCycle(
            num_initial=num_initial,
            depth_controlled=depth_controlled,
            focused_threshold=focused_threshold,
            focused_rands=focused_rands,
            focused_depth=focused_depth
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

        #Shuffle initials for iteration
        initial_shuffled = initials[:]
        random.shuffle(initial_shuffled)

        #Shuffle environmental variables for iteration
        env_vars_names: List[str] = [var.name for var in self.environment.env_variables]
        random.shuffle(env_vars_names)

        num_vars = len(env_vars_names)

        self.printer.print_emptyline()

        for i, controlled_initial in enumerate(initial_shuffled):
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

    def conduct_entropy_scores(self, controlled_groups: List[ControlledGroup]) -> List[float]:
        scores: List[float] = []
        assert controlled_groups is not None
        varied_vars = [group.varied_var for group in controlled_groups]

        for group in controlled_groups:
            outputs = [obs.output for obs in group.observations]

            #Counts to provide probability distribution from obtained counts
            counts = Counter(outputs)
            total = len(outputs)

            #Shannon entropy formula: H(X) = -SIG( P(xi) log2( P(xi) ) )
            entropy = 0.0
            for count in counts.values():
                p = count / total
                entropy -= p * math.log2(p)

            scores.append(entropy)

        """Printer call"""
        self.printer.print_entropy_scores(scores, varied_vars)
        return scores

    def conduct_focused(self) -> List[ControlledGroup]:
        """Fairly complicated; gets suitably entropic variables, for every suitably entropic variable
        gather all the observations from all suitably entropic observation groups into a variable
        keyed dict. Add focused_rands additional random contexts. For every variable, perform
        focused_depth controlled guesses on those contexts. Return a list of controlled groups."""
        focused_groups: List[ControlledGroup] = []
        contributing: Dict[str, List[Observation]] = {}

        assert self.cycle.controlled_groups is not None
        assert self.cycle.entropy_scores is not None

        #Promising variables populate dict storing observations
        for group, score in zip(self.cycle.controlled_groups, self.cycle.entropy_scores):
            if score < self.cycle.focused_threshold:
                continue
            var = group.varied_var
            contributing.setdefault(var, []).extend(group.observations)

        #For every chosen variable
        for var, base_observations in contributing.items():
            focused_observations: List[Observation] = []

            #Add randomized baselines
            for _ in range(self.cycle.focused_rands):
                baseline_guess = self.make_guess()
                output = self.environment.evaluate(baseline_guess)
                baseline_obs = Observation(inputs=baseline_guess.values, output=output)

                base_observations.append(baseline_obs)
                self.history.append(baseline_obs)

            #Expand each baseline by varying only `var`
            for base_obs in base_observations:
                used_values = {base_obs.inputs[var]}

                for _ in range(self.cycle.focused_depth):
                    guess = self.make_controlled_guess(
                        base_obs=base_obs,
                        varied_var=var,
                        used_values=used_values
                    )
                    output = self.environment.evaluate(guess)
                    obs = Observation(inputs=guess.values, output=output)

                    focused_observations.append(obs)

            focused_groups.append(
                ControlledGroup(varied_var=var, observations=focused_observations)
            )

        """Printer calls"""
        self.printer.print_emptyline()
        self.printer.print_focused(focused_groups)
        return focused_groups