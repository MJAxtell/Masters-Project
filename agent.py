from typing import List, Dict, Optional
from dataclasses import dataclass
from printer import MainPrinter
from collections import Counter, defaultdict
import random
import math

import my_types as mt
from environment import Environment
from symbolic_regression import SymbolicRegressor
from rule_processing import conduct_identical_merge
from validation import Validator

@dataclass
class TrainingCycle:
    num_initial: int
    depth_controlled: int
    focused_threshold: float
    focused_rands: int
    focused_depth: int
    regression_threshold: int
    weak_threshold: int
    validation_samples: int
    validation_threshold: float

    initial_observations: Optional[List[mt.Observation]] = None
    controlled_groups: Optional[List[mt.ControlledGroup]] = None
    entropy_scores: List[float] = None

    probe_traces: List[mt.OrderedSweep] = None

    #focused_observations: List[FocusedGroup] = None
    #primary_hypotheses: List[ClusteredHypotheses] = None

class Agent:
    def __init__(self, env: Environment, printer: Optional[MainPrinter] = None):
        self.environment = env
        self.printer = printer or MainPrinter()
        self.var_by_name = {v.name: v for v in self.environment.env_variables}
        self.symbolic_regressor = SymbolicRegressor(self.var_by_name)

        #Persistent attributes
        self.agent_rules: List[mt.ProposedRule] = []

        #Active training cycle
        self.cycle: TrainingCycle | None = None

        self.validator = Validator(env)

    def train(self, num_initial: int, depth_controlled: int, focused_threshold: float, focused_rands: int, focused_depth: int, regression_threshold: int, weak_threshold: int, validation_samples: int, validation_threshold: float):
        self.begin_cycle(num_initial, depth_controlled, focused_threshold, focused_rands, focused_depth, regression_threshold, weak_threshold, validation_samples, validation_threshold)

        """Safety Checks"""
        #Safety 1 - depth of controlled is greater than number of possible controlled
        for var in self.environment.env_variables:
            domain_size = (var.max_val - var.min_val) + 1
            if (domain_size/2) < self.cycle.depth_controlled:
                raise ValueError(f"Cannot train! Domain of variable: {var.name} is too small! "
                                 f"Domain needs to be at least {self.cycle.depth_controlled * 2}",
                                 f"Domain size == {domain_size}, Depth == {self.cycle.depth_controlled}")

        self.printer.begin_cycle(self.cycle)

        self.cycle.initial_observations = self.conduct_initials()

        self.printer.print_controlled_begin()
        self.cycle.controlled_groups = self.conduct_controlled(self.cycle.initial_observations)

        self.cycle.entropy_scores = self.conduct_entropy_scores(self.cycle.controlled_groups)

        self.cycle.probe_traces = self.conduct_probe_traces()

        """Fragment detection"""
        all_fragments = []
        for sweep in self.cycle.probe_traces:
            fragments = self.conduct_fragment_sweep(sweep)
            all_fragments.extend(fragments)
        for fragment in all_fragments:
            fragment.signature = self._assign_signature(fragment)

        """Conduct symbolic regression on clustered fragments"""
        rule_candidates = self.cluster_fragments(all_fragments)
        rule_candidates = self.pool_weak_rules(rule_candidates)
        self.cycle.initial_rules = self.conduct_symbolic_regression(
            rule_candidates
        )
        self.printer.print_proposed_rules(self.cycle.initial_rules)

        """===Validation Begins==="""
        """Initial validation pass"""
        validator_result = self.validator.validate_initials(
            self.cycle.initial_rules,
            self.cycle.validation_samples,
            self.cycle.validation_threshold
        )
        self.printer.print_rule_validation(self.cycle.initial_rules, validator_result[1])
        self.cycle.initial_rules = validator_result[0]

        """Merge then validate identical rules - add to persistent agent laws"""
        validator_result = self.validator.validate_initials(
            conduct_identical_merge(self.cycle.initial_rules),
            self.cycle.validation_samples,
            self.cycle.validation_threshold
        )
        self.cycle.initial_rules = validator_result[0]
        self.printer.print_merged_rule_validation(self.cycle.initial_rules, validator_result[1])

        """Extend agent global ruleset"""
        self.agent_rules.extend(conduct_identical_merge(self.cycle.initial_rules))
        self.printer.print_identical_merged(self.agent_rules)

        """Validate/merge global rules"""
        self.agent_rules = self.validator.conduct_rule_merges(self.agent_rules, self.cycle.validation_samples, self.cycle.validation_threshold)
        self.printer.print_global_merged(self.agent_rules)

        """Merge Global Identicals"""
        self.agent_rules = conduct_identical_merge(self.agent_rules)
        self.printer.print_global_identical_merged(self.agent_rules)

        return self.agent_rules

    def begin_cycle(self, num_initial: int,
                    depth_controlled: int,
                    focused_threshold: float,
                    focused_rands: int,
                    focused_depth: int,
                    regression_threshold: int,
                    weak_threshold: int,
                    validation_samples: int,
                    validation_threshold: float,):
        self.cycle = TrainingCycle(
            num_initial=num_initial,
            depth_controlled=depth_controlled,
            focused_threshold=focused_threshold,
            focused_rands=focused_rands,
            focused_depth=focused_depth,
            regression_threshold=regression_threshold,
            weak_threshold=weak_threshold,
            validation_samples=validation_samples,
            validation_threshold=validation_threshold,
        )

    def make_guess(self) -> mt.Guess:
        values: Dict[str, int] = {}
        for var in self.environment.env_variables:
            values[var.name] = random.randint(var.min_val, var.max_val)
        return mt.Guess(values=values)

    def make_controlled_guess(self, base_obs: mt.Observation, varied_var: str, used_values: set) -> mt.Guess:
        values: Dict[str, int] = {}
        for var in self.environment.env_variables:
            if var.name == varied_var:

                #If no more values left in domain, return None to warn functions domain is exhausted
                remaining = [v for v in range(var.min_val, var.max_val + 1)if v not in used_values]
                if not remaining:
                    return None

                candidate = random.choice(remaining)
                used_values.add(candidate)
                values[var.name] = candidate
            else:
                values[var.name] = base_obs.inputs[var.name]

        return mt.Guess(values=values)

    def conduct_initials(self) -> List[mt.Observation]:
        initials: List[mt.Observation] = []
        for i in range(self.cycle.num_initial):
            guess = self.make_guess()
            output = self.environment.evaluate(guess)
            observation = mt.Observation(inputs=guess.values, output=output)
            initials.append(observation)

            """Printer call"""
            self.printer.print_initials(i, observation)

        return initials

    """Chooses num_controlled initial observations, limited by number of initial observations, then
    produces depth_controlled observations with a random variable varied"""
    def conduct_controlled(self, initials: List[mt.Observation]) -> List[mt.ControlledGroup]:
        controlled: List[mt.ControlledGroup] = []

        #Shuffle initials for iteration
        initial_shuffled = initials[:]
        random.shuffle(initial_shuffled)

        #Shuffle environmental variables for iteration
        env_vars_names: List[str] = [var.name for var in self.environment.env_variables]
        random.shuffle(env_vars_names)

        num_vars = len(env_vars_names)

        for i, controlled_initial in enumerate(initial_shuffled):
            varied_var = env_vars_names[i % num_vars]

            used_values = {controlled_initial.inputs[varied_var]}
            batch_observations: List[mt.Observation] = []

            """Printer call"""
            self.printer.print_controlled_header(i, varied_var)

            for j in range(self.cycle.depth_controlled):
                guess = self.make_controlled_guess(controlled_initial, varied_var, used_values)

                #Domain is exhausted
                if guess is None:
                    break

                output = self.environment.evaluate(guess)

                observation = mt.Observation(inputs=guess.values, output=output)
                batch_observations.append(observation)

                """Printer Call"""
                self.printer.print_controlled(i=i, depth=j, obs=observation)

            controlled.append(
                mt.ControlledGroup(varied_var=varied_var,
                                observations=batch_observations,)
            )

        self.printer.print_emptyline()
        return controlled

    def conduct_entropy_scores(self, controlled_groups: List[mt.ControlledGroup]) -> List[float]:
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

    """=== RULE BOUNDARY SELECTION BEGINS ==="""
    def conduct_fragment_sweep(self, sweep: mt.OrderedSweep) -> List[mt.Fragment]:
        xs = [x for x, _ in sweep.trace]
        ys = [y for _, y in sweep.trace]

        if len(xs) < 2:
            return []

        delta_list = [ys[i+1] - ys[i] for i in range(len(ys) - 1)]

        #Change signature
        def inner(sign: int):
            if sign > 0:
                return 1
            if sign < 0:
                return -1
            else:
                return 0

        fragments: List[mt.Fragment] = []

        start = 0
        previous_sign = inner(delta_list[0])
        previous_zero = delta_list[0] == 0

        for i in range(1, len(delta_list)):
            current_sign = inner(delta_list[i])
            current_zero = (delta_list[i] == 0)

            behavioural_change = ((current_zero != previous_zero) or (current_sign != previous_sign))

            #Point of change, close previous fragment
            if behavioural_change:
                samples = sweep.trace[start:(i+1)]
                fragments.append(mt.Fragment(
                    varied_var = sweep.varied_var,
                    context = sweep.context,
                    interval = (samples[0][0], samples[-1][0]),
                    samples = samples
                ))
                start = i

            previous_sign = current_sign
            previous_zero = current_zero

        #Near identical case for final fragment with return
        samples = sweep.trace[start:]
        fragments.append(mt.Fragment(
            varied_var=sweep.varied_var,
            context=sweep.context,
            interval=(samples[0][0], samples[-1][0]),
            samples=samples
        ))

        return fragments

    def conduct_probe_traces(self) -> list[mt.OrderedSweep]:
        probe_traces = []

        probe_variables = self._choose_probe_variables()
        variable_contexts = self._choose_probe_contexts(probe_variables)

        for var, contexts in variable_contexts.items():
            for observation in contexts:
                x = self._build_ordered_sweep(var, observation)
                probe_traces.append(x)

        return probe_traces

    """Graphical approach designed to propose rule candidates while
    overcoming greedy merging implementation."""
    def cluster_fragments(self, fragments: List[mt.Fragment]) -> List[mt.RuleCandidate]:
        clusterable = self._filter_border_fragments(fragments)

        if not clusterable:
            return []

        graph = self._build_fragment_graph(clusterable)
        components = self._connected_components(graph)
        rule_candidates: List[mt.RuleCandidate] = []

        for component in components:
            fragments = [clusterable[i] for i in component]
            rule_candidates.append(mt.RuleCandidate(
                varied_var = fragments[0].varied_var,
                signature = fragments[0].signature,
                fragments = fragments,
            ))

        return rule_candidates

    def extract_rule_candidate_observations(self, rule: mt.RuleCandidate) -> list[mt.Observation]:
        observations: list[mt.Observation] = []

        for fragment in rule.fragments:
            for x, y in fragment.samples:
                inputs = dict(fragment.context)
                inputs[fragment.varied_var] = x

                observations.append(mt.Observation(inputs=inputs, output=y))

        return observations

    """Rule boundary selection helpers"""
    def _build_ordered_sweep(self, var: str, base_obs: mt.Observation) -> mt.OrderedSweep:
        focused_variable = self.var_by_name[var]

        #Generates the isolated context
        context = dict(base_obs.inputs)
        context.pop(var)

        sweep: list[tuple[int, int]] = []

        for x in range(focused_variable.min_val, focused_variable.max_val + 1):
            inputs = dict(base_obs.inputs)
            inputs[var] = x

            guess = mt.Guess(values=inputs)
            y = self.environment.evaluate(guess)

            sweep.append((x,y))

        return mt.OrderedSweep(
            varied_var=var,
            context=context,
            trace=sweep,
        )

    def _choose_probe_variables(self) -> set[str]:
        assert self.cycle.controlled_groups is not None
        assert self.cycle.entropy_scores is not None

        probing_variables: set[str] = set()

        for group, score in zip(self.cycle.controlled_groups, self.cycle.entropy_scores):
            if score > self.cycle.focused_threshold:
                probing_variables.add(group.varied_var)

        return probing_variables

    def _choose_probe_contexts(self, probe_variables: set[str]) -> Dict[str, List[mt.Observation]]:
        assert self.cycle.controlled_groups is not None
        assert self.cycle.entropy_scores is not None
        assert self.cycle.initial_observations is not None

        variable_contexts: Dict[str, List[mt.Observation]] = {}

        for group, score in zip(self.cycle.controlled_groups, self.cycle.entropy_scores):
            if score < self.cycle.focused_threshold:
                continue

            var = group.varied_var
            if var not in probe_variables:
                continue
            if not group.observations:
                continue

            representative = random.choice(group.observations)
            variable_contexts.setdefault(var, []).append(representative)

        for var in probe_variables:
            contexts = variable_contexts.setdefault(var, [])
            for _ in range (self.cycle.focused_rands):
                guess = self.make_guess()
                output = self.environment.evaluate(guess)
                observation = mt.Observation(inputs=guess.values, output=output)
                contexts.append(observation)

        return variable_contexts

    """Signature related helper methods"""
    def _fragment_compatibility(self, frag1: mt.Fragment, frag2: mt.Fragment) -> bool:
        #Classic compatibility check - same varied_var + signature
        if frag1.varied_var != frag2.varied_var:
            return False
        if frag1.signature != frag2.signature:
            return False

        #Constant fragments consistency - identical value
        if frag1.signature == mt.BehaviourSignature.CONSTANT:
            return frag1.samples[0][1] == frag2.samples[0][1]

        #Editing note - may need to attend for later invariants (for linear ETC)
        return True

    def _assign_signature(self, fragment: mt.Fragment) -> mt.BehaviourSignature:
        ys = [y for _, y in fragment.samples]

        #Fragment describes a boundary, will be filtered out
        if len(ys) < 3:
            return mt.BehaviourSignature.DISCONTINUOUS

        deltas = [ys[i + 1] - ys[i] for i in range(len(ys) - 1)]

        if all(delta == 0 for delta in deltas):
            return mt.BehaviourSignature.CONSTANT

        #Computation of second differences to determine linearity
        second_deltas = [
            deltas[i + 1] - deltas[i]
            for i in range(len(deltas) - 1)
        ]

        #Compute slope constant to determine if linear or non-linear
        if all(second_delta == 0 for second_delta in second_deltas):
            if deltas[0] > 0:
                return mt.BehaviourSignature.LINEAR_POSITIVE
            else:
                return mt.BehaviourSignature.LINEAR_NEGATIVE

        #Monotonicity, even though subtraction is not in pipeline
        monotonicity = (
            all(delta >= 0 for delta in deltas) or
            all(delta <= 0 for delta in deltas)
        )
        if monotonicity:
            return mt.BehaviourSignature.MONOTONE_NONLINEAR
        else:
            return mt.BehaviourSignature.NOT_MONOTONE_NONLINEAR

        # #Some kind of stable curve exists
        # return FragmentSignature.NONLINEAR

    def _filter_border_fragments(self, fragments: List[mt.Fragment]) -> List[mt.Fragment]:
        """Remove all discontinuous fragments for symbolic regression,
        these are the border cases of len == 2"""
        return [
            frag for frag in fragments if frag.signature != mt.BehaviourSignature.DISCONTINUOUS
        ]

    """Takes fragments across contexts and relates them on a graph
    to propose which fragments may represent the same rule based on their signature."""
    def _build_fragment_graph(self, fragments: List[mt.Fragment]) -> dict[int, set[int]]:
        #Indexed dictionary proposing compatible fragments
        potential_graph: dict[int, set[int]] = {i: set() for i in range(len(fragments))}

        for i in range(len(fragments)):
            for j in range(i + 1, len(fragments)):
                if self._fragment_compatibility(fragments[i], fragments[j]):
                    potential_graph[i].add(j)
                    potential_graph[j].add(i)
        return potential_graph

    """Determine multi-fragment connected groups from the graph"""
    def _connected_components(self, graph: dict[int, set[int]]) -> list[list[int]]:
        visited = set()

        """Every fragment in each internal list is compatible via some chain
        of relation with every other such fragment."""
        components: list[list[int]] = []

        for node in graph:
            if node in visited:
                continue

            stack = [node]
            component = []

            while stack:
                current = stack.pop()
                if current in visited:
                    continue

                visited.add(current)
                component.append(current)
                stack.extend(graph[current] - visited)

            components.append(component)

        """Essentially, our proposed rule candidates as lists of fragments"""
        return components

    """=== RULE BOUNDARY SELECTION ENDS ==="""

    """"Candidate rule to proto-rule formation"""
    def conduct_symbolic_regression(self, rule_candidates: list[mt.RuleCandidate]) -> List[mt.ProposedRule]:
        results: List[mt.ProposedRule] = []

        symreg = SymbolicRegressor(self.var_by_name)

        for candidate in rule_candidates:
            observations = self.extract_rule_candidate_observations(candidate)

            if len(observations) < self.cycle.regression_threshold:
                continue

            conditions = self.determine_conditions(
                observations,
                candidate.varied_var,
            )

            varied_values = [
                obs.inputs[candidate.varied_var]
                for obs in observations
            ]

            conditions[candidate.varied_var] = self._values_to_ranges(varied_values)

            suitable = self._determine_suitable_regression(
                observations,
                candidate.varied_var,
            )

            # Regression variables = varied var + admissible others
            regression_vars = [candidate.varied_var] + suitable

            expression = symreg.regress(
                observations=observations,
                names=regression_vars,
                varied_var=candidate.varied_var,
            )

            signature = candidate.signature
            if any(frag.signature != signature for frag in candidate.fragments):
                raise ValueError(f"Fragments have inconsistent signatures! Abort!")

            results.append(
                mt.ProposedRule(
                    varied_var = candidate.varied_var,
                    contributing_vars = {candidate.varied_var},
                    conditions = conditions,
                    expression = expression,
                    signature = signature,
                )
            )

        return results

    #Takes variable values and computes to ranges if applicable
    def determine_conditions(self, observations: list[mt.Observation], varied_var: str):
        if not observations:
            return {}

        conditions: dict[str, List[tuple[int, int]]] = {}

        for variable in observations[0].inputs.keys():
            if variable == varied_var:
                continue

            values = [observation.inputs[variable] for observation in observations]
            conditions[variable] = self._values_to_ranges(values)

        return conditions

    #Move to rule processing to help fix rule representation!
    def _values_to_ranges(self, values: list[int]) -> list[tuple[int, int]]:
        if not values:
            return []

        values = sorted(set(values))
        ranges: list[tuple[int, int]] = []

        start = values[0]
        previous = values[0]
        for value in values[1:]:
            if value == previous + 1:
                previous = value
                continue

            ranges.append((start, previous))
            start = value
            previous = value

        ranges.append((start, previous))
        return ranges

    """As we already presume that a proposed rule operates within a single conditional
    we can prevent symbolic regression from using variables that do not cause
    change in the fragment when varied. This prevents using variables as convenient algebraic props
    and keeps the tree clean!"""
    def _determine_suitable_regression(self, observations: list[mt.Observation], varied_var: str) -> list[str]:
        by_v = defaultdict(list)
        output = []
        for observation in observations:
            by_v[observation.inputs[varied_var]].append(observation)

        for variable_name in observations[0].inputs.keys():
            if variable_name == varied_var:
                continue

            for v_val, observation_list in by_v.items():
                visited = {}

                for observation in observation_list:
                    x = observation.inputs[variable_name]
                    y = observation.output

                    if x in visited:
                        continue
                    visited[x] = y

                if len(set(visited.values())) >= 2:
                    output.append(variable_name)
                    break

        return output

    """Combines weak rules with the same varied_var and its respective range.
    Assumption of uni-law membership is broken by this technique. However, uses
    otherwise useless fragments cheaply."""
    def pool_weak_rules(self, rule_candidates: List[mt.RuleCandidate]):
        weaklings: list[mt.RuleCandidate] = []
        stronglings: list[mt.RuleCandidate] = []

        for candidate in rule_candidates:
            context: dict[str, set[int]] = {}
            for fragment in candidate.fragments:
                for variable, value in fragment.context.items():
                    context.setdefault(variable, set()).add(value)
            if context:
                deepest_var = max(len(values) for values in context.values())
            else:
                deepest_var = 0

            if deepest_var < self.cycle.weak_threshold:
                weaklings.append(candidate)
            else:
                stronglings.append(candidate)

        return stronglings + self._pool_weak_helper(weaklings)

    """Helper to process 'weaklings' because otherwise return is ugly."""
    def _pool_weak_helper(self, weaklings: list[mt.RuleCandidate]):
        grouped: dict[tuple[str, tuple[int, int]], list[mt.RuleCandidate]] = {}

        for candidate in weaklings:
            my_key = (candidate.varied_var, candidate.signature, self._determine_interval(candidate))
            grouped.setdefault(my_key, []).append(candidate)

        pooled_candidates: list[mt.RuleCandidate] = []

        for group in grouped.values():
            pooled_fragments = []
            for candidate in group:
                pooled_fragments.extend(candidate.fragments)

            pooled_candidates.append(
                mt.RuleCandidate(
                    varied_var=group[0].varied_var,
                    signature=group[0].signature,
                    fragments=pooled_fragments,
                )
            )

        return pooled_candidates

    def _determine_interval(self, candidate: mt.RuleCandidate) -> tuple[int, int]:
        lo = min(f.interval[0] for f in candidate.fragments)
        hi = max(f.interval[1] for f in candidate.fragments)
        return (lo, hi)