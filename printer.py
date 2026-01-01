from my_types import Observation

""""Null object pattern"""
class Printer:
    def print_emptyline(self):
        pass

    def begin_cycle(self, cycle):
        pass

    def print_initials(self, i, obs):
        pass

    def print_controlled_begin(self):
        pass

    def print_controlled_header(self, i, locked_var):
        pass

    def print_controlled(self, i, depth, obs):
        pass

    def print_entropy_scores(self, scores, vars):
        pass

    def print_focused(self, focused_groups):
        pass

class MainPrinter(Printer):
    def print_emptyline(self):
        print("")

    def begin_cycle(self, cycle):
        #No, the emojis are not ChatGPT generated. They help the header stand out.
        print("🌱🔴 === NEW TRAINING CYCLE === 🔴🌱")
        print(
            f"Number of initial experiments: {cycle.num_initial}\nControlled experiment depth: {cycle.depth_controlled}\n",
        )

    def print_initials(self, i, obs):
        print(f"    Initial {i}: inputs {obs.inputs}, output {obs.output}")

    def print_controlled_begin(self):
        print(f"= CONTROLLED EXPERIMENTS =")

    def print_controlled_header(self, i, locked_var):
        print(f"~ Controlled set {i}, controlling {locked_var} ~")

    def print_controlled(self, i, depth, obs):
        print(f"    Controlled experiment {i}|{depth}: inputs {obs.inputs}, output {obs.output}")

    def print_entropy_scores(self, scores, vars):
        print(f"= ENTROPY SCORES =")
        for i, score in enumerate(scores):
            print(f"    Controlled Set {i}, variable {vars[i]}: {score}")

    def print_focused(self, focused_groups):
        print(f"= FOCUSED EXPERIMENTATION =")

        if not focused_groups:
            print("No focused groups identified!")
            return

        context_depth = len(focused_groups[0].observations[0])
        print(f"Depth: {context_depth}")

        for i, group in enumerate(focused_groups):
            var = group.varied_var

            num_contexts = len(group.observations)
            total_obs = num_contexts * context_depth

            print(
                f"Focused Group {i}, variable {var}: "
                f"{num_contexts} contexts | {total_obs} observations"
            )