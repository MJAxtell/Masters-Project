from my_types import Observation

""""Null object pattern"""
class Printer:
    def print_newline(self):
        pass

    def begin_cycle(self, cycle) -> None:
        pass

    def print_initials(self, i, obs) -> None:
        pass

    def print_controlled_header(self, i, locked_var):
        pass

    def print_controlled(self, i, depth, obs) -> None:
        pass

class MainPrinter(Printer):
    def print_emptyline(self) -> None:
        print("")

    def begin_cycle(self, cycle) -> None:
        print("=== NEW TRAINING CYCLE ===")
        print(
            f"Number of initial experiments: {cycle.num_initial}\n",
            f"Number of controlled experiments: {cycle.num_controlled}\n",
            f"Controlled experiment depth: {cycle.depth_controlled}\n",
        )

    def print_initials(self, i, obs) -> None:
        print(f"    Initial {i}: inputs {obs.inputs}, output {obs.output}")

    def print_controlled_header(self, i, locked_var) -> None:
        print(f"~ Controlled set {i}, controlling {locked_var} ~")

    def print_controlled(self, i, depth, obs) -> None:
        print(f"    Controlled experiment {i}|{depth}: inputs {obs.inputs}, output {obs.output}")