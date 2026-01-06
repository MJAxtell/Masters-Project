from my_types import Observation, ClusteredHypotheses

""""Null object pattern"""
class Printer:
    def print_emptyline(self):
        pass

    def print_expression(self, expression):
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

    def print_clustered_hypotheses(self, clustered):
        pass

class MainPrinter(Printer):
    def print_emptyline(self):
        print("")

    def print_expression(self, expression):
        if hasattr(expression, "value"):
            return str(expression.value)
        if hasattr(expression, "var"):
            return str(expression.var.name)

        #Assumes no only add and mul for left-right attributes
        if hasattr(expression, "left") and hasattr(expression, "right"):
            if expression.__class__.__name__ == "Add":
                return f"({self.print_expression(expression.left)} + {self.print_expression(expression.right)})"
            else:
                return f"({self.print_expression(expression.left)} * {self.print_expression(expression.right)})"

        #Assumes only pow for base and exponent attributes
        if hasattr(expression, "base") and hasattr(expression, "exponent"):
            return f"({self.print_expression(expression.base)} ** {expression.exponent})"

        raise TypeError(f"Expression Unknown: {type(expression)}")

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

        context_depth = len(focused_groups[0].clusters[0])
        print(f"Depth: {context_depth}")

        for i, group in enumerate(focused_groups):
            var = group.varied_var

            num_contexts = len(group.clusters)
            total_obs = num_contexts * context_depth

            print(
                f"Focused Group {i}, variable {var}: "
                f"{num_contexts} clusters | {total_obs} observations"
            )

    def print_clustered_hypotheses(self, primary_hypotheses):
        print(f"= PRIMARY HYPOTHESES =")

        if not primary_hypotheses:
            print("No hypotheses identified!")
            return

        for i, group in enumerate(primary_hypotheses):
            print(f"Hypothesis Group {i}, Variable {group.varied_var}")
            for j, expression in enumerate(group.hypotheses):
                print(f"    {i}, {self.print_expression(expression)}")