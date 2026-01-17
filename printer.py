from my_types import Observation, ClusteredHypotheses
from typing import Dict

""""Null object pattern"""
class Printer:
    def print_emptyline(self):
        print("\n")

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

    def print_proposed_rules(self, rules):
        pass

    def print_rule_validation(self, rules, validation_stats):
        pass

    def print_merged_rule_validation(self, rules, validation_stats):
        pass

    def print_identical_merged(self, merged_rules):
        pass

    def global_merged(self, merged_rule):
        pass

    def print_global_identical_merged(self, merged_rules):
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
        print("\n")
        print("🌱🔴 === NEW TRAINING CYCLE === 🔴🌱")
        print(
            f"Number of initial experiments: {cycle.num_initial}\nControlled experiment depth: {cycle.depth_controlled}\n",
        )

    def print_initials(self, i, obs):
        print(f"    Initial {i}: inputs {obs.inputs}, output {obs.output}")

    def print_controlled_begin(self):
        print(f"\n= CONTROLLED EXPERIMENTS =")

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

    def print_proposed_rules(self, rules):
        print(f"\n= PROPOSED RULES =")

        if not rules:
            print("No rules identified!")
            return

        for i, rule in enumerate(rules):
            print(f"Rule {i}, Variable {rule.varied_var}")

            print("    Conditions:")
            for var, ranges in rule.conditions.items():
                print(f"        {var}: {ranges}")

            print("    Expression:")
            print(f"        {self.print_expression(rule.expression)}")

    def print_rule_validation(self, rules, validation_stats):
        print("\n= RULE VALIDATION =")

        if not rules:
            print("No rules validated!")
            return

        for i, (rule, stats) in enumerate(zip(rules, validation_stats)):
            passed, accuracy = stats
            status = "PASSED" if passed else "FAILED"

            print(f"Rule {i}, Variable {rule.varied_var}")
            print(f"    Validation: {status}")
            print(f"        Accuracy: {accuracy:.3f}")

    def print_merged_rule_validation(self, rules, validation_stats):
        print("\n= MERGED RULE VALIDATION =")

        if not rules:
            print("No rules validated!")
            return

        for i, (rule, stats) in enumerate(zip(rules, validation_stats)):
            passed, accuracy = stats
            status = "PASSED" if passed else "FAILED"

            print(f"Rule {i}, Variable {rule.varied_var}")
            vars_str = ", ".join(sorted(rule.contributing_vars))
            print(f"    Contributing vars {i}, [{vars_str}]")
            print(f"    Validation: {status}")
            print(f"        Accuracy: {accuracy:.3f}")

    def print_identical_merged(self, merged_rules):
        print("\n= IDENTICAL MERGE =")

        if not merged_rules:
            print("Identical merge produced no rules!")
            return

        for i, rule in enumerate(merged_rules):
            print(f"Rule {i}, Contributing Variables {rule.contributing_vars}")

            if rule.conditions:
                print("    Conditions:")
                for var, ranges in rule.conditions.items():
                    print(f"        {var}: {ranges}")
            else:
                print("    Conditions: <none>")

            print("    Expression:")
            print(f"        {self.print_expression(rule.expression)}")

            print(f"    Signature: {rule.signature}")

    def print_global_merged(self, merged_rules):
        print("\n= GLOBAL MERGE =")

        if not merged_rules:
            print("Identical merge produced no rules!")
            return

        for i, rule in enumerate(merged_rules):
            print(f"Rule {i}, Contributing Variables {rule.contributing_vars}")

            if rule.conditions:
                print("    Conditions:")
                for var, ranges in rule.conditions.items():
                    print(f"        {var}: {ranges}")
            else:
                print("    Conditions: <none>")

            print("    Expression:")
            print(f"        {self.print_expression(rule.expression)}")

            print(f"    Signature: {rule.signature}")

    def print_global_identical_merged(self, merged_rules):
        print("\n= GLOBAL IDENTICAL MERGE =")

        if not merged_rules:
            print("Identical merge produced no rules!")
            return

        for i, rule in enumerate(merged_rules):
            print(f"Rule {i}, Contributing Variables {rule.contributing_vars}")

            if rule.conditions:
                print("    Conditions:")
                for var, ranges in rule.conditions.items():
                    print(f"        {var}: {ranges}")
            else:
                print("    Conditions: <none>")

            print("    Expression:")
            print(f"        {self.print_expression(rule.expression)}")

            print(f"    Signature: {rule.signature}")