from examples._advanced_support.cli import run_cli
from examples.counterfactual_approval.deterministic import run as deterministic
from examples.counterfactual_approval.live import run as live

if __name__ == "__main__":
    run_cli(deterministic=deterministic, live=live)
