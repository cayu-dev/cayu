from examples._advanced_support.cli import run_cli
from examples.bounded_fork_group.deterministic import run as deterministic
from examples.bounded_fork_group.live import run as live

if __name__ == "__main__":
    run_cli(deterministic=deterministic, live=live)
