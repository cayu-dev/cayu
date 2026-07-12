from examples._advanced_support.cli import run_cli
from examples.repo_maintainer_tournament.deterministic import run as deterministic
from examples.repo_maintainer_tournament.live import run as live

if __name__ == "__main__":
    run_cli(deterministic=deterministic, live=live)
