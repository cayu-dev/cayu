from examples._advanced_support.cli import run_cli
from examples.cache_aware_research_council.deterministic import run as deterministic
from examples.cache_aware_research_council.live import run as live

if __name__ == "__main__":
    run_cli(deterministic=deterministic, live=live)
