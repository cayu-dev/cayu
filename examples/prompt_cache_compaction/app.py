from examples._advanced_support.cli import run_cli
from examples.prompt_cache_compaction.deterministic import run as deterministic
from examples.prompt_cache_compaction.live import run as live

if __name__ == "__main__":
    run_cli(deterministic=deterministic, live=live)
