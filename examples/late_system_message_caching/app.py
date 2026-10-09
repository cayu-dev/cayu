from examples._advanced_support.cli import run_cli
from examples.late_system_message_caching.deterministic import run as deterministic
from examples.late_system_message_caching.live import run as live

if __name__ == "__main__":
    run_cli(deterministic=deterministic, live=live)
