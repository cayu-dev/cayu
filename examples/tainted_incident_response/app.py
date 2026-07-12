from examples._advanced_support.cli import run_cli
from examples.tainted_incident_response.deterministic import run as deterministic
from examples.tainted_incident_response.live import run as live

if __name__ == "__main__":
    run_cli(deterministic=deterministic, live=live)
