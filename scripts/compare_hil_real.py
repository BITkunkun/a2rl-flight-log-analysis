"""Compare HIL and REAL logs through the shared pipeline."""
from analysis_core.runner import main

if __name__ == "__main__":
    main(("HIL", "REAL"))
