from pathlib import Path

VPINBALL_EXECUTABLE_PATH = Path("../x64/Debug/VPinball.exe")
"""Path: Path to the VPinball executable."""

# Socket Configuration
PORT = 5555
"""int: Port to use for the socket."""

# Pinball Agent Settings
DECISION_MODE = "random"
"""str: The decision mode to use (Random, DQN Agent, Classification Model)."""

