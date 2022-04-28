from pathlib import Path

VPINBALL_EXECUTABLE_PATH = Path("../x64/Debug/VPinballX.exe")
"""Path: Path to the VPinball executable."""

# Socket Configuration
PORT = 5555
"""int: Port to use for the socket."""
SOCKET_VERBOSE = True
"""bool: Whether to print socket messages."""

# Pinball Agent Settings
DECISION_MODE = "experience"
"""str: The decision mode to use (Random, DQN Agent, Classification Model)."""

