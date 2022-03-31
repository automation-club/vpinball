import threading
import config
import os

from SocketServer import SocketServer
from PinballPlayer import PinballPlayer


def main():
    # Initialize the socket server
    server = SocketServer(config.PORT)

    # Launch the VPinball executable
    os.system(f"{config.VPINBALL_EXECUTABLE_PATH} -EnableSockets")

    # Thread for handling playing pinball
    player_thread = threading.Thread(target=PinballPlayer, args=(config.DECISION_MODE, server), daemon=True)
    # Thread for stopping program
    stop_thread = threading.Thread(target=server.stop_server)

    # Start the threads
    player_thread.start()


if __name__ == "__main__":
    main()
