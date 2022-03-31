import threading
import config

from SocketServer import SocketServer
from PinballPlayer import PinballPlayer


def main():
    # Initialize the socket server
    server = SocketServer(config.PORT)

    # Thread for handling playing pinball
    player_thread = threading.Thread(target=PinballPlayer, args=(config.DECISION_MODE, server))

    # Launch threads
    player_thread.start()


if __name__ == "__main__":
    main()
