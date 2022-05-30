import threading
import numpy as np

from utils import config
from utils.socket_server import SocketServer
from utils.pinball_player import PinballPlayer


def train_dqn(mode, server):
    np.random.seed(123)
    env = PinballPlayer(mode=config.DECISION_MODE, socket_server=server)
    while True:
        observation = None
        while observation is None:
            observation = env.observe()

        action = np.random.choice(env.action_space)
        env.step(action)


def main():
    # Initialize the socket server
    server = SocketServer(config.PORT)

    # Thread for handling playing pinball
    player_thread = threading.Thread(target=train_dqn, args=(config.DECISION_MODE, server), daemon=True)
    # Thread for stopping program
    stop_thread = threading.Thread(target=SocketServer.stop_server)
    # Thread for launching VPinball
    launch_thread = threading.Thread(target=SocketServer.launch_vpinball, args=(config.VPINBALL_EXECUTABLE_PATH,))

    # Start the threads
    player_thread.start()
    stop_thread.start()
    launch_thread.start()

    # Wait for the stop server thread to finish
    stop_thread.join()
    print("Exiting...")
    exit(0)


if __name__ == "__main__":
    main()
