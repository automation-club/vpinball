import numpy as np

from SocketServer import SocketServer


class PinballPlayer:
    """
    This class represents a player in a pinball game and carries out decision making in the game.

    Methods
    -------
    choose_action(observation=["0.0", "0.0", "0.0", "0.0", "0.0", "0.0"])
        Determines the action to take based on the observation and decision mode.

    """

    def __init__(self, mode, socket_server):
        """
        Parameters
        ----------
        mode : str
            The decision making mode.
        socket_server : SocketServer
            The socket server to communicate with the game.

        """
        self._mode = mode
        self._server = socket_server
        self._action_space = ['L', 'R', 'B', 'N']
        self._play_pinball()

    def _play_pinball(self):
        """
        Interacts with Visual Pinball to play pinball.

        Returns
        -------
        None
        """

        while True:
            # Get the observation from the game
            message = self._server.receive_message().split(",")

            # Process observation and take next action accordingly
            if message[0] == "BALL INFO":  # Ball is in play
                pass
            elif message[0] == "BALL CREATED":  # New ball put into play
                pass
            elif message[0] == "NO BALL ON FIELD":  # No active ball on the field
                pass
            else:  # Any other calls (not relevant)
                self._server.send_message("")  # Empty action response

    def _choose_action(self, observation):
        """
        Determines the action to take based on the observation and decision mode.

        Parameters
        ----------
        observation : list
            The observation of the game state.

        Returns
        -------
        str
            The action to take.

        """
        if self._mode == "random":
            return self._random_action()
        elif self._mode == "dqn_agent":
            return self._dqn_action(observation)
        else:
            print("Error: Unknown mode")

    def _random_action(self):
        """
        Randomly chooses an action from the action space.

        Returns
        -------
        str
            The action to take.
        """
        return np.random.choice(self._action_space)

    def _dqn_action(self, observation):
        """
        Determines the action to take based on the observation using a Deep Q Learning Agent

        Parameters
        ----------
        observation : list
            The observation of the game state.

        Returns
        -------
        str
            The action to take.
        """
        return None
