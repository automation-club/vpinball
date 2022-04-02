import numpy as np


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
            message_type = message[0]
            observation = message[1:]

            # Process observation and take next action accordingly
            if message_type == "BALL INFO":  # Ball is in play
                action = self._choose_action(observation)
                self._server.send_message(action)
            elif message_type == "BALL CREATED":  # New ball put into play
                self._launch_ball()
            elif message_type == "NO BALL ON FIELD":  # No active ball on the field
                self._start_new_game()
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
        elif self._mode == "real_player":
            return ""
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

    def _launch_ball(self):
        """
        Launches a ball into play.

        Returns
        -------
        None
        """
        self._server.send_message("P")
        for i in range(120):
            self._server.receive_message()
            self._server.send_message("")

        self._server.receive_message()
        self._server.send_message("N")
        for i in range(60):
            self._server.receive_message()
            self._server.send_message("")

    def _start_new_game(self):
        """
        Starts a new game.

        Returns
        -------
        None
        """
        # Input first credit
        self._server.send_message("C")
        for i in range(90):  # Wait for credit to register
            if self._check_ball_created(""):
                return
        # Input second credit
        if self._check_ball_created("C"):
            return
        for i in range(200):  # Wait for credit to register
            if self._check_ball_created(""):
                return

        # Key down on start key
        if self._check_ball_created("s"):
            return
        # Key up on start key
        if self._check_ball_created("S"):
            return

        # Wait for ball to be created
        while True:
            if self._check_ball_created(""):
                return

    def _check_ball_created(self, message):
        """
        Checks if a ball has been created.

        Parameters
        ----------
        message : str
            The message to send to the game.

        Returns
        -------
        ok : bool
            True if a ball has been created, False otherwise.
        """

        if self._server.receive_message() == "BALL CREATED":
            self._launch_ball()
            return True
        self._server.send_message(message)
        return False
