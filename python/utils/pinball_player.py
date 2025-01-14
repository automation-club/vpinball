import numpy as np
import torch
import utils.config as config

from utils.models import Classifier


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
        self.action_space = ['L', 'R', 'B', 'N']
        self._model = self._load_model()
        # self._play_pinball()


    def step(self, action):
        self._server.send_message(action)

    def observe(self):
        # Get the observation from the game
        message = self._server.receive_message().split(",")
        message_type = message[0]
        observation = message[1:]
        observation = [float(x) for x in observation]

        # Print socket communication if enabled
        if config.SOCKET_VERBOSE:
            print("Received message: ", message)

        # Process observation and take next action accordingly
        if message_type == "BALL INFO":  # Ball is in play
            return observation
        elif message_type == "BALL CREATED":  # New ball put into play
            self._launch_ball()
            return "NEW GAME"
        elif message_type == "NO BALL ON FIELD":  # No active ball on the field
            self._start_new_game()
        else:  # Any other calls (not relevant)
            self._server.send_message("")  # Empty action response

        return None

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
            observation = [float(x) for x in observation]

            # Print socket communication if enabled
            if config.SOCKET_VERBOSE:
                print("Received message: ", message)

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

    def _load_model(self):
        """
        Loads the model from disk.

        Returns
        -------
        model : torch.nn.Module
            The model to use.
        """
        model = None
        if self._mode == "dqn_agent":
            model = torch.load("saved_models/dqn_model.pt")
        elif self._mode == "experience":
            model = Classifier(input_size=6, output_size=3, hidden_layers=3)
            model.model.load_state_dict(torch.load("saved_models/experience.pt"))

        if model is not None:
            model.eval()
        return model

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
        elif self._mode == "experience":
            return self._experience_replay_action(observation)
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
        return np.random.choice(self.action_space)

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

    def _experience_replay_action(self, observation):
        """
        Determines the action to take based on the observation using an experience replay agent

        Parameters
        ----------
        observation : list
            The observation of the game state.

        Returns
        -------
        str
            The action to take.
        """
        pred = self._model(torch.tensor(observation, dtype=torch.float32))
        action_idx = torch.argmax(pred).item()
        action = self._model.action_space[action_idx]
        if action != "N":
            print(action)
        return action

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
