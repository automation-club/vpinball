import os

import zmq
import keyboard


class SocketServer:
    """
    Handles communication with the client at specified port through TCP connection


    Methods
    -------
    send_message(message="L")
        Sends message to the client
    receive_message()
        Receives message from the client

    """
    def __init__(self, port):
        """
        Parameters
        ----------
        port : int
            Port number to connect to

        """
        self._socket = self._establish_server(port)

    def _establish_server(self, port):
        """
        Establishes a TCP connection with the client

        Parameters
        ----------
        port : int
            Port number to connect to

        Returns
        -------
        socket : zmq.sugar.socket.Socket
            Socket to communicate with the client

        """
        context = zmq.Context()
        socket = context.socket(zmq.REP)
        socket.bind(f"tcp://*:{port}")
        print(f"Socket bound to tcp://localhost:{port}")

        return socket

    def send_message(self, message):
        """
        Sends a message to the client

        Parameters
        ----------
        message : str
            Message to send to the client

        """
        self._socket.send_string(message)

    def receive_message(self):
        """
        Receives a message from the client

        Returns
        -------
        message : str
            Message received from the client

        """
        message = self._socket.recv_string()

        return message

    @staticmethod
    def stop_server():
        """
        Stops the server

        Returns
        -------
        None

        """
        while True:
            if keyboard.is_pressed('q'):
                print("Shutdown key detected. Shutting down server...")
                break

    @staticmethod
    def launch_vpinball(path):
        """
        Launches vpinball

        Returns
        -------
        None

        """
        os.system(f"{path} -EnableSockets")
