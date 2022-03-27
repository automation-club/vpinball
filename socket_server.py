import random
import zmq
import threading
import keyboard
import os
import torch
import numpy as np

from pathlib import Path
from zmq.sugar.socket import Socket
from torch import nn
from collections import deque


class SimpleDQN(nn.Module):
    def __init__(self, observation_space, action_space, seed):
        super().__init__()
        self.observation_space = observation_space
        self.action_space = action_space
        self.set_random_seeds(seed)
        self.main = self.create_agent()
        self.target = self.create_agent()
        self.replay_buffer = deque(maxlen=50000)

    def create_agent(self):
        return nn.Sequential(
            nn.Linear(self.observation_space, 64),
            nn.ReLU(),
            nn.Linear(64, 128),
            nn.ReLU(),
            nn.Linear(128, self.action_space),
        )

    def set_random_seeds(self, seed):
        torch.manual_seed(seed)
        random.seed(seed)
        np.random.seed(seed)

def fire_plunger(socket):
    socket.send("P".encode())
    for i in range(240):
        socket.recv()
        socket.send("".encode())

    socket.recv()
    socket.send("N".encode())
    for i in range(120):
        socket.recv()
        socket.send("".encode())


def handle_socket_server(socket: Socket):
    # Observation Space: Pos X, Y, Z & Vel X, Y, Z
    # Action Space: L Flipper, R Flipper, Both Flippers, No Flippers
    dqn = SimpleDQN(observation_space=6, action_space=4, seed=1)

    # DQN Parameters
    MAX_EPSILON = 1
    MIN_EPSILON = 0.1
    DECAY_RATE = 0.01

    # if np.random.rand() < epsilon

    i = 0
    while True:
        i+=1
        #  Wait for next request from client
        message = socket.recv().decode()
        client_request = message.split(",")  # Observations: PosX, PosY, PosZ, VelX, VelY, VelZ
        print(f"[RECEIVED FROM CLIENT]: {client_request} {i}")

        if client_request[0] == "BALL CREATED":
            fire_plunger(socket)
        elif client_request[0] == "BALL DESTROYED":
            socket.send("G".encode()) #  Start new game
        elif client_request[0] == "NOTHING":
            socket.send("G".encode())
        elif client_request[0] == "BALL POS":
            # Decide Action
            action = "N"
            # if i % 30 == 0:
            #     action_space = ["L", "R", "B", "N", "P"]
            #     action = random.choice(action_space)

            # Send reply back to client
            socket.send(action.encode())


def detect_keypress():
    while True:
        if keyboard.is_pressed('q'):
            print("Shutdown key detected.")
            break


def launch_visual_pinball(path):
    os.system(path)


def main():
    # Config
    VISUAL_PINBALL_EXE_PATH = Path("./x64/Debug/VPinballX.exe -EnableSockets")

    # Bind TCP socket and listen for clients
    context = zmq.Context()
    socket = context.socket(zmq.REP)
    socket.bind("tcp://*:5555")
    print("Socket bound to tcp://localhost:5555")

    # Thread for handling client requests
    server_thread = threading.Thread(target=handle_socket_server, args=(socket,), daemon=True)
    # Thread for handling shutdown (press Q)
    server_shutdown_thread = threading.Thread(target=detect_keypress, args=())
    # Thread for starting Visual Pinball
    # Pass in lanuch args to VPinball
    
    launch_vp_thread = threading.Thread(target=launch_visual_pinball, args=(str(VISUAL_PINBALL_EXE_PATH),))

    # Start Threads
    server_thread.start()
    server_shutdown_thread.start()
    launch_vp_thread.start()

    # Run once shutdown key detected
    server_shutdown_thread.join()
    print("Ending program.")
    exit(0)
    # Socket close handled by garbage collector


if __name__ == "__main__":
    main()
