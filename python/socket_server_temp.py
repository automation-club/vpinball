import random
import zmq
import threading
import keyboard
import os
import torch
import numpy as np
import pandas as pd

from pathlib import Path
from zmq.sugar.socket import Socket
from torch import nn
from collections import deque
from torch.utils.data import Dataset, DataLoader


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
    for i in range(120):
        socket.recv()
        socket.send("".encode())

    socket.recv()
    socket.send("N".encode())
    for i in range(60):
        socket.recv()
        socket.send("".encode())


# dont worry about how this works or what it does
# just trust that it works (until someone finds the fucking game end call)
def start_new_game(socket):
    socket.send("C".encode())
    for i in range(90):
        socket.recv()
        socket.send("".encode())
    socket.recv()
    socket.send("C".encode())
    for i in range(200):
        message = socket.recv().decode()
        socket.send(f"{i}".encode())
        if message == "BALL CREATED":
            socket.recv()
            fire_plunger(socket)
            return
    message = socket.recv().decode()
    if message == "BALL CREATED":
        fire_plunger(socket)
        return
    socket.send("s".encode())
    message = socket.recv().decode()
    if message == "BALL CREATED":
        fire_plunger(socket)
        return
    socket.send("S".encode())
    while True:
        message = socket.recv().decode()
        socket.send("".encode())
        if message == "BALL CREATED":
            socket.recv()
            fire_plunger(socket)
            break


# Parse the fucking file
def parse_file(file_path):
    columns = ["_", "X", "Y", "Z", "VelX", "VelY", "VelZ", "Action"]
    df = pd.read_table(file_path, sep=",", names=columns)
    df.drop(columns=["_"], inplace=True)
    df["Action"] = df["Action"].astype('category').cat.codes
    torch_tensor = torch.tensor(df.values)

    return torch_tensor
    # return df[["X", "Y", "Z", "VelX", "VelY", "VelZ"]], df[["Action"]]


class DatasetFromTXT(Dataset):
    def __init__(self, data):
        self.data = data

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        return self.data[idx, 0:6], self.data[idx, 6]


def test():
    data = parse_file("./runs/experience-learning.txt")
    dataset = DatasetFromTXT(data)
    loader = DataLoader(dataset, batch_size=32, shuffle=True)

    for x,y in loader:
        print(x.shape)
        break


def handle_socket_server(socket: Socket):
    # Observation Space: Pos X, Y, Z & Vel X, Y, Z
    # Action Space: L Flipper, R Flipper, Both Flippers, No Flippers
    dqn = SimpleDQN(observation_space=6, action_space=4, seed=1)

    # DQN Parameters
    MAX_EPSILON = 1
    MIN_EPSILON = 0.1
    DECAY_RATE = 0.01

    # if np.random.rand() < epsilon

    while True:
        #  Wait for next request from client
        message = socket.recv().decode()
        client_request = message.split(",")  # Observations: PosX, PosY, PosZ, VelX, VelY, VelZ
        print(f"[RECEIVED FROM CLIENT]: {client_request}")

        if client_request[0] == "BALL CREATED":
            fire_plunger(socket)

        elif client_request[0] == "BALL DESTROYED":
            socket.send("".encode())

        elif client_request[0] == "NOTHING":
            start_new_game(socket)  # Start new game

        elif client_request[0] == "BALL POS":
            # Decide Action
            action = ""
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
    # main()
    test()

