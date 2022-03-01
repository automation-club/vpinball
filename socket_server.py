import random
import zmq
import threading
import keyboard
import os

from pathlib import Path
from zmq.sugar.socket import Socket


def handle_socket_server(socket: Socket):
    i = 0
    while True:
        #  Wait for next request from client
        message = socket.recv().decode()
        observations = message.split(",")  # Observations: PosX, PosY, PosZ, VelX, VelY, VelZ
        print(f"[RECEIVED FROM CLIENT]: {observations}")

        # Decide Action
        action = None
        if i % 30 == 0:
            action_space = ["L", "R", "B", "N"]
            action = random.choice(action_space)

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
    VISUAL_PINBALL_EXE_PATH = Path("./x64/Debug/VPinballX.exe")

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
    launch_vp_thread = threading.Thread(target=launch_visual_pinball, args=(VISUAL_PINBALL_EXE_PATH,))

    # Start Threads
    server_thread.start()
    server_shutdown_thread.start()
    launch_vp_thread.start()

    # Run once shutdown key detected
    server_shutdown_thread.join()
    print("Ending program.")

    # Socket close handled by garbage collector


if __name__ == "__main__":
    main()
