import random
import time

from zmq.sugar.socket import Socket

import zmq
import threading
import keyboard
import os

# Global Variables
msg_to_send = ""


def handle_socket_server(socket: Socket):
    global msg_to_send
    i = 0
    while True:
        #  Wait for next request from q client
        message = socket.recv().decode()
        observations = message.split(",")  # Observations: PosX, PosY, PosZ, VelX, VelY, VelZ
        print(f"[RECEIVED FROM CLIENT]: {observations}")

        # Decide Action
        action = None
        if i % 30 == 0:
            action_space = ["L", "R", "B", "N"]
            action = random.choice(action_space)

        #  Send reply back to client
        # print(f"{action} sent")
        socket.send(action.encode())


def detect_keypress():
    global msg_to_send
    while True:
        if keyboard.is_pressed('q'):
            print("Shutdown key detected.")
            break
        # if keyboard.is_pressed('l'):
        #     msg_to_send = "L"
        # if keyboard.is_pressed('r'):
        #     msg_to_send = "R"


def launch_visual_pinball():
    VISUAL_PINBALL_PATH = r".\x64\Debug\VPinballX.exe"

    os.system(VISUAL_PINBALL_PATH)


def main():
    global msg_to_send

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
    launch_vp_thread = threading.Thread(target=launch_visual_pinball, args=())

    # Start Threads
    server_thread.start()
    server_shutdown_thread.start()
    launch_vp_thread.start()

    server_shutdown_thread.join()
    # Run once shutdown key detected
    print("Ending program.")

    # Socket close handled by garbage collector


if __name__ == "__main__":
    main()
