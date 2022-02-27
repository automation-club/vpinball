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
    while True:
        #  Wait for next request from q client
        message = socket.recv().decode()
        print(f"[RECEIVED FROM CLIENT]: {message}")

        #  Send reply back to client
        socket.send(msg_to_send.encode())
        if len(msg_to_send) > 0:
            pass
            # msg_to_send = ""


def detect_quit_keypress():
    while True:
        if keyboard.is_pressed('q'):
            print("Shutdown key detected.")
            break


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
    server_shutdown_thread = threading.Thread(target=detect_quit_keypress, args=())
    # Thread for starting Visual Pinball
    launch_vp_thread = threading.Thread(target=launch_visual_pinball, args=())

    # Start Threads
    server_thread.start()
    server_shutdown_thread.start()
    launch_vp_thread.start()

    time.sleep(10)

    msg_to_send = "L"

    time.sleep(10)

    msg_to_send = "N"

    server_shutdown_thread.join()
    # Run once shutdown key detected
    print("Ending program.")

    # Socket close handled by garbage collector


if __name__ == "__main__":
    main()
