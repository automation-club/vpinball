import zmq
import threading
import keyboard
import os

def handle_socket_server(socket):
    while True:
        #  Wait for next request from qclient
        message = socket.recv()
        print(f"[RECEIVED FROM CLIENT]: {message}")

        #  Send reply back to client
        socket.send("".encode())


def detect_quit_keypress():
    while True:
        if keyboard.is_pressed('q'):
            print("Shutdown key detected.")
            break


def launch_visual_pinball():
    VISUAL_PINBALL_PATH = r".\x64\Debug\VPinballX.exe"

    os.system(VISUAL_PINBALL_PATH)


def main():
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

    server_shutdown_thread.join()
    # Run once shutdown key detected
    print("Ending program.")

    # Socket close handled by garbage collector


if __name__ == "__main__":
    main()
