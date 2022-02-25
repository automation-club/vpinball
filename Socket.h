#pragma once
#include <string>
#include <iostream>

#include <zmq.hpp>
static class Socket
{

         
    public:

    static  void send(std::string data)
    {
        // Specify socket type
        zmq::context_t context { 1 };
        zmq::socket_t socket { context, zmq::socket_type::req };

        socket.connect("tcp://localhost:5555");
        std::cout << "Sending Data "
                << "..." << std::endl;
        socket.send(zmq::buffer(data), zmq::send_flags::none);
        zmq::message_t reply {};
        socket.recv(reply, zmq::recv_flags::none);
        std::cout << "Received " << reply.to_string(); 

   }
};
