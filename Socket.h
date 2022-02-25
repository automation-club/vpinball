#pragma once
#include <string>
#include <iostream>

#include <zmq.hpp>
static class Socket
{
public:

 static void send(std::string data) 
   {
    zmq::context_t context { 1 };
    zmq::socket_t socket { context, zmq::socket_type::req };
    try
    {
       socket.connect("tcp://localhost:5555");

       std::cout << "Sending Data "
                 << "..." << std::endl;
       socket.send(zmq::buffer(data), zmq::send_flags::none);
       zmq::message_t reply {};
       socket.recv(reply, zmq::recv_flags::none);
       std::cout << "Received " << reply.to_string(); 
    }
    catch (zmq::error_t::exception)
    {
       std::cout << "Error Sending Data" << std::endl;
    }
   }
};
