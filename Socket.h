#pragma once
#include <string>
#include <iostream>

#include <zmq.hpp>
class Socket
{
#define sockAddr "tcp://localhost:5555"
   zmq::context_t context { 1 };
   zmq::socket_t socket { context, zmq::socket_type::req };

public:
   Socket()
   {
      if (socket.handle() != nullptr)
      {
         try
         {
            std::cout << "Connecting to " << sockAddr << std::endl;
            socket.connect(sockAddr);
         }
         catch (zmq::error_t)
         {
            std::cout << "Error" << &zmq::error_t::what << std::endl;
         }
      }
   }
     
   
   void send(std::string data)
   {
     
        // std::cout << "Sending Data "<< std::endl;
      if (&zmq::error_t::num)
      {
         socket.send(zmq::buffer(data), zmq::send_flags::none);
         zmq::message_t reply {};
         socket.recv(reply, zmq::recv_flags::none);
      }
   };
   void cleanup()
   {
      std::cout << "Shutting down socket" << std::endl;
      try
      {
         socket.disconnect(sockAddr);

      }
      catch (zmq::error_t)
      {
         std::cout << "Error disconnecting data" << &zmq::error_t::what << std::endl;
      }
   };
};
