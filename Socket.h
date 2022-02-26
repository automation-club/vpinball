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
      if (socket.handle() != nullptr) // If we have a connection don't create a new one
      {
         try
         {
            std::cout << "Connecting to " << sockAddr << std::endl; // Make sure the 
            socket.connect(sockAddr);
            std::cout << "Error " << &zmq::error_t::what << std::endl;
         }
         catch (zmq::error_t &e)
         {
            std::cout << "Error " << e.what() << std::endl;
         }
      }
   }
     
   
   void send(std::string data)
   {
      try
      {
         socket.send(zmq::buffer(data), zmq::send_flags::none);
         zmq::message_t reply {};
         socket.recv(reply, zmq::recv_flags::none);
      }
      catch (zmq::error_t e)
      {
         std::cout << "Error Sending Data " << e.what() << std::endl;
      }
        
   };
   void cleanup()
   {
      std::cout << "Shutting down socket" << std::endl;
      try
      {
         socket.disconnect(sockAddr);

      }
      catch (zmq::error_t e)
      {

         std::cout << "Error with socket."
             << "Please check that Python socket server is running and restart the table." 
             << std::endl;
      }
   };
};
