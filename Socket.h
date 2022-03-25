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
            // std::cout << "Error " << &zmq::error_t::what << std::endl;
         }
         catch (zmq::error_t &e)
         {
            std::cout << "Error " << e.what() << std::endl;
         }
      }
   }
     
   
   void send_request(std::string data)
   {
      try
      {
         // Send data to socket
         socket.send(zmq::buffer(data), zmq::send_flags::none);

         // Process response
         zmq::message_t reply {};
         socket.recv(reply, zmq::recv_flags::none);
         char replyCode = (char)reply.data();
         std::string strReply = std::string(static_cast<char *>(reply.data()), reply.size());
         std::cout << "[RECEIVED FROM SERVER]: " << strReply << std::endl;
         
         // Act on response
         Socket::process_agent_input(strReply);

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

private:
    
    void process_agent_input(std::string agent_request) 
    {
       switch (agent_request[0])
       {
            // Fire left flipper and turn off right flipper
            case 'L':
                g_pplayer->m_ptable->
                    FireKeyEvent(DISPID_GameEvents_KeyDown, g_pplayer->m_rgKeys[eLeftFlipperKey]);
                g_pplayer->m_ptable->
                    FireKeyEvent(DISPID_GameEvents_KeyUp, g_pplayer->m_rgKeys[eRightFlipperKey]);
               g_pplayer->m_ptable->
                   FireKeyEvent(DISPID_GameEvents_KeyUp, g_pplayer->m_rgKeys[ePlungerKey]);

                break;

            // Fire right flipper and turn off left flipper
            case 'R':
                g_pplayer-> m_ptable->
                    FireKeyEvent(DISPID_GameEvents_KeyDown, g_pplayer->m_rgKeys[eRightFlipperKey]);
                g_pplayer-> m_ptable->
                    FireKeyEvent(DISPID_GameEvents_KeyUp, g_pplayer->m_rgKeys[eLeftFlipperKey]);
                g_pplayer->m_ptable->
                   FireKeyEvent(DISPID_GameEvents_KeyUp, g_pplayer->m_rgKeys[ePlungerKey]);
                break;

            // Activate both flippers
            case 'B':
               g_pplayer->m_ptable->
                    FireKeyEvent(DISPID_GameEvents_KeyDown, g_pplayer->m_rgKeys[eLeftFlipperKey]);
               g_pplayer-> m_ptable->
                    FireKeyEvent(DISPID_GameEvents_KeyDown, g_pplayer->m_rgKeys[eRightFlipperKey]);
               g_pplayer->m_ptable->
                   FireKeyEvent(DISPID_GameEvents_KeyUp, g_pplayer->m_rgKeys[ePlungerKey]);
                break;

            // Both flippers off
            case 'N':
			   g_pplayer->m_ptable->
                   FireKeyEvent(DISPID_GameEvents_KeyUp, g_pplayer->m_rgKeys[eRightFlipperKey]);
               g_pplayer-> m_ptable->
                   FireKeyEvent(DISPID_GameEvents_KeyUp, g_pplayer->m_rgKeys[eLeftFlipperKey]);
               g_pplayer->m_ptable->
                   FireKeyEvent(DISPID_GameEvents_KeyUp, g_pplayer->m_rgKeys[ePlungerKey]);
               break;

            case 'P':
               g_pplayer->m_ptable->
                   FireKeyEvent(DISPID_GameEvents_KeyDown, g_pplayer->m_rgKeys[ePlungerKey]);
               g_pplayer->m_ptable->
                   FireKeyEvent(DISPID_GameEvents_KeyUp, g_pplayer->m_rgKeys[eRightFlipperKey]);
               g_pplayer-> m_ptable->
                   FireKeyEvent(DISPID_GameEvents_KeyUp, g_pplayer->m_rgKeys[eLeftFlipperKey]);
               break;

            case 'G':
               g_pplayer->m_ptable->
                   FireKeyEvent(DISPID_GameEvents_KeyDown, g_pplayer->m_rgKeys[eAddCreditKey]);
               g_pplayer->m_ptable->
                   FireKeyEvent(DISPID_GameEvents_KeyUp, g_pplayer->m_rgKeys[eAddCreditKey]);
               g_pplayer->m_ptable->
                   FireKeyEvent(DISPID_GameEvents_KeyDown, g_pplayer->m_rgKeys[eAddCreditKey]);
               g_pplayer->m_ptable->
                   FireKeyEvent(DISPID_GameEvents_KeyUp, g_pplayer->m_rgKeys[eAddCreditKey]);
               g_pplayer->m_ptable->
                   FireKeyEvent(DISPID_GameEvents_KeyDown, g_pplayer->m_rgKeys[eStartGameKey]);
               g_pplayer->m_ptable->
                   FireKeyEvent(DISPID_GameEvents_KeyUp, g_pplayer->m_rgKeys[eStartGameKey]);
               break;


       }     
    }
};
