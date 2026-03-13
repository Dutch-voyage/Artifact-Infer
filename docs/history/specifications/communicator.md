### Communicator for differener services

There are different services in the current inference engine framework. For different services, they are pipelined together to satisfy different workflows. 

Some of the service callings are I/O heavy and some of are compute heavy. I want to create a communicator that provide interface to call cross-service functions, for different occasions. 

For computing-heavy, such cross-service are inter-process communication. For IO heavy, such corss-service are simple implented by asyncio. For simpler cases, there might be cases of in-process function calling(probably use partial function as returns). 

Please design interface for these occasions and implement them to suit the current service interface. You implement multi-processes in the service_manager.py, and offer necesary resources needed for corss-service communications.

### New demands

1. For in-process calls, do not use aysnc calling
2. For IO bound, use asyncio.Event, to handle repsonse send and receive
3. For compute-bound, use zmq, to create socket and pipeline context

4. For communicator, please view it as request dispatcher, for different service calling, handle the service call dependently. 