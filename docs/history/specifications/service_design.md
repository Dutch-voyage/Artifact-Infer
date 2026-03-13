### Further Service API Designs

1. Some service run separate processes. The relevant management is provided by service_manager.

2. The communicator should provide a unified interface for a service to call the function of another service, but it seems that the for sync and async call the abstraction is difficult to unify both approaches. 

3. 