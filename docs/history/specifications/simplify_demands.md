### Simplify the current micro-service architecture

1. please restrospect on the design of micro service, by RESTful, it does not have to use sereialized objects for all cases, is there more efficient way to pass python object, or provide already existed use examples or codes fragments. 
2. please make sure the whole structure is easy for horizontal scaling. For example, when the user want to add a new attention processing for sparsification or when the user want to extend to a new parallelism, the system can be easily modified by RESTful apis. 