import socket, errno

def is_port_in_use(port_num):
    s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)

    try:
        s.bind(("127.0.0.1", port_num))
        s.close()
        return False
    except socket.error as e:
        if e.errno == errno.EADDRINUSE:
            print(f"Port{port_num} is already in use")
        else:
            # something else raised the socket.error exception
            print(e)
        s.close()
        return True

    
