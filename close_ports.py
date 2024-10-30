import os
import signal
import subprocess

def close_ports_in_range(start_port, end_port):
    for port in range(start_port, end_port + 1):
        try:
            # Use lsof to find the process using the port
            result = subprocess.run(['lsof', '-ti', f':{port}'], capture_output=True, text=True)
            if result.stdout:
                pids = result.stdout.splitlines()
                for pid in pids:
                    # Terminate the process
                    print(f'Closing port {port}, terminating PID {pid}')
                    os.kill(int(pid), signal.SIGTERM)  # Graceful termination
                    # Optionally, use SIGKILL if it doesn't close
                    # os.kill(int(pid), signal.SIGKILL)
            else:
                print(f'No process is using port {port}')
        except Exception as e:
            print(f'Error while processing port {port}: {e}')

if __name__ == '__main__':
    start = 7000
    end = int(input("Enter the end port: "))
    close_ports_in_range(start, end)
