"""Start the localhost research workspace: python -m src.dashboard."""
import argparse
from pathlib import Path
import socket
import threading
import time
import webbrowser
import uvicorn
from .app import create_app


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--source',type=Path)
    parser.add_argument('--port',type=int,default=8000)
    parser.add_argument('--open',action='store_true',help='Open the browser after startup')
    args = parser.parse_args()
    if not 1<=args.port<=65535:
        parser.error('port must be between 1 and 65535')
    try:
        with socket.socket() as probe:
            probe.bind(('127.0.0.1',args.port))
    except OSError:
        parser.error(f'Port {args.port} is occupied; choose another --port')
    if args.open:
        def open_when_ready():
            for _ in range(100):
                try:
                    with socket.create_connection(('127.0.0.1',args.port),timeout=.2):
                        webbrowser.open(f'http://127.0.0.1:{args.port}')
                        return
                except OSError:
                    time.sleep(.1)
        threading.Thread(target=open_when_ready,daemon=True).start()
    uvicorn.run(create_app(args.source),host='127.0.0.1',port=args.port)


if __name__=='__main__':
    main()
