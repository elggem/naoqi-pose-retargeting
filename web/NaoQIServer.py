from __future__ import print_function

# note this for Pythong 2.7 and 3.x compatibility
try:
  from SimpleHTTPServer import SimpleHTTPRequestHandler
  import SocketServer as socketserver
except ImportError:
  from http.server import SimpleHTTPRequestHandler
  import socketserver

# this makes it possible to run the script without naqi for testing purposes
try:
  import qi

  HAS_NAOQI = True
except:#ImportError:
  HAS_NAOQI = False


import threading
import traceback
import json

PORT = 8000

# debug stuff
def print_methods(o):
  object_methods = [method_name for method_name in dir(o) if callable(getattr(o, method_name))]
  print(object_methods)

module = None

class NaoQIManager:
    def __init__(self):
      self.services = {}

      # global
      if HAS_NAOQI:
        self.session = module.session()

      # create a connect a local session (in standalone case)
      #self.session = qi.Session()
      #self.session.connect("tcp://localhost:9559")

    # make a call to a service
    def call(self, service, call):
        if not HAS_NAOQI:
          print("WARNING: running without naoqi")

        # service is not connected yet
        if not service in self.services:
          print("Connect to Proxy: {}".format(service))

          print("INFO: Get service: {}".format(service))
          if HAS_NAOQI:
            self.services[service] = self.session.service(service)
          else:
            self.services[service] = None

        #print("call {}({})".format(service, call))
        if HAS_NAOQI:
          return self.services[service].call(*call)
          
        #print("DONE")

        return None


class NaoQiHandler(SimpleHTTPRequestHandler):
    def _set_headers(self):
        self.send_response(200)
        self.send_header("Content-type", "text/html")
        self.end_headers()
        
    '''
    def do_GET(self):
        #super().do_GET()
        #super(NaoQiHandler, self).do_GET()
        import os
        self._set_headers()
        self.wfile.write(str(path).encode("utf8"))
    '''

    def do_HEAD(self):
        self._set_headers()

    def run_naoqi(self, service, call):
        # create a naoqi manager if not avaliable yet
        if not hasattr(self, 'naoqi'):
            self.naoqi = NaoQIManager()

        # encode everything as ascii, this is crucial
        service = service.encode('ascii')
        call = [c.encode('ascii') for c in call if isinstance(c, str)]

        return self.naoqi.call(service, call)


    def do_POST(self):
        try:
          # get the length of the data to read
          # python 2.7
          if hasattr(self.headers, 'getheader'):
            length = int(self.headers.getheader('content-length'))
          else: # python 3.x
            length = int(self.headers.get('content-length'))

          data = self.rfile.read(length)

          # parse json data
          msg = json.loads(data)
          print(msg)

          # forward the call to naoqi
          result = self.run_naoqi(msg['proxy'], msg['call'])

          # sed a response
          self._set_headers()
          self.wfile.write(str(result).encode("utf8"))

        except Exception:
          self._set_headers()
          trace = traceback.format_exc()
          print(trace)
          self.wfile.write(str(trace).encode("utf8"))
          
    def log_message(self, format, *args):
        return


class ThreadedTCPServer(socketserver.ThreadingMixIn, socketserver.TCPServer):
    pass

          
if __name__ == "__main__":
  #socketserver.TCPServer.allow_reuse_address = True
  #server = socketserver.TCPServer(("", PORT), NaoQiHandler)
  
  ThreadedTCPServer.allow_reuse_address = True
  server = ThreadedTCPServer(("", PORT), NaoQiHandler)
  
  print("Serving at port: {}".format(PORT))
  server.serve_forever()
  print("Stopped serving at port: {}".format(PORT))
  