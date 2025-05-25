import pyrealsense2 as rs
from http.server import BaseHTTPRequestHandler, HTTPServer
from socketserver import ThreadingMixIn
import numpy as np
import cv2
import threading
import time
import yaml

# Load configuration from YAML file
with open('./src/grasp/config/camera_server.yaml', 'r') as file:
    config = yaml.safe_load(file)

# Threaded HTTP server to handle multiple requests
class ThreadedHTTPServer(ThreadingMixIn, HTTPServer):
    pass

# HTTP request handler
class CameraHandler(BaseHTTPRequestHandler):
    last_depth_image = None
    last_color_image = None
    lock = threading.Lock()

    def do_GET(self):
        if self.path == '/depth':
            with self.lock:
                if self.last_depth_image is None:
                    self.send_error(404, "Depth image not available")
                    return
                self.send_response(200)
                self.send_header('Content-type', 'image/jpeg')
                self.end_headers()
                self.wfile.write(self.last_depth_image)
                
        elif self.path == '/color':
            with self.lock:
                if self.last_color_image is None:
                    self.send_error(404, "Color image not available")
                    return
                self.send_response(200)
                self.send_header('Content-type', 'image/jpeg')
                self.end_headers()
                self.wfile.write(self.last_color_image)
        else:
            self.send_error(404, "Not Found")

def process_frames(server):
    # Configure depth and color streams using YAML config
    pipeline = rs.pipeline()
    rs_config = rs.config()
    
    # Get depth settings from config
    depth_cfg = config['camera']['depth']
    rs_config.enable_stream(
        rs.stream.depth, 
        depth_cfg['width'], 
        depth_cfg['height'], 
        getattr(rs.format, depth_cfg['format']), 
        depth_cfg['fps']
    )
    
    # Get color settings from config
    color_cfg = config['camera']['color']
    rs_config.enable_stream(
        rs.stream.color, 
        color_cfg['width'], 
        color_cfg['height'], 
        getattr(rs.format, color_cfg['format']), 
        color_cfg['fps']
    )

    # Start streaming
    pipeline.start(rs_config)

    try:
        while True:
            # Wait for a coherent pair of frames: depth and color
            frames = pipeline.wait_for_frames()
            depth_frame = frames.get_depth_frame()
            color_frame = frames.get_color_frame()
            
            if not depth_frame or not color_frame:
                continue

            # Convert images to numpy arrays
            depth_image = np.asanyarray(depth_frame.get_data())
            color_image = np.asanyarray(color_frame.get_data())

            # Apply colormap to depth image
            depth_colormap = cv2.applyColorMap(
                cv2.convertScaleAbs(depth_image, alpha=0.03), 
                cv2.COLORMAP_JET
            )

            # Encode images to JPEG
            _, depth_jpeg = cv2.imencode('.jpg', depth_colormap)
            _, color_jpeg = cv2.imencode('.jpg', color_image)

            # Update the server with the latest images
            with server.RequestHandlerClass.lock:
                server.RequestHandlerClass.last_depth_image = depth_jpeg.tobytes()
                server.RequestHandlerClass.last_color_image = color_jpeg.tobytes()

    finally:
        pipeline.stop()

def main():
    server_config = config['server']
    host = server_config['host']
    port = server_config['port']
    server = ThreadedHTTPServer((host, port), CameraHandler)
    print(f"Server started on http://{host}:{port}")
    
    # Start frame processing thread
    frame_thread = threading.Thread(
        target=process_frames, 
        args=(server,),
        daemon=True
    )
    frame_thread.start()
    
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        pass
    
    server.server_close()
    print("Server stopped")

if __name__ == "__main__":
    main()