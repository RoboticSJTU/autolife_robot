import requests
import numpy as np
import matplotlib.pyplot as plt
import time
import cv2

SERVER_HOST = "localhost"
SERVER_PORT = 8080
COLOR_ENDPOINT = f"http://{SERVER_HOST}:{SERVER_PORT}/color"

def fetch_color_image():
    try:
        response = requests.get(COLOR_ENDPOINT, stream=True)
        if response.status_code == 200:
            img_bytes = np.frombuffer(response.content, dtype=np.uint8)
            img = cv2.imdecode(img_bytes, cv2.IMREAD_COLOR)
            return img
        else:
            print(f"Error: {response.status_code} - {response.reason}")
            return None
    except Exception as e:
        print(f"Failed to fetch image: {e}")
        return None

def main():
    plt.ion()  # Interactive mode
    fig, ax = plt.subplots()
    
    try:
        while True:
            img = fetch_color_image()
            if img is not None:
                ax.clear()
                ax.imshow(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))  # Convert BGR to RGB
                plt.pause(0.03)  # Small delay to update the plot
    except KeyboardInterrupt:
        plt.ioff()
        print("Stopping client...")

if __name__ == "__main__":
    main()