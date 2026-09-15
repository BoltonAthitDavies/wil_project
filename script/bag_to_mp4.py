import cv2
import numpy as np
from pathlib import Path
from rosbags.highlevel import AnyReader
from rosbags.typesys import Stores, get_typestore

# --- Configuration ---
bagpath = Path('/home/ambushee/wil_project/dataset/yolo_dataset') # Point to the directory containing the .db3
topic_name = '/cam1/image_raw/compressed'      # Replace with your actual topic
output_video = 'output.mp4'
fps = 30                                         # Adjust to match your sensor's publish rate
# ---------------------

video_writer = None

# 1. Load the default message type definitions for ROS 2
typestore = get_typestore(Stores.ROS2_HUMBLE)

# 2. Open the bag and pass the typestore
with AnyReader([bagpath], default_typestore=typestore) as reader:
    
    # Filter for the specific topic
    connections = [x for x in reader.connections if x.topic == topic_name]
    
    if not connections:
        print(f"Topic '{topic_name}' not found in the bag.")
        print("Available topics are:")
        for conn in reader.connections:
            print(f" - {conn.topic}")
        exit()

    print(f"Extracting images from {topic_name}...")
    
    # Iterate through the messages
    for connection, timestamp, rawdata in reader.messages(connections=connections):
        
        # Deserialize using the typestore we loaded
        msg = typestore.deserialize_cdr(rawdata, connection.msgtype)
        
        # Decode the byte array into an OpenCV image
        np_arr = np.frombuffer(msg.data, np.uint8)
        cv_image = cv2.imdecode(np_arr, cv2.IMREAD_COLOR)
        
        if cv_image is None:
            continue
            
        # Initialize the VideoWriter on the first frame once dimensions are known
        if video_writer is None:
            height, width, _ = cv_image.shape
            fourcc = cv2.VideoWriter_fourcc(*'mp4v')
            video_writer = cv2.VideoWriter(output_video, fourcc, fps, (width, height))
            print(f"Video dimensions: {width}x{height} at {fps} FPS")
            
        video_writer.write(cv_image)

if video_writer is not None:
    video_writer.release()
    print(f"Success! Video saved to {output_video}")
else:
    print("No valid images were found to write.")