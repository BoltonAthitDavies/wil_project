import cv2
import numpy as np
import av
from pathlib import Path
from rosbags.highlevel import AnyReader
from rosbags.typesys import Stores, get_typestore

# --- Configuration ---
# bagpath = Path('/home/ambushee/wil_project/dataset/yolo_dataset') # Point to the directory containing the .db3
# topic_name = '/cam1/image_raw/compressed'      # Replace with your actual topic
bagpath = Path('/home/ambushee/wil_project/dataset/dataset_map_realistic') # Point to the directory containing the .db3
topic_name = '/cam1/image_raw/compressed'      # Replace with your actual topic
output_video = '/home/ambushee/wil_project/dataset/dataset_map_realistic/output.mp4'
fps = 30                                         # Adjust to match your sensor's publish rate
crf = 18                                         # H.264 quality: lower = better/larger (18 ~ visually lossless, 23 = x264 default)
preset = 'medium'                                # x264 speed/size trade-off: ultrafast ... veryslow
# ---------------------

# NOTE: OpenCV's bundled FFmpeg has no software H.264 encoder (cv2.VideoWriter
# with 'avc1'/'H264' fails), so the MP4 is written with PyAV + libx264 instead.

container = None
stream = None
n_frames = 0

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
            
        # Initialize the H.264 encoder on the first frame once dimensions are known
        if container is None:
            height, width, _ = cv_image.shape
            # yuv420p (the universally playable H.264 pixel format) needs even dimensions
            width -= width % 2
            height -= height % 2
            container = av.open(output_video, mode='w')
            stream = container.add_stream('libx264', rate=fps)
            stream.width = width
            stream.height = height
            stream.pix_fmt = 'yuv420p'
            stream.options = {'crf': str(crf), 'preset': preset}
            print(f"Video dimensions: {width}x{height} at {fps} FPS (H.264 / libx264, crf={crf}, preset={preset})")
        
        cv_image = cv_image[:stream.height, :stream.width]
        frame = av.VideoFrame.from_ndarray(cv_image, format='bgr24')
        for packet in stream.encode(frame):
            container.mux(packet)
        n_frames += 1

if container is not None:
    # Flush any frames still buffered inside the encoder
    for packet in stream.encode():
        container.mux(packet)
    container.close()
    print(f"Success! {n_frames} frames saved to {output_video}")
else:
    print("No valid images were found to write.")
