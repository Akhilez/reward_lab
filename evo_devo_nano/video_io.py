import cv2  # pip install opencv-python


class VideoWriter:
    def __init__(self, output_path: str, h: int, w: int, fps: int = 30):
        self.h = h
        self.w = w

        # Define the codec and create VideoWriter object
        fourcc = cv2.VideoWriter_fourcc(*"mp4v")  # Codec for MP4

        # Output file name, codec, frames per second, and frame size
        self.out = cv2.VideoWriter(output_path, fourcc, fps, (w, h))

    def write_frame(self, frame):
        assert frame.shape == (self.h, self.w, 3), f"Expected frame shape: {(self.h, self.w, 3)}, but got {frame.shape}"
        self.out.write(frame)

    def release(self):
        self.out.release()
