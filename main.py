#!/usr/bin/env python3
"""Simple camera viewer to test OpenCV CUDA GPU support."""

import time
import cv2

# ----------------------------
# Config
# ----------------------------
DISPLAY_W, DISPLAY_H = 1280, 800  # your screen resolution
CAPTURE_W, CAPTURE_H = 1280, 800  # capture size (lower = less latency)
CAPTURE_FPS = 60

from smbus2 import SMBus, i2c_msg
import time
import subprocess

class Lightning():
    def __init__(self):
        subprocess.Popen(["gpioset", "gpiochip2", "19=1"])
        subprocess.Popen(["gpioset", "gpiochip2", "16=1"])
        self.bus_number = 7
        self.device_address = 0x12
        self.LED_array = [
            [11, 70, 0, 0,0,0,0,0,0,0,255,88],
            [11, 70, 7, 0,7,0,7,0,7, 0,1,88],
            [11, 70, 7, 0,7,0,7,0,7, 0,2,88],
            [11, 70, 7, 0,7,0,7,0,7, 0,4,88],
            [11, 70, 7, 0,7,0,7,0,7, 0,8,88],
            [11, 70, 0, 255,0,255,1,255, 1,255,64,88],
            [11, 70, 0, 255,0,255,1,255, 1,255,16,88],
            [11, 70, 7, 255,7,255,10,255, 15,255,128,88],
            [11, 70, 7, 255,7,255,10,255, 15,255,32,88]
        ]
        self.sortBite= [0x03,0x01,0x01,0x02]
    def start(self):
        with SMBus(self.bus_number) as bus:
            for data in self.LED_array:
                write = i2c_msg.write(self.device_address, data)
                time.sleep(0.1)
                bus.i2c_rdwr(write)

    def stop(self):
        subprocess.Popen(["gpioset", "gpiochip2", "19=0"])
        subprocess.Popen(["gpioset", "gpiochip2", "16=0"])

    def update(self):
        pass

# ----------------------------
# Camera pipeline
# ----------------------------
def gstreamer_pipeline() -> str:
    return (
        "nvarguscamerasrc sensor-id=0 ! "
        f"video/x-raw(memory:NVMM), width={CAPTURE_W}, height={CAPTURE_H}, "
        f"framerate={CAPTURE_FPS}/1, format=NV12 ! "
        "nvvidconv flip-method=2 ! "
        "video/x-raw, format=BGRx ! "
        "queue max-size-buffers=1 leaky=downstream ! "
        "appsink drop=1 max-buffers=1 sync=false"
    )

# ----------------------------
# Main
# ----------------------------
def main():
    # Check OpenCV CUDA support
    print(f"OpenCV version: {cv2.__version__}")
    print(f"CUDA enabled: {cv2.cuda.getCudaEnabledDeviceCount() > 0}")
    if cv2.cuda.getCudaEnabledDeviceCount() > 0:
        print(f"CUDA devices: {cv2.cuda.getCudaEnabledDeviceCount()}")
        cv2.cuda.printShortCudaDeviceInfo(0)
    print()

    l = Lightning()
    l.start()

    delay = 2
    print(f"Waiting {delay} seconds for camera to initialize...")
    time.sleep(delay)
    
    cap = None
    pipeline = gstreamer_pipeline()
    try:
        candidate = cv2.VideoCapture(pipeline, cv2.CAP_GSTREAMER)
        if candidate.isOpened():
            cap = candidate
            cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
    except Exception as e:
        print(f"Error opening pipeline: {e}")

    if cap is None or not cap.isOpened():
        raise RuntimeError("Failed to open camera with CSI GStreamer pipelines")

    # Setup window
    window_name = "Camera Stream"
    cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)
    cv2.resizeWindow(window_name, DISPLAY_W, DISPLAY_H)

    # FPS tracking
    fps = 0.0
    t_prev = time.time()

    print("Press ESC to exit\n")

    try:
        while True:
            ok, frame = cap.read()
            if not ok:
                print("Failed to read frame")
                break

            # Calculate FPS
            now = time.time()
            dt = now - t_prev
            if dt > 0:
                fps = 0.9 * fps + 0.1 * (1.0 / dt)
            t_prev = now

            # Draw FPS counter
            cv2.putText(
                frame,
                f"FPS: {fps:.1f}",
                (20, 50),
                cv2.FONT_HERSHEY_SIMPLEX,
                1.5,
                (0, 255, 0),
                3,
                cv2.LINE_AA,
            )

            # Display
            cv2.imshow(window_name, frame)

            # Exit on ESC
            if (cv2.waitKey(1) & 0xFF) == 27:
                break

    finally:
        cap.release()
        cv2.destroyAllWindows()
        print(f"\nAverage FPS: {fps:.1f}")


if __name__ == "__main__":
    main()