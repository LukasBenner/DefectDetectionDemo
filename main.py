#!/usr/bin/env python3
"""Simple camera viewer to test OpenCV CUDA GPU support."""

import time
import cv2
import numpy as np
import tensorrt as trt
import pycuda.driver as cuda
import pycuda.autoinit

# ----------------------------
# Config
# ----------------------------
DISPLAY_W, DISPLAY_H = 1280, 800  # your screen resolution
CAPTURE_W, CAPTURE_H = 1280, 800  # capture size (lower = less latency)
CAPTURE_FPS = 60
ENGINE_PATH = "mobilenetv3_480_fp16.engine"
MODEL_W, MODEL_H = 480, 480
PROB_THRESHOLD = 0.9
NUM_CLASSES = 9
CLASS_NAMES = [
    "black_stain",
    "corrosion",
    "crack",
    "deformation",
    "missing_part",
    "no_defects",
    "other",
    "silicate_stain",
    "water_stain",
]

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


class TensorRTClassifier:
    def __init__(self, engine_path: str, model_w: int, model_h: int, num_classes: int):
        self.model_w = model_w
        self.model_h = model_h
        self.num_classes = num_classes
        self.logger = trt.Logger(trt.Logger.WARNING)
        self.runtime = trt.Runtime(self.logger)
        with open(engine_path, "rb") as f:
            self.engine = self.runtime.deserialize_cuda_engine(f.read())
        if self.engine is None:
            raise RuntimeError(f"Failed to load TensorRT engine: {engine_path}")
        self.context = self.engine.create_execution_context()
        if self.context is None:
            raise RuntimeError("Failed to create TensorRT execution context")

        self.use_io_tensors = hasattr(self.engine, "num_io_tensors")
        self.input_index = None
        self.output_index = None
        self.input_name = None
        self.output_name = None

        if self.use_io_tensors:
            output_names = []
            for i in range(self.engine.num_io_tensors):
                name = self.engine.get_tensor_name(i)
                mode = self.engine.get_tensor_mode(name)
                if mode == trt.TensorIOMode.INPUT:
                    self.input_name = name
                else:
                    output_names.append(name)
            if self.input_name is None or not output_names:
                raise RuntimeError("Failed to find TensorRT input/output tensors")

            self.input_dtype = trt.nptype(self.engine.get_tensor_dtype(self.input_name))

            input_shape = self.engine.get_tensor_shape(self.input_name)
            if -1 in input_shape:
                self.context.set_input_shape(
                    self.input_name, (1, 3, self.model_h, self.model_w)
                )
            self.input_shape = tuple(self.context.get_tensor_shape(self.input_name))
            self.output_name = self._select_output_name(output_names)
            self.output_dtype = trt.nptype(self.engine.get_tensor_dtype(self.output_name))
            self.output_shape = tuple(self.context.get_tensor_shape(self.output_name))
        else:
            output_indices = []
            for i in range(self.engine.num_bindings):
                if self.engine.binding_is_input(i):
                    self.input_index = i
                else:
                    output_indices.append(i)
            if self.input_index is None or not output_indices:
                raise RuntimeError("Failed to find TensorRT input/output bindings")

            self.input_dtype = trt.nptype(self.engine.get_binding_dtype(self.input_index))
            self.output_dtype = trt.nptype(self.engine.get_binding_dtype(self.output_index))

            input_shape = self.engine.get_binding_shape(self.input_index)
            if -1 in input_shape:
                self.context.set_binding_shape(
                    self.input_index, (1, 3, self.model_h, self.model_w)
                )
            self.input_shape = tuple(self.context.get_binding_shape(self.input_index))
            self.output_index = self._select_output_index(output_indices)
            self.output_dtype = trt.nptype(self.engine.get_binding_dtype(self.output_index))
            self.output_shape = tuple(self.context.get_binding_shape(self.output_index))

        self.input_size = int(np.prod(self.input_shape))
        self.output_size = int(np.prod(self.output_shape))

        self.host_in = cuda.pagelocked_empty(self.input_size, dtype=self.input_dtype)
        self.host_out = cuda.pagelocked_empty(self.output_size, dtype=self.output_dtype)
        self.device_in = cuda.mem_alloc(self.host_in.nbytes)
        self.device_out = cuda.mem_alloc(self.host_out.nbytes)
        self.stream = cuda.Stream()

        if not self.use_io_tensors:
            self.bindings = [0] * self.engine.num_bindings
            self.bindings[self.input_index] = int(self.device_in)
            self.bindings[self.output_index] = int(self.device_out)

        self.mean = np.array([0.485, 0.456, 0.406], dtype=np.float32)
        self.std = np.array([0.229, 0.224, 0.225], dtype=np.float32)

    def preprocess(self, frame: np.ndarray) -> np.ndarray:
        resized = cv2.resize(frame, (self.model_w, self.model_h), interpolation=cv2.INTER_LINEAR)
        rgb = cv2.cvtColor(resized, cv2.COLOR_BGR2RGB)
        input_data = rgb.astype(np.float32) / 255.0
        input_data = (input_data - self.mean) / self.std
        chw = np.transpose(input_data, (2, 0, 1))
        return chw

    def _select_output_name(self, output_names: list[str]) -> str:
        for name in output_names:
            shape = tuple(self.context.get_tensor_shape(name))
            if shape and shape[-1] == self.num_classes:
                return name
            size = int(np.prod(shape)) if shape else 0
            if size == self.num_classes:
                return name
        return output_names[0]

    def _select_output_index(self, output_indices: list[int]) -> int:
        for idx in output_indices:
            shape = tuple(self.context.get_binding_shape(idx))
            if shape and shape[-1] == self.num_classes:
                return idx
            size = int(np.prod(shape)) if shape else 0
            if size == self.num_classes:
                return idx
        return output_indices[0]

    def _to_probs(self, output: np.ndarray) -> np.ndarray:
        logits = output.reshape(-1).astype(np.float32)
        if np.all(logits >= 0.0) and np.all(logits <= 1.0):
            s = float(np.sum(logits))
            if 0.99 <= s <= 1.01:
                return logits
        logits = logits - np.max(logits)
        exp = np.exp(logits)
        return exp / np.sum(exp)

    def infer(self, frame: np.ndarray) -> tuple[int, float]:
        chw = self.preprocess(frame)
        if self.input_dtype == np.float16:
            chw = chw.astype(np.float16)
        np.copyto(self.host_in, chw.ravel())

        cuda.memcpy_htod_async(self.device_in, self.host_in, self.stream)
        if self.use_io_tensors:
            self.context.set_tensor_address(self.input_name, int(self.device_in))
            self.context.set_tensor_address(self.output_name, int(self.device_out))
            self.context.execute_async_v3(stream_handle=self.stream.handle)
        else:
            self.context.execute_async_v2(bindings=self.bindings, stream_handle=self.stream.handle)
        cuda.memcpy_dtoh_async(self.host_out, self.device_out, self.stream)
        self.stream.synchronize()

        output = self.host_out.reshape(self.output_shape)
        probs = self._to_probs(output)
        top1 = int(np.argmax(probs))
        prob = float(probs[top1])
        return top1, prob

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

    classifier = TensorRTClassifier(ENGINE_PATH, MODEL_W, MODEL_H, NUM_CLASSES)

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

            class_id, prob = classifier.infer(frame)
            if prob >= PROB_THRESHOLD:
                label = CLASS_NAMES[class_id] if class_id < len(CLASS_NAMES) else str(class_id)
                cv2.putText(
                    frame,
                    f"{label}: {prob:.2f}",
                    (20, 100),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    1.2,
                    (0, 255, 255),
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