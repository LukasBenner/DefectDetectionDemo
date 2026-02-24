#!/usr/bin/env python3
"""Simple camera viewer to test OpenCV CUDA GPU support."""

import time
import cv2
import numpy as np
import tensorrt as trt
import pycuda.driver as cuda
import pycuda.autoinit
import os
import argparse
from collections import deque

# ----------------------------
# Config
# ----------------------------
DISPLAY_W, DISPLAY_H = 1280, 800  # your screen resolution
CAPTURE_W, CAPTURE_H = 1280, 800  # capture size (lower = less latency)
CAPTURE_FPS = 60
ENGINE_PATH = "mobilenet_l_480_fp16.engine"
MODEL_W, MODEL_H = 480, 480
CROP_SIZE = 960
PROB_THRESHOLD = 0.85
NUM_CLASSES = 9
DUMP_INTERVAL_SEC = 5.0
# CLASS_NAMES = [
#     "background",
#     "ok",
#     "defect"
# ]

CLASS_NAMES = [
    "black_stain",
    "corrosion",
    "crack",
    "deformation",
    "missing_part",
    "ok",
    "other",
    "silicate_stain",
    "water_stain"
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
        self.last_crop_bgr: np.ndarray | None = None
        self.last_logits: np.ndarray | None = None
        self.last_probs: np.ndarray | None = None

    def preprocess(self, frame: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        h, w = frame.shape[:2]
        is_bgra = frame.ndim == 3 and frame.shape[2] == 4
        scale = float(CROP_SIZE) / float(min(h, w))
        new_w = int(round(w * scale))
        new_h = int(round(h * scale))
        y0 = (new_h - CROP_SIZE) // 2
        x0 = (new_w - CROP_SIZE) // 2

        gpu = cv2.cuda_GpuMat()
        gpu.upload(frame)
        if is_bgra:
            gpu = cv2.cuda.cvtColor(gpu, cv2.COLOR_BGRA2BGR)
        gpu_resized = cv2.cuda.resize(gpu, (new_w, new_h), interpolation=cv2.INTER_LINEAR)
        gpu_crop = gpu_resized.rowRange(y0, y0 + CROP_SIZE).colRange(x0, x0 + CROP_SIZE)
        gpu_model = cv2.cuda.resize(
            gpu_crop, (self.model_w, self.model_h), interpolation=cv2.INTER_LINEAR
        )
        gpu_rgb = cv2.cuda.cvtColor(gpu_model, cv2.COLOR_BGR2RGB)
        cropped = gpu_model.download()
        rgb = gpu_rgb.download()
       
        input_data = rgb.astype(np.float32) / 255.0
        input_data = (input_data - self.mean) / self.std
        chw = np.transpose(input_data, (2, 0, 1))
        return chw, cropped

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

    def infer(self, frame: np.ndarray) -> tuple[int, float, float]:
        infer_start = time.perf_counter()

        chw, crop_bgr = self.preprocess(frame)
        self.last_crop_bgr = crop_bgr
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
        infer_ms = (time.perf_counter() - infer_start) * 1000.0

        output = self.host_out.reshape(self.output_shape)
        logits = output.reshape(-1).astype(np.float32)
        probs = self._to_probs(output)
        self.last_logits = logits
        self.last_probs = probs
        top1 = int(np.argmax(probs))
        prob = float(probs[top1])
        return top1, prob, infer_ms



# ----------------------------
# Camera pipeline
# ----------------------------
def gstreamer_pipeline(
        sensor_id: int = 0,
        sensor_mode: int = 1,
        capture_width: int = 2464,
        capture_height: int = 2064,
        flip_method: int = 2,
        framerate: int = 60
) -> str:
    return (
        f"nvarguscamerasrc sensor-id={sensor_id} sensor-mode={sensor_mode} aelock=true awblock=false wbmode=2 tnr-mode=0 tnr-strength=-1 ee-mode=2 ee-strength=0 saturation=0.75 gainrange=\"1 1\" ispdigitalgainrange=\"1 1\" "
        f"! video/x-raw(memory:NVMM), width={capture_width}, height={capture_height}, framerate={framerate}/1, format=NV12 "
        f"! nvvidconv flip-method={flip_method} "
        "! video/x-raw, format=BGRx "
        "! queue max-size-buffers=1 leaky=downstream "
        "! appsink drop=1 max-buffers=1 sync=false"
    )

# ----------------------------
# Main
# ----------------------------
def main():
    parser = argparse.ArgumentParser(description="Camera stream with TensorRT inference")
    parser.add_argument(
        "--debug-crops",
        action="store_true",
        help="Enable periodic debug crop dumps and output logging",
    )
    args = parser.parse_args()

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
    pipeline = gstreamer_pipeline(capture_width=CAPTURE_W, capture_height=CAPTURE_H, framerate=CAPTURE_FPS)
    try:
        candidate = cv2.VideoCapture(pipeline, cv2.CAP_GSTREAMER)
        if candidate.isOpened():
            cap = candidate
            cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
    except Exception as e:
        print(f"Error opening pipeline: {e}")

    if cap is None or not cap.isOpened():
        raise RuntimeError("Failed to open camera with CSI GStreamer pipelines")

    window_name = "Camera Stream"
    cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)
    cv2.resizeWindow(window_name, DISPLAY_W, DISPLAY_H)

    # FPS tracking
    fps = 0.0
    t_prev = time.time()
    frame_idx = 0
    lighting_idx = 0
    last_dump_time = 0.0
    infer_times = deque(maxlen=30)
    last_class_id: int | None = None
    last_prob: float | None = None

    classifier = TensorRTClassifier(ENGINE_PATH, MODEL_W, MODEL_H, NUM_CLASSES)

    if args.debug_crops and DUMP_INTERVAL_SEC > 0:
        print(f"Press ESC to exit. Auto dump every {DUMP_INTERVAL_SEC:.0f} seconds.\n")
    else:
        print("Press ESC to exit.\n")

    try:
        while True:
            ok, frame = cap.read()
            if not ok:
                print("Failed to read frame")
                break
            frame_idx += 1
            if frame_idx % 2 == 0:
                continue
            lighting_idx += 1

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


            if lighting_idx % 2 == 1:
                class_id, prob, infer_ms = classifier.infer(frame)
                infer_times.append(infer_ms)
                last_class_id = class_id
                last_prob = prob

            if infer_times:
                mean_infer_ms = sum(infer_times) / len(infer_times)
                cv2.putText(
                    frame,
                    f"Infer: {mean_infer_ms:.1f} ms (avg 30)",
                    (20, 140),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    1.0,
                    (0, 255, 255),
                    2,
                    cv2.LINE_AA,
                )

            if last_class_id is not None and last_prob is not None:
                if last_prob >= PROB_THRESHOLD:
                    label = (
                        CLASS_NAMES[last_class_id]
                        if last_class_id < len(CLASS_NAMES)
                        else str(last_class_id)
                    )
                    cv2.putText(
                        frame,
                        f"{label}: {last_prob:.2f}",
                        (20, 100),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        1.2,
                        (0, 255, 255),
                        3,
                        cv2.LINE_AA,
                    )

            cv2.imshow(window_name, frame)

            if args.debug_crops and DUMP_INTERVAL_SEC > 0 and (now - last_dump_time) >= DUMP_INTERVAL_SEC:
                if classifier.last_crop_bgr is not None:
                    os.makedirs("debug_crops", exist_ok=True)
                    stamp = time.strftime("%Y%m%d_%H%M%S")
                    out_path = os.path.join("debug_crops", f"crop_{stamp}.jpg")
                    cv2.imwrite(out_path, classifier.last_crop_bgr)
                    print(f"Saved crop: {out_path}")
                if classifier.last_logits is not None and classifier.last_probs is not None:
                    print("Logits:", np.round(classifier.last_logits, 4))
                    print("Probs:", np.round(classifier.last_probs, 4))
                last_dump_time = now

            key = cv2.waitKey(1) & 0xFF
            # Exit on ESC
            if key == 27:
                break

    finally:
        cap.release()
        cv2.destroyAllWindows()
        print(f"\nAverage FPS: {fps:.1f}")


if __name__ == "__main__":
    main()