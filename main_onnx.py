from ultralytics.utils import ops
import onnxruntime as ort
import numpy as np
import bettercam
import win32api
import win32con
import json
import math
import time
import torch
import sys
import os

script_directory = os.path.dirname(os.path.abspath(__file__ if "__file__" in locals() else __file__))
sys.path.append(f"{script_directory}/yolov5")

from yolov5.utils.general import non_max_suppression

if torch.cuda.is_available():
    import cupy as cp

models_path = os.path.join(os.getenv("APPDATA"), "ai-aimbot-launcher", "models")

with open(f"{script_directory}/configuration/key_mapping.json", 'r') as json_file:
    key_mapping = json.load(json_file)

with open(f"{script_directory}/configuration/config.json", 'r') as json_file:
    settings = json.load(json_file)

targets = []
distances = []
coordinates = []


def calculate_targets(x1, y1, x2, y2):
    width_half = settings['width'] / 2
    height_half = settings['height'] / 2
    headshot_percent = settings['headshot'] / 100
    x = int(((x1 + x2) / 2) - width_half)
    y = int(((y1 + y2) / 2) + headshot_percent * (y1 - ((y1 + y2) / 2)) - height_half)
    distance = math.sqrt(x ** 2 + y ** 2)
    return (x, y), distance


def main(**argv):
    global settings
    paid_tier = argv['paidTier']

    with open(os.path.join(os.getenv("APPDATA"), "ai-aimbot-launcher", "aimbotSettings", f"{argv['settingsProfile'].lower()}.json"), "r") as f:
        launcher_settings = json.load(f)

    activation_key = key_mapping.get(launcher_settings['activationKey'])
    quit_key = key_mapping.get(launcher_settings['quitKey'])

    if activation_key is None:
        activation_key = win32api.VkKeyScan(launcher_settings['activationKey'])
    if quit_key is None:
        quit_key = win32api.VkKeyScan(launcher_settings['quitKey'])

    settings['sensitivity'] = launcher_settings['movementAmp'] * 100
    settings['headshot'] = launcher_settings['headshotDistanceModifier'] * 100 if launcher_settings['headshotMode'] else 20
    settings['confidence'] = launcher_settings['confidence'] * 100
    settings['height'] = launcher_settings['screenShotHeight']
    settings['width'] = launcher_settings['screenShotHeight']
    settings['yolo_version'] = f"v{argv['yoloVersion']}"
    settings['yolo_model'] = argv['modelFileName']
    settings['yolo_device'] = {1: "cpu", 2: "amd", 3: "nvidia"}.get(launcher_settings['onnxChoice'])
    settings['activation_key'] = activation_key
    settings['quit_key'] = quit_key

    onnx_provider = ""
    if settings['yolo_device'] == "cpu":
        onnx_provider = "CPUExecutionProvider"
    elif settings['yolo_device'] == "amd":
        onnx_provider = "DmlExecutionProvider"
    elif settings['yolo_device'] == "nvidia":
        onnx_provider = "CUDAExecutionProvider"

    so = ort.SessionOptions()
    so.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
    model = ort.InferenceSession(f"{models_path}/{settings['yolo_model']}", sess_options=so, providers=[onnx_provider])

    left = int(win32api.GetSystemMetrics(0) / 2 - settings['width'] / 2)
    top = int(win32api.GetSystemMetrics(1) / 2 - settings['height'] / 2)
    right = int(left + settings['width'])
    bottom = int(top + settings['height'])
    screen = bettercam.create(output_color="BGRA", max_buffer_len=512)
    screen.start(region=(left, top, right, bottom), target_fps=120, video_mode=True)

    start_time = time.time()
    frame_count = 0

    while True:
        frame_count += 1

        frame = np.array(screen.get_latest_frame())

        if frame.shape[2] == 4:
            frame = frame[:, :, :3]

        with torch.no_grad():
            if settings['yolo_device'] == "nvidia":
                frame = torch.from_numpy(frame).to('cuda')
                frame = torch.movedim(frame, 2, 0)
                frame = frame.half()
                frame /= 255
                if len(frame.shape) == 3:
                    frame = frame[None]
            else:
                frame = np.array([frame])
                frame = frame / 255
                frame = frame.astype(np.half)
                frame = np.moveaxis(frame, 3, 1)

            if settings['yolo_device'] == "nvidia":
                outputs = model.run(None, {'images': cp.asnumpy(frame)})
            else:
                outputs = model.run(None, {'images': np.array(frame)})

            frame = torch.from_numpy(outputs[0])

            if settings['yolo_version'] == "v5":
                predictions = non_max_suppression(frame, settings['confidence'] / 100, settings['confidence'] / 100, 0, False, max_det=4)

            elif settings['yolo_version'] == "v8":
                predictions = ops.non_max_suppression(frame, settings['confidence'] / 100, settings['confidence'] / 100, 0, False, max_det=4)

            for i, det in enumerate(predictions):
                if len(det):
                    for *xyxy, conf, cls in reversed(det):
                        if int(cls) == 0:
                            box_result = calculate_targets(xyxy[0].item(), xyxy[1].item(), xyxy[2].item(), xyxy[3].item())
                            coordinates.append((xyxy[0].item(), xyxy[1].item(), xyxy[2].item(), xyxy[3].item()))
                            targets.append(box_result[0])
                            distances.append(box_result[1])

            if distances:
                sensitivity_factor = settings['sensitivity'] / 20

                target_x, target_y = targets[distances.index(min(distances))]

                if win32api.GetKeyState(settings['activation_key']) in (-127, -128):
                    mouse_move_x = int(target_x * sensitivity_factor)
                    mouse_move_y = int(target_y * sensitivity_factor)

                    win32api.mouse_event(win32con.MOUSEEVENTF_MOVE, mouse_move_x, mouse_move_y, 0, 0)

        elapsed_time = time.time() - start_time
        if elapsed_time >= 1:
            print(f"Fps: {round(frame_count / elapsed_time)}")
            frame_count = 0
            start_time = time.time()

        if win32api.GetKeyState(settings['quit_key']) in (-127, -128):
            screen.stop()
            screen.release()
            quit()

        targets.clear()
        distances.clear()
        coordinates.clear()
