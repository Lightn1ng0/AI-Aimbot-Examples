from ultralytics import YOLO
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

    if settings['yolo_version'] == "v8":
        model = YOLO(f"{models_path}/{settings['yolo_model']}", task="detect")
    elif settings['yolo_version'] == "v5":
        model = torch.hub.load('ultralytics/yolov5', 'custom', path=f"{models_path}/{settings['yolo_model']}", verbose=False, trust_repo=True, force_reload=True)

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
            if settings['yolo_version'] == "v5":
                model.conf = settings['confidence'] / 100
                model.iou = settings['confidence'] / 100
                results = model(frame, size=[settings['height'], settings['width']])
                if len(results.xyxy[0]) != 0:
                    for box in results.xyxy[0]:
                        box_result = calculate_targets(box[0], box[1], box[2], box[3])
                        coordinates.append((box[0], box[1], box[2], box[3]))
                        targets.append(box_result[0])
                        distances.append(box_result[1])

            elif settings['yolo_version'] == "v8":
                results = model.predict(frame, verbose=False, conf=settings['confidence'] / 100, iou=settings['confidence'] / 100, half=False, imgsz=[settings['height'], settings['width']])
                for result in results:
                    if len(result.boxes.xyxy) != 0:
                        for box in result.boxes.xyxy:
                            box_result = calculate_targets(box[0], box[1], box[2], box[3])
                            coordinates.append((box[0], box[1], box[2], box[3]))
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
