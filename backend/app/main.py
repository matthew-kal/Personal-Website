from fastapi import FastAPI, File, UploadFile
from fastapi.responses import JSONResponse
import cv2
import numpy as np
import mediapipe as mp
import pyautogui
import time

app = FastAPI()

mp_hands = mp.solutions.hands
hands = mp_hands.Hands()

@app.on_event("startup")
async def startup_event():
    pass

@app.post("/upload")
async def receive_frame(file: UploadFile = File(...)):
    contents = await file.read()
    nparr = np.frombuffer(contents, np.uint8)
    image = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    image = cv2.flip(image, 1)
    results = hands.process(image)
    commands_executed = []
    current_time = time.time()

    # Define last action times and delays
    last_click_time = 0
    click_delay = 1
    scroll_threshold = 20
    last_scroll_time = 0
    scroll_delay = 0.01


    if results.multi_hand_landmarks:
        for hand_landmarks in results.multi_hand_landmarks:
            landmark_dict = {id: (int(lm.x * image.shape[1]), int(lm.y * image.shape[0]))
                             for id, lm in enumerate(hand_landmarks.landmark)}

            # Mouse movement controlled by index finger
            if 20 in landmark_dict:
                screen_width, screen_height = pyautogui.size()
                mouse_x = int(screen_width / 640 * landmark_dict[20][0])
                mouse_y = int(screen_height / 480 * landmark_dict[20][1])
                pyautogui.moveTo(mouse_x, mouse_y)
                commands_executed.append(f"Moved mouse to ({mouse_x}, {mouse_y})")

            # Click action controlled by thumb and index finger proximity
            if 4 in landmark_dict and 8 in landmark_dict:
                if abs(landmark_dict[4][1] - landmark_dict[8][1]) < 20 and (current_time - last_click_time > click_delay):
                    pyautogui.click()
                    last_click_time = current_time
                    commands_executed.append("Clicked")

            # Scrolling controlled by middle and ring finger
            if 12 in landmark_dict and 16 in landmark_dict:
                if (current_time - last_scroll_time > scroll_delay):
                    if landmark_dict[16][1] - landmark_dict[12][1] > scroll_threshold:
                        pyautogui.scroll(10)  # Scroll up
                        last_scroll_time = current_time
                        commands_executed.append("Scrolled up")
                    elif landmark_dict[12][1] - landmark_dict[16][1] > scroll_threshold:
                        pyautogui.scroll(-10)  # Scroll down
                        last_scroll_time = current_time
                        commands_executed.append("Scrolled down")

    return JSONResponse(content={"message": "Frame processed", "commands": commands_executed})

@app.on_event("shutdown")
async def shutdown_event():
    hands.close()





