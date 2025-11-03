import cv2
import numpy as np
import mediapipe as mp
import time
import os 

# ----------------------------
# Config / Layout Definitions
# ----------------------------
WIN_W, WIN_H = 1920, 1080  # Fullscreen HD
TOOL_W = 280
CANVAS_W, CANVAS_H = WIN_W - TOOL_W, WIN_H
VIDEO_SMALL_W, VIDEO_SMALL_H = 320, 200
FPS = 30
PAD = 12

COLOR_PALETTE = [
    (0, 0, 255), (0, 255, 0), (255, 0, 0),
    (0, 255, 255), (0, 165, 255),
    (255, 0, 255), (255, 255, 255), (0, 0, 0)
]
BG_PALETTE = [
    (40, 40, 40), (255, 255, 255),
    (220, 220, 255), (200, 255, 200),
    (255, 230, 200)
]

# ----------------------------
# Globals & State
# ----------------------------
mp_hands = mp.solutions.hands
mp_draw = mp.solutions.drawing_utils
cap = cv2.VideoCapture(0)
cap.set(cv2.CAP_PROP_FRAME_WIDTH, WIN_W)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, WIN_H)

canvas = np.zeros((CANVAS_H, CANVAS_W, 3), dtype=np.uint8)
background_color = BG_PALETTE[0]
canvas[:] = background_color

current_color = COLOR_PALETTE[0]
brush_size = 12
eraser_size = 80
prev_point = None
help_visible = False
paused = False

buttons = {}

# Notification
notification_text = ""
notification_time = 0

# ----------------------------
# Toolbar Drawing
# ----------------------------
def draw_toolbar(img):
    global buttons
    buttons.clear()

    cv2.rectangle(img, (0, 0), (TOOL_W, WIN_H), (245, 245, 245), -1)
    x0 = 15
    y0 = PAD + 20
    cv2.putText(img, " AI Virtual Painter", (x0, y0 + 10),
                cv2.FONT_HERSHEY_COMPLEX, 0.8, (0, 102, 204), 2)
    y0 += 60

    # Brush Size
    cv2.putText(img, " Brush Size", (x0, y0 + 20),
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (50, 50, 50), 2)
    minus_rect = (x0, y0 + 45, 36, 36)
    plus_rect = (x0 + 150, y0 + 45, 36, 36)
    size_box = (x0 + 50, y0 + 45, 90, 36)
    cv2.rectangle(img, (minus_rect[0], minus_rect[1]), (minus_rect[0]+36, minus_rect[1]+36), (200,200,200), -1)
    cv2.putText(img, "-", (minus_rect[0]+10, minus_rect[1]+27), cv2.FONT_HERSHEY_SIMPLEX, 1, (30,30,30), 2)
    cv2.rectangle(img, (size_box[0], size_box[1]), (size_box[0]+size_box[2], size_box[1]+size_box[3]), (255,255,255), -1)
    cv2.putText(img, str(brush_size), (size_box[0]+20, size_box[1]+25), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0,0,0), 2)
    cv2.rectangle(img, (plus_rect[0], plus_rect[1]), (plus_rect[0]+36, plus_rect[1]+36), (200,200,200), -1)
    cv2.putText(img, "+", (plus_rect[0]+8, plus_rect[1]+27), cv2.FONT_HERSHEY_SIMPLEX, 1, (30,30,30), 2)
    buttons['size_minus'] = minus_rect
    buttons['size_plus'] = plus_rect
    y0 += 100

    # Color Palette
    cv2.putText(img, "Colors", (x0, y0 + 20),
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (50, 50, 50), 2)
    y0 += 40
    sw, gap = 40, 10
    for i, col in enumerate(COLOR_PALETTE):
        cx = x0 + (i % 2) * (sw + gap)
        cy = y0 + (i // 2) * (sw + gap)
        cv2.rectangle(img, (cx, cy), (cx+sw, cy+sw), col, -1)
        cv2.rectangle(img, (cx, cy), (cx+sw, cy+sw), (60,60,60), 1)
        buttons[f'color_{i}'] = (cx, cy, sw, sw)
    y0 += ((len(COLOR_PALETTE)+1)//2)*(sw+gap) + 20

    # Backgrounds
    cv2.putText(img, " Backgrounds", (x0, y0 + 20),
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (50,50,50), 2)
    y0 += 40
    sw2 = 40
    for i, col in enumerate(BG_PALETTE):
        cx = x0 + (i % 3)*(sw2 + gap)
        cy = y0 + (i // 3)*(sw2 + gap)
        cv2.rectangle(img, (cx, cy), (cx+sw2, cy+sw2), col, -1)
        cv2.rectangle(img, (cx, cy), (cx+sw2, cy+sw2), (60,60,60), 1)
        buttons[f'bg_{i}'] = (cx, cy, sw2, sw2)
    y0 += 100

    # Buttons
    def draw_button(label, color, key):
        nonlocal y0
        btn = (x0, y0, TOOL_W - 2*x0, 45)
        cv2.rectangle(img, (btn[0], btn[1]), (btn[0]+btn[2], btn[1]+btn[3]), color, -1)
        cv2.putText(img, label, (btn[0]+15, btn[1]+30),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255,255,255), 2)
        buttons[key] = btn
        y0 += 60

    draw_button(" Clear Canvas", (0, 102, 204), 'clear')
    draw_button(" Save Artwork", (34, 139, 34), 'save')
    draw_button(" Help", (100, 100, 100), 'help')

# ----------------------------
# Utility
# ----------------------------
def point_in_rect(px, py, rect):
    x, y, w, h = rect
    return x <= px <= x + w and y <= py <= y + h

def handle_toolbar_click(px, py):
    global brush_size, current_color, background_color, canvas, help_visible, notification_text, notification_time
    for key, rect in buttons.items():
        if point_in_rect(px, py, rect):
            if key == 'size_plus':
                brush_size = min(80, brush_size + 2)
                notification_text = f"Brush Size: {brush_size}"
            elif key == 'size_minus':
                brush_size = max(2, brush_size - 2)
                notification_text = f"Brush Size: {brush_size}"
            elif key.startswith('color_'):
                current_color = COLOR_PALETTE[int(key.split('_')[1])]
                notification_text = "Color Changed"
            elif key.startswith('bg_'):
                background_color = BG_PALETTE[int(key.split('_')[1])]
                canvas[:] = background_color
                notification_text = "Background Changed"
            elif key == 'clear':
                canvas[:] = background_color
                notification_text = "Canvas Cleared"
            elif key == 'save':
                save_dir = "./saved_artworks"
                os.makedirs(save_dir, exist_ok=True)
                ts = int(time.time())
                filename = os.path.join(save_dir, f"artwork_{ts}.png")
                success = cv2.imwrite(filename, canvas)
                notification_text = f"Saved as artwork_{ts}.png" if success else f"Failed to Save"
            elif key == 'help':
                help_visible = True
                notification_text = "Help Opened"
            notification_time = time.time()
            break

def mouse_cb(event, x, y, flags, param):
    global help_visible
    if event == cv2.EVENT_LBUTTONDOWN:
        if help_visible:
            help_visible = False
            return
        handle_toolbar_click(x, y)

# ----------------------------
# Main Loop
# ----------------------------
def main_loop():
    global prev_point, canvas, help_visible, brush_size, current_color, paused, background_color
    global notification_text, notification_time

    cv2.namedWindow("AI Virtual Painter", cv2.WINDOW_NORMAL)
    cv2.setWindowProperty("AI Virtual Painter", cv2.WND_PROP_FULLSCREEN, cv2.WINDOW_FULLSCREEN)
    cv2.setMouseCallback("AI Virtual Painter", mouse_cb)

    hands = mp_hands.Hands(max_num_hands=2, min_detection_confidence=0.7, min_tracking_confidence=0.7)
    last_colour=0
    cooldown=0.5

    while True:
        ret, frame = cap.read()
        if not ret:
            break
        frame = cv2.flip(frame, 1)
        frame_resized = cv2.resize(frame, (CANVAS_W, CANVAS_H))
        rgb = cv2.cvtColor(frame_resized, cv2.COLOR_BGR2RGB)
        result = hands.process(rgb)

        drawing_point = None
        gesture_performed = False
        left_open = False
        right_hand = None

        if result.multi_hand_landmarks and result.multi_handedness:
            hands_state = []

            for idx, hand_label in enumerate(result.multi_handedness):
                label = hand_label.classification[0].label
                hand = result.multi_hand_landmarks[idx]
                h, w, _ = frame_resized.shape
                lm = [(int(p.x * w), int(p.y * h)) for p in hand.landmark]

                def finger_up(tip, pip):
                    return (lm[pip][1] - lm[tip][1]) > 20

                index_up = finger_up(8, 6)
                middle_up = finger_up(12, 10)
                ring_up = finger_up(16, 14)
                pinky_up = finger_up(20, 18)
                thumb_up = (lm[4][0] > lm[3][0])

                all_open = all([thumb_up, index_up, middle_up, ring_up, pinky_up])
                all_closed = not any([thumb_up, index_up, middle_up, ring_up, pinky_up])

                hands_state.append((label, all_open, all_closed, thumb_up, index_up, middle_up, ring_up, pinky_up, lm))

            left_hand = next((h for h in hands_state if h[0] == "Left"), None)
            right_hand = next((h for h in hands_state if h[0] == "Right"), None)

            # LEFT HAND FULL OPEN → PAUSE EVERYTHING
            if left_hand and left_hand[1]:
                paused = True
                prev_point=None
            else:
                paused = False

            # Two-hands gestures (only if not paused)
            if not paused and left_hand and right_hand:
                l_thumb, l_index, l_middle, l_ring, l_little = left_hand[3], left_hand[4], left_hand[5], left_hand[6], left_hand[7]
                r_thumb, r_index, r_middle, r_ring, r_little = right_hand[3], right_hand[4], right_hand[5], right_hand[6], right_hand[7]

                # Background change → middle + ring + little fingers up both hands
                if l_middle and l_ring and l_little and not l_thumb and not l_index and \
                   r_middle and r_ring and r_little and not r_thumb and not r_index:
                    bg_idx = (BG_PALETTE.index(background_color) + 1) % len(BG_PALETTE)
                    background_color = BG_PALETTE[bg_idx]
                    canvas[:] = background_color
                    cv2.putText(frame_resized, " Background Changed", (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 3)
                    gesture_performed = True
                    time.sleep(0.5)

                # Brush increase → both index fingers only
                elif l_index and r_index and not any([l_thumb, l_middle, l_ring, l_little, r_thumb, r_middle, r_ring, r_little]) and not gesture_performed:
                    brush_size = min(brush_size + 2, 80)
                    cv2.putText(frame_resized, f"Brush + ({brush_size})", (50, 100), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 3)
                    gesture_performed = True
                    time.sleep(0.3)

                # Brush decrease → both little fingers only
                elif l_little and r_little and not any([l_thumb, l_index, l_middle, l_ring, r_thumb, r_index, r_middle, r_ring]) and not gesture_performed:
                    brush_size = max(brush_size - 2, 2)
                    cv2.putText(frame_resized, f"Brush - ({brush_size})", (50, 150), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 255), 3)
                    gesture_performed = True
                    time.sleep(0.3)

            # Right-hand single actions (only if not paused)
            if right_hand and not gesture_performed and not paused:
                lm, thumb_up, index_up, middle_up, ring_up, pinky_up = right_hand[8], right_hand[3], right_hand[4], right_hand[5], right_hand[6], right_hand[7]
                index_x, index_y = right_hand[8][8]
                drawing_point = (index_x, index_y)

                # Drawing → index finger only
                if index_up and not middle_up:
                    if prev_point is not None:
                        cv2.line(canvas, prev_point, drawing_point, current_color, brush_size, cv2.LINE_AA)
                    prev_point = drawing_point

                # Erase entire canvas → four fingers up except thumb
                elif index_up and middle_up and ring_up and pinky_up and not thumb_up:
                    canvas[:] = background_color
                    cv2.putText(frame_resized, "🧹 Canvas Cleared", (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 3)
                    prev_point = None
                    gesture_performed = True
                    time.sleep(0.5)

                # Erasing while drawing → full fist
                elif not any([thumb_up, index_up, middle_up, ring_up, pinky_up]):
                    cv2.circle(canvas, drawing_point, eraser_size, background_color, -1)
                    prev_point = None
                
                # Color change → index + middle
                elif index_up and middle_up and not thumb_up:
                    current_time=time.time()
                    if current_time-last_colour>cooldown:
                        color_idx = (COLOR_PALETTE.index(current_color) + 1) % len(COLOR_PALETTE)
                        current_color = COLOR_PALETTE[color_idx]
                        cv2.putText(frame_resized, "Color Changed", (230, 165), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 3)
                        prev_point = None
                        last_colour=current_time

                else:
                    prev_point = None
        else:
            prev_point = None

        # Compose the Window
        window = np.zeros((WIN_H, WIN_W, 3), dtype=np.uint8)
        draw_toolbar(window)
        window[0:CANVAS_H, TOOL_W:WIN_W] = canvas.copy()

        # Current color indicator
        cv2.rectangle(window, (TOOL_W + 10, 10), (TOOL_W + 60, 60), current_color, -1)
        cv2.putText(window, "Current Color", (TOOL_W + 70, 45), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255,255,255), 2)

        # Notification
        if notification_text and (time.time() - notification_time) < 1.5:
            cv2.putText(window, notification_text, (TOOL_W + 20, 90), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0,0,0), 2)
        else:
            notification_text = ""

        small = cv2.resize(frame_resized, (VIDEO_SMALL_W, VIDEO_SMALL_H))
        small_canvas = cv2.resize(canvas, (VIDEO_SMALL_W, VIDEO_SMALL_H))
        overlay = cv2.addWeighted(small, 0.6, small_canvas, 0.4, 0)
        window[10:10+VIDEO_SMALL_H, WIN_W - VIDEO_SMALL_W - 10:WIN_W - 10] = overlay

        if paused:
            cv2.putText(window, "LEFT HAND OPEN - PAUSED", (TOOL_W + 100, 60), cv2.FONT_HERSHEY_SIMPLEX, 1.2, (0, 0, 255), 3)

        # ----------------------------
        # Draw Help Overlay if visible
        # ----------------------------
        if help_visible:
            help_overlay = window.copy()
            alpha = 0.9
            cv2.rectangle(help_overlay, (TOOL_W + 50, 50), (WIN_W - 50, WIN_H - 50), (50, 50, 50), -1)
            cv2.addWeighted(help_overlay, alpha, window, 1 - alpha, 0, window)

            help_texts = [
                "AI Virtual Painter - Help",
                "Draw: Raise Index Finger",
                "Erase: Make a Fist",
                "Change Brush Size: Both Index (Increase) / Both Little (Decrease)",
                "Change Color: Index + Middle Finger",
                "Change Background: Middle+Ring+Little Fingers Both Hands",
                "Clear Canvas: Four Fingers Up (Right Hand)",
                "Pause: Open Left Hand"
            ]
            y0 = 100
            for line in help_texts:
                cv2.putText(window, line, (TOOL_W + 80, y0), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)
                y0 += 50

        cv2.putText(window, f"Draw: Index | Erase: Fist | Brush: {brush_size}", (20, WIN_H - 20), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (80,80,80), 2)

        cv2.imshow("AI Virtual Painter", window)
        key = cv2.waitKey(1) & 0xFF
        if key == 27:  # ESC
            break

    hands.close()
    cap.release()
    cv2.destroyAllWindows()





# ----------------------------
# Run
# ----------------------------
if __name__ == "__main__":
    main_loop()

