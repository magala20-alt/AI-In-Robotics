import tkinter as tk
from datetime import time
from tkinter import filedialog, messagebox
from PIL import Image, ImageTk, UnidentifiedImageError
import cv2


import numpy as np
from fastai.vision.all import *
import cv2, time
from fastai.vision.all import PILImage, Resize, ResizeMethod
import torch

# Load model
learn = load_learner('OfficeHomeDataset_10072016/model.pkl')
learn.to('cpu')

# ---- preparing image for model ----
def prepare_model_for_model (cv_img,learn, resize_size=460):
    # Convert BGR to RGB
    img_rgb = cv2.cvtColor(cv_img, cv2.COLOR_BGR2RGB)

    # Convert to FastAI PILImage
    pil_img = PILImage.create(img_rgb)

    # Apply same resizing transform as training
    resize_tfm = Resize(resize_size, method=ResizeMethod.Pad, pad_mode='zeros')
    fastai_img = resize_tfm(pil_img)

    # Create a test DataLoader to apply all dls transforms correctly
    dl = learn.dls.test_dl([fastai_img])

    return dl, fastai_img



# --- Main interface -----------
root = tk.Tk()
root.title("🤖 Aerius Interface")
root.geometry("600x500")
root.config(bg="#1a1a1a")

# --- Hover Effects ---
def on_enter(e):
    e.widget["bg"] = "#1e90ff"

def on_leave(e):
    e.widget["bg"] = "#00bfff"

# --- Tooltip Class ---
class ToolTip:
    def __init__(self, widget, text):
        self.widget = widget
        self.text = text
        self.tip_window = None
        widget.bind("<Enter>", self.show_tip)
        widget.bind("<Leave>", self.hide_tip)

    def show_tip(self, event=None):
        if self.tip_window or not self.text:
            return
        x, y, _, cy = self.widget.bbox("insert")
        x += self.widget.winfo_rootx() + 25
        y += self.widget.winfo_rooty() + cy + 25
        self.tip_window = tw = tk.Toplevel(self.widget)
        tw.wm_overrideredirect(True)
        tw.wm_geometry(f"+{x}+{y}")
        label = tk.Label(
            tw,
            text=self.text,
            justify="left",
            background="#0ff",
            fg="black",
            relief="solid",
            borderwidth=1,
            font=("Consolas", 10)
        )
        label.pack(ipadx=5, ipady=3)

    def hide_tip(self, event=None):
        tw = self.tip_window
        self.tip_window = None
        if tw:
            tw.destroy()


# --- Close App ---
def close_app():
    root.destroy()


# --- Welcome Frame (Front Page) ---
welcome_frame = tk.Frame(root, bg="#1a1a1a")
welcome_label = tk.Label(
    welcome_frame,
    text="🤖 WELCOME TO AERIUS 🤖",
    font=("Orbitron", 24, "bold"),
    fg="#00ffff",
    bg="#1a1a1a"
)
welcome_label.pack(pady=40)

def show_main_screen():
    welcome_frame.pack_forget()
    main_frame.pack(fill="both", expand=True)

welcome_button = tk.Button(
    welcome_frame,
    text="ENTER SYSTEM",
    font=("Consolas", 16, "bold"),
    bg="#00bfff",
    fg="white",
    activebackground="#1e90ff",
    command=show_main_screen,
    width=20
)
welcome_button.bind("<Enter>", on_enter)
welcome_button.bind("<Leave>", on_leave)
welcome_button.pack(pady=20)

welcome_frame.pack(fill="both", expand=True)


# --- Main Frame ---
main_frame = tk.Frame(root, bg="#1a1a1a")
main_label = tk.Label(main_frame, text="Select Operation Mode:", font=("Consolas", 20, "bold"),
                      fg="#00ffff", bg="#1a1a1a")
main_label.pack(pady=30)

# --- Helper to create buttons with tooltips ---
def create_button_with_info(parent, text, command, tooltip_text):
    frame = tk.Frame(parent, bg="#1a1a1a")
    button = tk.Button(
        frame,
        text=text,
        font=("Consolas", 14, "bold"),
        bg="#00bfff",
        fg="white",
        command=command,
        width=20
    )
    button.bind("<Enter>", on_enter)
    button.bind("<Leave>", on_leave)
    button.pack(side="left", padx=(0,5))

    info = tk.Label(frame, text="❓", font=("Consolas", 14, "bold"), fg="#00ffff", bg="#1a1a1a")
    info.pack(side="left")
    ToolTip(info, tooltip_text)
    frame.pack(pady=10)
    return frame


# --- Detail Frame ---
detail_frame = tk.Frame(root, bg="#1a1a1a")
img_label = tk.Label(detail_frame, bg="#1a1a1a")
img_label.pack(pady=10)

details_label = tk.Label(detail_frame, text="Image details will appear here...",
                         font=("Consolas", 12), fg="#00ffff", bg="#1a1a1a", justify="left")
details_label.pack(pady=10)

button_frame = tk.Frame(detail_frame, bg="#1a1a1a")
button_frame.pack(pady=20)

def go_previous():
    detail_frame.pack_forget()
    main_frame.pack(fill="both", expand=True)

prev_btn = tk.Button(button_frame, text="Previous", font=("Consolas", 12, "bold"),
                     bg="#00bfff", fg="white", width=12, command=go_previous)
prev_btn.bind("<Enter>", on_enter)
prev_btn.bind("<Leave>", on_leave)
prev_btn.pack(side="left", padx=10)

cancel_btn = tk.Button(button_frame, text="Cancel", font=("Consolas", 12, "bold"),
                       bg="#00bfff", fg="white", width=12, command=close_app)
cancel_btn.bind("<Enter>", on_enter)
cancel_btn.bind("<Leave>", on_leave)
cancel_btn.pack(side="left", padx=10)


def upload_image_next_screen():
    file_path = filedialog.askopenfilename(
        title="Select an image",
        filetypes=[("Image Files", "*.jpg;*.jpeg;*.png;*.bmp")]
    )
    if not file_path:
        return

    try:
        img = Image.open(file_path)
    except UnidentifiedImageError:
        messagebox.showerror("Error", "Invalid image format!")
        return

    # --- Display the uploaded image ---
    img.thumbnail((500, 300))
    img_tk = ImageTk.PhotoImage(img)
    img_label.config(image=img_tk)
    img_label.image = img_tk

    # --- Prepare image for model prediction ---
    dl, fastai_img = prepare_model_for_model(img_tk, learn)

    # --- Run model prediction ---

    # Run prediction
    preds, *_ = learn.get_preds(dl=dl)
    pred_idx = preds.argmax(dim=1)[0].item()
    pred_label = learn.dls.vocab[pred_idx]
    confidence = float(preds[0][pred_idx]) * 100

    # --- Detect dominant color (approximate) ---
    img_cv = cv2.imread(file_path)
    img_cv = cv2.cvtColor(img_cv, cv2.COLOR_BGR2RGB)
    avg_color = cv2.mean(img_cv)[:3]
    avg_color = tuple(map(int, avg_color))
    dominant_color = f"RGB{avg_color}"

    # --- Update details label ---
    details_label.config(
        text=f"Prediction: {pred_label}\n"
             f"Confidence: {confidence:.2f}%\n"
             f"Dominant Colour: {dominant_color}\n"
             f"Image Size: {img.size}"
    )

    main_frame.pack_forget()
    detail_frame.pack(fill="both", expand=True)


# --- Function to open webcam ----
def open_webcam():
    import time
    import torch
    import cv2
    from PIL import Image
    from fastai.vision.all import PILImage

    cam = None
    for i in range(3):
        temp_cam = cv2.VideoCapture(i)
        if temp_cam.isOpened():
            cam = temp_cam
            break

    if not cam:
        messagebox.showerror("Error", "No webcam found!")
        return

    cv2.namedWindow("Webcam Feed")

    # Variables for rectangle drawing
    rect_start = None
    rect_end = None
    drawing = False
    captured_image = None
    frame_for_draw = None

    # Live prediction control
    last_prediction_time = time.time()
    prediction_interval = 1.0  # seconds between predictions
    last_pred_label = ""
    last_confidence = 0.0

    # GPU setup
    device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
    learn.dls.device = device

    # --- Mouse callback ---
    def draw_rectangle(event, x, y, flags, param):
        nonlocal rect_start, rect_end, drawing, frame_for_draw
        frame = param
        if event == cv2.EVENT_LBUTTONDOWN:
            rect_start = (x, y)
            drawing = True
        elif event == cv2.EVENT_MOUSEMOVE and drawing:
            frame_for_draw = frame.copy()
            cv2.rectangle(frame_for_draw, rect_start, (x, y), (0, 255, 0), 2)
        elif event == cv2.EVENT_LBUTTONUP:
            rect_end = (x, y)
            drawing = False

    while True:
        ret, frame = cam.read()
        if not ret or frame is None or frame.size == 0:
            continue  # Skip invalid frames

        display_frame = frame.copy()

        # Draw faint crosshair
        h, w, _ = frame.shape
        cv2.line(display_frame, (w // 2 - 20, h // 2), (w // 2 + 20, h // 2), (0, 255, 255), 1)
        cv2.line(display_frame, (w // 2, h // 2 - 20), (w // 2, h // 2 + 20), (0, 255, 255), 1)

        cv2.setMouseCallback("Webcam Feed", draw_rectangle, frame)

        if frame_for_draw is not None and drawing:
            display_frame = frame_for_draw

        # --- LIVE PREDICTION ---
        current_time = time.time()
        if rect_start and rect_end and (current_time - last_prediction_time > prediction_interval):
            try:
                x1, y1 = rect_start
                x2, y2 = rect_end
                x_min, x_max = sorted([x1, x2])
                y_min, y_max = sorted([y1, y2])
                cropped_frame = frame[y_min:y_max, x_min:x_max]

                if cropped_frame is not None and cropped_frame.size > 0:
                    # Convert BGR → RGB
                    img_rgb = cv2.cvtColor(cropped_frame, cv2.COLOR_BGR2RGB)
                    pil_img = Image.fromarray(img_rgb)
                    resized_pil = pil_img.resize((460, 460))
                    fastai_img = PILImage.create(resized_pil)  # ✅ FastAI PILImage

                    # Predict
                    pred, pred_idx, probs = learn.predict(fastai_img)
                    last_pred_label = str(pred)
                    last_confidence = float(probs[pred_idx]) * 100
                    last_prediction_time = current_time
                else:
                    print("Warning: Bounding box empty, skipping prediction")

            except Exception as e:
                print(f"Prediction error: {e}")

        # Display live prediction text
        if last_pred_label:
            cv2.putText(
                display_frame,
                f"{last_pred_label} ({last_confidence:.1f}%)",
                (20, 40),
                cv2.FONT_HERSHEY_SIMPLEX,
                1,
                (0, 255, 0),
                2,
                cv2.LINE_AA
            )

        cv2.imshow("Webcam Feed", display_frame)
        key = cv2.waitKey(1) & 0xFF

        # Quit webcam
        if key == ord('q'):
            captured_image = None
            break

        # Manual capture
        elif key == ord(' '):
            if rect_start and rect_end:
                x1, y1 = rect_start
                x2, y2 = rect_end
                x_min, x_max = sorted([x1, x2])
                y_min, y_max = sorted([y1, y2])
                captured_image = frame[y_min:y_max, x_min:x_max]
            else:
                captured_image = frame.copy()
            cv2.destroyAllWindows()
            break

        # Window closed manually
        if cv2.getWindowProperty("Webcam Feed", cv2.WND_PROP_VISIBLE) < 1:
            break

    cam.release()
    cv2.destroyAllWindows()

    # --- Process captured image for UI ---
    if captured_image is not None and captured_image.size > 0:
        img_rgb = cv2.cvtColor(captured_image, cv2.COLOR_BGR2RGB)
        pil_img = PILImage.create(img_rgb)
        fastai_img = pil_img.resize((460, 460))

        pred, pred_idx, probs = learn.predict(fastai_img)
        pred_label = str(pred)
        confidence = float(probs[pred_idx]) * 100

        # Dominant color
        avg_color = cv2.mean(cv2.cvtColor(captured_image, cv2.COLOR_BGR2RGB))
        avg_color = tuple(map(int, avg_color))
        dominant_color = f"RGB{avg_color}"

        # Display in Tkinter
        img_display = Image.fromarray(img_rgb)
        img_display.thumbnail((500, 300))
        img_tk = ImageTk.PhotoImage(img_display)
        img_label.config(image=img_tk)
        img_label.image = img_tk

        # Update details
        details_label.config(
            text=f"Prediction: {pred_label}\n"
                 f"Confidence: {confidence:.2f}%\n"
                 f"Dominant Colour: {dominant_color}\n"
                 f"Image Size: {img_display.size}"
        )

        main_frame.pack_forget()
        detail_frame.pack(fill="both", expand=True)






# --- Main Menu Buttons ---
create_button_with_info(main_frame, "UPLOAD IMAGE", upload_image_next_screen,
                        "Acceptable formats: JPG, JPEG, PNG, BMP.")
create_button_with_info(main_frame, "USE WEBCAM", open_webcam,
                        "Activate webcam feed (press Q to close).")
create_button_with_info(main_frame, "CLOSE SYSTEM", close_app,
                        "Exit the AI Robot Interface.")


root.mainloop()
