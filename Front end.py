import tkinter as tk
from datetime import time
from tkinter import filedialog, messagebox
from PIL import Image, ImageTk, UnidentifiedImageError
import cv2
import numpy as np
from fastai.vision.all import *
import time
import torch
import threading

# Load model
learn = load_learner('OfficeHomeDataset_1/model3.pkl')
learn.to('cpu')
print(f"Model type: {type(learn.model)}")
print(f"Classes: {learn.dls.vocab if hasattr(learn.dls, 'vocab') else 'No vocab found'}")


# ---- Preparing image for model (FIXED VERSION) ----
def prepare_image_for_model(cv_img, learn, resize_size=460, crop_size=224):
    """
    Manually preprocess image to match training pipeline:
    1. Resize to 460x460 with padding (item_tfms)
    2. Center crop to 224x224 (batch_tfms)
    3. Normalize with ImageNet stats (batch_tfms)
    """

    # 1️⃣ Validate input
    if cv_img is None or cv_img.size == 0:
        raise ValueError("Received empty image")

    # 2️⃣ Convert BGR → RGB
    img_rgb = cv2.cvtColor(cv_img, cv2.COLOR_BGR2RGB)

    # 3️⃣ Convert to PIL
    pil_img = Image.fromarray(img_rgb)

    # 4️⃣ Resize with padding to 460x460 (match training)
    w, h = pil_img.size
    if w > h:
        new_w = resize_size
        new_h = int(h * resize_size / w)
    else:
        new_h = resize_size
        new_w = int(w * resize_size / h)

    pil_img = pil_img.resize((new_w, new_h), Image.BILINEAR)

    # Add zero padding
    padded_img = Image.new('RGB', (resize_size, resize_size), (0, 0, 0))
    paste_x = (resize_size - new_w) // 2
    paste_y = (resize_size - new_h) // 2
    padded_img.paste(pil_img, (paste_x, paste_y))

    # 5️⃣ Convert to tensor and normalize
    img_array = np.array(padded_img).astype(np.float32) / 255.0
    img_tensor = torch.from_numpy(img_array).permute(2, 0, 1)

    # Apply ImageNet normalization
    mean = torch.tensor([0.485, 0.456, 0.406]).view(3, 1, 1)
    std = torch.tensor([0.229, 0.224, 0.225]).view(3, 1, 1)
    img_tensor = (img_tensor - mean) / std

    # 6️⃣ Center crop to 224x224 (match training)
    c, h, w = img_tensor.shape
    start_h = (h - crop_size) // 2
    start_w = (w - crop_size) // 2
    img_tensor = img_tensor[:, start_h:start_h + crop_size, start_w:start_w + crop_size]

    # 7️⃣ Add batch dimension
    img_tensor = img_tensor.unsqueeze(0)

    return img_tensor, padded_img


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
    button.pack(side="left", padx=(0, 5))

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


# --- FIXED: Upload Image Function ---
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

    # Store original size
    original_size = img.size

    # --- Display the uploaded image ---
    img_display = img.copy()
    img_display.thumbnail((500, 300))
    img_tk = ImageTk.PhotoImage(img_display)
    img_label.config(image=img_tk)
    img_label.image = img_tk

    # --- Run model prediction with manual preprocessing ---
    try:
        # Load with OpenCV
        cv_img = cv2.imread(file_path)
        if cv_img is None:
            raise ValueError("Failed to load image")

        # Manually preprocess (bypass broken transforms)
        img_tensor, processed_img = prepare_image_for_model(cv_img, learn, resize_size=460, crop_size=224)

        # Direct model inference
        learn.model.eval()
        with torch.no_grad():
            output = learn.model(img_tensor.to('cpu'))
            probs = torch.nn.functional.softmax(output[0], dim=0)
            pred_idx = probs.argmax().item()
            confidence = float(probs[pred_idx]) * 100

        # Get class label
        if hasattr(learn.dls, 'vocab'):
            pred_label = str(learn.dls.vocab[pred_idx])
        else:
            pred_label = f"Class_{pred_idx}"

        print(f"✅ Prediction: {pred_label}, Confidence: {confidence:.2f}%")

    except Exception as e:
        import traceback
        error_msg = f"Prediction Error:\n{str(e)}\n\n{traceback.format_exc()}"
        print(error_msg)
        messagebox.showerror("Prediction Error", error_msg)
        return

    # --- Detect dominant color ---
    img_rgb = cv2.cvtColor(cv_img, cv2.COLOR_BGR2RGB)
    avg_color = tuple(map(int, cv2.mean(img_rgb)[:3]))
    dominant_color = f"RGB{avg_color}"

    # --- Update details label ---
    details_label.config(
        text=f"Prediction: {pred_label}\n"
             f"Confidence: {confidence:.2f}%\n"
             f"Dominant Colour: {dominant_color}\n"
             f"Image Size: {original_size}"
    )

    main_frame.pack_forget()
    detail_frame.pack(fill="both", expand=True)


# --- FIXED: Webcam Function ---
def open_webcam():
    def webcam_thread():
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

        rect_start = None
        rect_end = None
        drawing = False
        captured_image = None
        frame_for_draw = None

        last_prediction_time = time.time()
        prediction_interval = 1.0
        last_pred_label = ""
        last_confidence = 0.0

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
                continue

            display_frame = frame.copy()

            # Crosshair
            h, w, _ = frame.shape
            cv2.line(display_frame, (w // 2 - 20, h // 2), (w // 2 + 20, h // 2), (0, 255, 255), 1)
            cv2.line(display_frame, (w // 2, h // 2 - 20), (w // 2, h // 2 + 20), (0, 255, 255), 1)

            cv2.setMouseCallback("Webcam Feed", draw_rectangle, frame)

            if frame_for_draw is not None and drawing:
                display_frame = frame_for_draw

            # --- Live Prediction (FIXED) ---
            current_time = time.time()
            if rect_start and rect_end and (current_time - last_prediction_time > prediction_interval):
                try:
                    x1, y1 = rect_start
                    x2, y2 = rect_end
                    x_min, x_max = sorted([x1, x2])
                    y_min, y_max = sorted([y1, y2])
                    cropped_frame = frame[y_min:y_max, x_min:x_max]

                    if cropped_frame is not None and cropped_frame.size > 0:
                        # Manual preprocessing
                        img_tensor, _ = prepare_image_for_model(cropped_frame, learn, resize_size=460, crop_size=224)

                        # Direct inference
                        learn.model.eval()
                        with torch.no_grad():
                            output = learn.model(img_tensor.to('cpu'))
                            probs = torch.nn.functional.softmax(output[0], dim=0)
                            pred_idx = probs.argmax().item()

                            if hasattr(learn.dls, 'vocab'):
                                last_pred_label = str(learn.dls.vocab[pred_idx])
                            else:
                                last_pred_label = f"Class_{pred_idx}"

                            last_confidence = float(probs[pred_idx]) * 100
                            last_prediction_time = current_time
                except Exception as e:
                    print(f"Prediction error: {e}")

            # Display label
            if last_pred_label:
                cv2.putText(display_frame,
                            f"{last_pred_label} ({last_confidence:.1f}%)",
                            (20, 40), cv2.FONT_HERSHEY_SIMPLEX,
                            1, (0, 255, 0), 2, cv2.LINE_AA)

            cv2.imshow("Webcam Feed", display_frame)
            key = cv2.waitKey(1) & 0xFF

            if key == ord('q'):
                captured_image = None
                break
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

            if cv2.getWindowProperty("Webcam Feed", cv2.WND_PROP_VISIBLE) < 1:
                break

        cam.release()
        cv2.destroyAllWindows()

        # Process captured image (FIXED)
        if captured_image is not None and captured_image.size > 0:
            try:
                img_tensor, _ = prepare_image_for_model(captured_image, learn, resize_size=460, crop_size=224)

                learn.model.eval()
                with torch.no_grad():
                    output = learn.model(img_tensor.to('cpu'))
                    probs = torch.nn.functional.softmax(output[0], dim=0)
                    pred_idx = probs.argmax().item()

                    if hasattr(learn.dls, 'vocab'):
                        pred_label = str(learn.dls.vocab[pred_idx])
                    else:
                        pred_label = f"Class_{pred_idx}"

                    confidence = float(probs[pred_idx]) * 100

                img_rgb = cv2.cvtColor(captured_image, cv2.COLOR_BGR2RGB)
                avg_color = tuple(map(int, cv2.mean(img_rgb)[:3]))
                dominant_color = f"RGB{avg_color}"

                pil_img = Image.fromarray(img_rgb)
                img_display = pil_img.copy()
                img_display.thumbnail((500, 300))
                img_tk = ImageTk.PhotoImage(img_display)
                img_label.config(image=img_tk)
                img_label.image = img_tk

                details_label.config(
                    text=f"Prediction: {pred_label}\n"
                         f"Confidence: {confidence:.2f}%\n"
                         f"Dominant Colour: {dominant_color}\n"
                         f"Image Size: {pil_img.size}"
                )

                main_frame.pack_forget()
                detail_frame.pack(fill="both", expand=True)
            except Exception as e:
                print(f"Error processing captured image: {e}")

    threading.Thread(target=webcam_thread, daemon=True).start()


# --- Main Menu Buttons ---
create_button_with_info(main_frame, "UPLOAD IMAGE", upload_image_next_screen,
                        "Acceptable formats: JPG, JPEG, PNG, BMP.")
create_button_with_info(main_frame, "USE WEBCAM", open_webcam,
                        "Activate webcam feed (press Q to close).")
create_button_with_info(main_frame, "CLOSE SYSTEM", close_app,
                        "Exit the AI Robot Interface.")

root.mainloop()