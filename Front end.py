import tkinter as tk
from tkinter import filedialog, messagebox
from PIL import Image, ImageTk, UnidentifiedImageError
import cv2


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


# --- Root Window ---
root = tk.Tk()
root.title("🤖 AI Robot Interface")
root.geometry("600x500")
root.config(bg="#1a1a1a")


# --- Hover Effects ---
def on_enter(e):
    e.widget["bg"] = "#1e90ff"


def on_leave(e):
    e.widget["bg"] = "#00bfff"


# --- Close App ---
def close_app():
    root.destroy()


# --- Welcome Frame ---
def show_main_screen():
    welcome_frame.pack_forget()
    main_frame.pack(fill="both", expand=True)


welcome_frame = tk.Frame(root, bg="#1a1a1a")
welcome_label = tk.Label(
    welcome_frame,
    text="🤖 WELCOME TO AI LAB 🤖",
    font=("Orbitron", 24, "bold"),
    fg="#00ffff",
    bg="#1a1a1a"
)
welcome_label.pack(pady=40)

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

main_label = tk.Label(
    main_frame,
    text="Select Operation Mode:",
    font=("Consolas", 20, "bold"),
    fg="#00ffff",
    bg="#1a1a1a"
)
main_label.pack(pady=30)

   

# --- Functions for Next Screen ---
def open_webcam():
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        messagebox.showerror("Error", "Cannot open webcam")
        return

    while True:
        ret, frame = cap.read()
        if not ret:
            break
        cv2.imshow("Webcam - Press Q to close", frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
    cap.release()
    cv2.destroyAllWindows()


# Keep a global reference to container_frame
container_frame = None

# --- Container Frame (global) ---
container_frame = tk.Frame(root, bg="#1a1a1a")
container_frame.pack_forget()  # hide initially

# --- Top Frame for Image and Details ---
top_frame = tk.Frame(container_frame, bg="#1a1a1a")
top_frame.pack(side="top", fill="both", expand=True)

img_label = tk.Label(top_frame, bg="#1a1a1a")
img_label.pack(pady=10)

details_label = tk.Label(
    top_frame,
    text="Image details will appear here...",
    font=("Consolas", 12),
    fg="#00ffff",
    bg="#1a1a1a",
    justify="left"
)
details_label.pack(anchor="w", pady=10)

# --- Bottom Buttons Frame ---
bottom_frame = tk.Frame(container_frame, bg="#1a1a1a")
bottom_frame.pack(side="bottom", pady=20)

def go_previous():
    container_frame.pack_forget()
    main_frame.pack(fill="both", expand=True)

prev_btn = tk.Button(
    bottom_frame,
    text="Previous",
    font=("Consolas", 12, "bold"),
    bg="#00bfff",
    fg="white",
    width=12,
    command=go_previous
)
prev_btn.bind("<Enter>", on_enter)
prev_btn.bind("<Leave>", on_leave)
prev_btn.pack(side="left", padx=10)

close_btn = tk.Button(
    bottom_frame,
    text="Cancel",
    font=("Consolas", 12, "bold"),
    bg="#00bfff",
    fg="white",
    width=12,
    command=close_app
)
close_btn.bind("<Enter>", on_enter)
close_btn.bind("<Leave>", on_leave)
close_btn.pack(side="left", padx=10)

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

    # Hide main screen
    main_frame.pack_forget()

    # Update image
    img.thumbnail((500, 300))
    img_tk = ImageTk.PhotoImage(img)
    img_label.config(image=img_tk)
    img_label.image = img_tk

    # Update details
    details_label.config(
        text=f"Image name: {file_path.split('/')[-1]}\n"
             f"Image colour: TBD\n"
             f"Image size: {img.size}\n"
             f"Confidence: TBD"
    )

    # Show container frame (buttons are already packed once)
    if not container_frame.winfo_ismapped():
        container_frame.pack(fill="both", expand=True, padx=20, pady=20)

    
# --- Helper for uniform button creation ---
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
    frame.pack(pady=15)
    return frame


# --- Buttons (Aligned Vertically, Equal Size) ---
create_button_with_info(main_frame, "UPLOAD IMAGE", upload_image_next_screen,
                        "Acceptable formats: JPG, JPEG, PNG, BMP.")
create_button_with_info(main_frame, "USE WEBCAM", open_webcam, "Activate webcam feed (press Q to close).")
create_button_with_info(main_frame, "CLOSE SYSTEM", close_app, "Exit the AI Robot Interface.")

root.mainloop()


