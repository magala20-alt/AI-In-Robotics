🧠 Project: AERIOUS INTERFACE

This project combines a FastAI-trained deep learning model with a Tkinter graphical interface to predict and display objects from either:

A webcam feed (real-time or manual capture), or

A manually uploaded image.

The interface shows:

The uploaded/captured image

The predicted label

The confidence score

The dominant color of the image

## ⚙️ Features

✅ Upload an image and get instant predictions

✅ Capture directly from your webcam

✅ Live real-time prediction (optional continuous detection)

✅ GPU acceleration supported (if available)

✅ Simple Tkinter interface

## How to Use system

✔️ Click Upload image to open file explorer and predict image.

✔️ Click "open webcam", allow permission from antivirus and click and drag to draw bounding box on object.

✔️ Press space bar to capture image and activate prediction.

✔️ Click "Close system" to exit the system

## 📦 Requirements

Given the model.ipynb and Front end.py is already in the code submission link you can opt to download the dataset from the repo:

[https://github.com/magala20-alt/AI-In-Robotics]

Create a file named ".env file" in the project root

create variables:

      DATASET_PATH= " "

      TEST_PATH= " "

- **DATASET_PATH** should point to the folder containing the main **OfficeHomeDataset_1** dataset (used for training and validation).
- **TEST_PATH** should point to the folder containing the **Test Data** (used for testing the model).
  For example:
  DATASET_PATH="./OfficeHomeDataset_1"
  TEST_PATH="./Test Data"

# Otherwise .....

1️⃣ Install Python
Make sure you have Python 3.8+ installed.
You can check with:
python --version

2️⃣ Clone this repository
git clone [https://github.com/magala20-alt/AI-In-Robotics]
cd AI-In-Robotics

3️⃣ Create and activate a virtual environment
python -m venv venv

On Windows:
venv\Scripts\activate

On Mac/Linux:
source venv/bin/activate

4️⃣ Install dependencies
pip install -r requirements.txt

If you don’t have a requirements.txt, here’s what to include:

fastai==2.7.15

torch>=2.0.0

torchvision

opencv-python

pillow

tk

To run
python Front end.py
