🧠 Project: AERIOUS INTERFACE

This project combines a FastAI-trained deep learning model with a Tkinter graphical interface to predict and display objects from either:

  A webcam feed (real-time or manual capture), or
  
  A manually uploaded image.

The interface shows:

  The uploaded/captured image
  
  The predicted label
  
  The confidence score
  
  The dominant color of the image

 ##  ⚙️ Features

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
Given the model.ipynd and Front end.py is already in the code submission link you can opt to download the dataset from the repo:

[https://github.com/magala20-alt/AI-In-Robotics]

Create a .env file

  create variables:
  
      DATASET_PATH= " "
      
      TEST_PATH= " "
      
  Fill corresponding filepaths 
  
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

