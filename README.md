# Vehicle Detection System Using Faster R-CNN 🚗🚌🚚

This project implements an **image-based vehicle detection system** using the **Faster R-CNN** deep learning object detection algorithm.  
The system is trained to **detect and classify 12 different types of vehicles** from a single image with bounding boxes.

---

## 📌 Project Objective

The main objective of this project is to accurately detect **multiple vehicle categories** in road images using a **Faster R-CNN model**, providing precise localization and classification.

---

## 🚘 Vehicle Classes Detected

The model detects the following **12 vehicle types**:

| Class ID | Vehicle Type |
|--------|--------------|
| 1 | Big Bus |
| 2 | Big Truck |
| 3 | Bus-L |
| 4 | Bus-S |
| 5 | Car |
| 6 | Mid Truck |
| 7 | Small Bus |
| 8 | Small Truck |
| 9 | Truck-L |
| 10 | Truck-M |
| 11 | Truck-S |
| 12 | Truck-XL |

---

## 🧠 Model Used

- **Algorithm**: Faster R-CNN  
- **Backbone Network**: ResNet-50 with Feature Pyramid Network (FPN)  
- **Task**: Multi-class vehicle detection in images  

Faster R-CNN uses a **Region Proposal Network (RPN)** to identify possible object regions and then classifies each region into one of the vehicle categories.

---

## 🗂 Project Structure

```
dataset/ # Vehicle image dataset
fasterrcnn_resnet50_fpn.pth # Trained Faster R-CNN model
model.ipynb # Model loading and inspection
testing.ipynb # Image detection testing
test1.jpg
test2.jpg
test3.jpg
test4.jpg
model_layers.txt # Model architecture details
README.md
```
---

## 📷 Detection Output
- Bounding boxes are drawn around detected vehicles   
- Each box is labeled with:
  - Vehicle class name   
  - Confidence score   
The system can detect **multiple vehicles of different types in a single image**.

---

## 📓 Notebooks Description

| Notebook | Description |
|--------|------------|
| `model.ipynb` | Loads the Faster R-CNN model and inspects layers |
| `testing.ipynb` | Runs detection on test images and visualizes results |
---

## 🛠 Applications
- Traffic monitoring systems   
- Smart city surveillance   
- Intelligent transportation systems   



---
## 👤 Author
Sandaru Nethmina Samarasekara
GitHub: https://github.com/sandaruns2004
