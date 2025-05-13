# 🧠 Brain Tumor Detection using ResNet50 and Flask

This project provides an API to detect brain tumors from MRI images using a deep learning model based on **ResNet50**, trained to classify images into four categories: `None`, `Meningioma`, `Glioma`, and `Pituitary`.

## 📁 Project Structure

```
Brain-Tumor-Detection/
│
├── app.py                     # Flask app to serve the model
├── models/
│   └── bt_resnet50_model.pt   # Trained PyTorch model
├── requirements.txt           # Python dependencies
└── README.md                  # Project documentation
```

## 🔍 Model Details

- **Architecture**: Pretrained ResNet50 from `torchvision.models`.
- **Custom Classifier Head**:
  - Linear → SELU → Dropout
  - Linear → SELU → Dropout
  - Output Layer: 4 classes with LogSigmoid activation
- **Classes**:
  - `0`: None (No tumor)
  - `1`: Meningioma
  - `2`: Glioma
  - `3`: Pituitary

## 🚀 How to Run the Project

### 1. Clone the repository

```bash
git clone https://github.com/shrinidhi-a/Brain-Tumor-Detection.git
cd Brain-Tumor-Detection
```

### 2. Set up the environment

```bash
# Create virtual environment (optional but recommended)
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt
```

> Make sure `torch` and `torchvision` are installed with GPU support if CUDA is available.

### 3. Run the Flask server

```bash
python app.py
```

Server will start at `http://127.0.0.1:5000`

## 📸 API Usage

### Endpoint: `/predict`

**Method**: `POST`  
**Payload**: Multipart form-data with key `"file"` and the MRI image file as value.

#### Example using `curl`:

```bash
curl -X POST http://127.0.0.1:5000/predict \
  -F file=@sample_mri.jpg
```

#### Response:
```json
{
  "class_id": "2",
  "class_name": "Glioma"
}
```

## 🧪 Preprocessing

- Image is resized to `512x512`
- Converted to PyTorch tensor
- Unsqueezed to add batch dimension

## 💡 Future Enhancements

- Add support for batch predictions
- Add frontend UI for image upload and result visualization
- Include model training notebook and dataset references

## 📜 License

This project is open-source and available under the [MIT License](LICENSE).

## 🙋‍♂️ Author

**Shrinidhi A**  
🔗 [GitHub Profile](https://github.com/shrinidhi-a)
