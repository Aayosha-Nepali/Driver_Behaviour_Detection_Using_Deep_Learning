from flask import Flask, request, jsonify, render_template
import torch
import torchvision.transforms as transforms
import torchvision.models as models
import torch.nn as nn
from PIL import Image
import io

app = Flask(__name__)

MODEL_PATH = "/Users/linux/Desktop/AI Y3 S2/Computer Vision/Coursework/23048605 Aayosha Nepali/Development/models/resnet18_finetuned_4class.pth"

CLASS_NAMES = [
    "Safe / Talking",
    "Using Phone",
    "Distracted Driving",
    "Reaching / Grooming"
]


device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")

infer_transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406],
                         std=[0.229, 0.224, 0.225])
])

def load_model():
    model = models.resnet18(weights=None)
    model.fc = nn.Linear(model.fc.in_features, 4)
    model.load_state_dict(torch.load(MODEL_PATH, map_location=device))
    model.to(device)
    model.eval()
    return model

print("loading model...")
model = load_model()
print(f"model ready on {device}")


@app.route("/")
def index():
    return render_template("index.html")


@app.route("/predict", methods=["POST"])
def predict():
    if "image" not in request.files:
        return jsonify({"error": "no image uploaded"}), 400

    file = request.files["image"]
    img_bytes = file.read()
    pil_img = Image.open(io.BytesIO(img_bytes)).convert("RGB")

    # run inference
    tensor = infer_transform(pil_img).unsqueeze(0).to(device)
    with torch.no_grad():
        outputs = model(tensor)
        probs = torch.softmax(outputs, dim=1)[0]

    top4_conf, top4_idx = torch.topk(probs, 4)
    top4_idx  = top4_idx.cpu().numpy()
    top4_conf = top4_conf.cpu().numpy()

    results = []
    for i in range(4):
        name = CLASS_NAMES[top4_idx[i]]
        results.append({
            "rank": i + 1,
            "label": name,
            "confidence": round(float(top4_conf[i]) * 100, 1)
        })

    return jsonify({"predictions": results})


if __name__ == "__main__":
    app.run(debug=True, host="0.0.0.0", port=5001)