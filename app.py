import streamlit as st
import torch
import torch.nn as nn
from torchvision import transforms
from PIL import Image
import numpy as np

class SimpleCNN(nn.Module):
    def __init__(self):
        super(SimpleCNN, self).__init__()

        self.conv1 = nn.Conv2d(
            in_channels=1,
            out_channels=32,
            kernel_size=3,
            padding=1
        )

        self.conv2 = nn.Conv2d(
            in_channels=32,
            out_channels=64,
            kernel_size=3,
            padding=1
        )

        self.relu = nn.ReLU()
        self.pool = nn.MaxPool2d(2)

        self.fc1 = nn.Linear(64 * 7 * 7, 128)
        self.fc2 = nn.Linear(128, 10)

        self.dropout = nn.Dropout(0.25)

    def forward(self, x):

        x = self.pool(self.relu(self.conv1(x)))
        x = self.pool(self.relu(self.conv2(x)))

        x = x.view(x.size(0), -1)
        x=self.fc1(x)
        x = self.relu(x)
        x = self.dropout(x)

        return self.fc2(x)


@st.cache_resource
def load_model():

    device = torch.device(
        "cuda" if torch.cuda.is_available()
        else "cpu"
    )

    model = SimpleCNN()

    model.load_state_dict(
        torch.load(
            "models/cnn_model.pth",
            map_location=device
        )
    )

    model = model.to(device)  

    model.eval()

    return model, device


model, device = load_model()



transform = transforms.Compose([
    transforms.Grayscale(),
    transforms.Resize((28, 28)),
    transforms.ToTensor(),
    transforms.Normalize(
        (0.1307,),
        (0.3081,)
    )
])


def predict(image):

    image = transform(image)

    image = image.unsqueeze(0).to(device)

    with torch.no_grad():

        output = model(image)

        probs = torch.softmax(output, dim=1)

        pred = torch.argmax(probs).item()

    return pred, probs.cpu().numpy()[0]



st.set_page_config(
    page_title="MNIST Digit Classifier",
    page_icon="🔢"
)

st.title("🔢 Handwritten Digit Classifier")

st.write(
    "Upload an image of a handwritten digit (0–9)"
)

uploaded = st.file_uploader(
    "Choose image",
    type=["png", "jpg", "jpeg"]
)

if uploaded:

    image = Image.open(uploaded)

    st.image(
        image,
        caption="Uploaded Image",
        width=250
    )

    if st.button("Predict"):

        pred, probs = predict(image)

        st.success(
            f"Predicted Digit: {pred}"
        )

        st.subheader("Confidence")
   
        confidence = probs[pred]


        st.write(
            f"Confidence: {confidence:.2%}"
        )

        st.progress(float(confidence))