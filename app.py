import os
import streamlit as st
import torch
import torch.nn as nn
from torchvision import transforms, datasets
from PIL import Image
import zipfile


TEST_FOLDER = "test_image"
os.makedirs(TEST_FOLDER, exist_ok=True)


class SimpleCNN(nn.Module):

    def __init__(self):

        super().__init__()

        self.conv1 = nn.Conv2d(1, 32, 3, padding=1)
        self.conv2 = nn.Conv2d(32, 64, 3, padding=1)

        self.pool = nn.MaxPool2d(2)
        self.relu = nn.ReLU()

        self.fc1 = nn.Linear(64 * 7 * 7, 128)
        self.fc2 = nn.Linear(128, 10)

        self.dropout = nn.Dropout(0.25)

    def forward(self, x):

        x = self.pool(self.relu(self.conv1(x)))
        x = self.pool(self.relu(self.conv2(x)))

        x = x.view(x.size(0), -1)

        x = self.relu(self.fc1(x))

        x = self.dropout(x)

        return self.fc2(x)


@st.cache_resource
def load_model():

    device = torch.device(
        "cuda" if torch.cuda.is_available()
        else "cpu"
    )

    model = SimpleCNN().to(device)

    model.load_state_dict(
        torch.load(
            "models/cnn_model.pth",
            map_location=device
        )
    )

    model.eval()

    return model, device


model, device = load_model()


@st.cache_data
def create_test_images():

    test = datasets.MNIST(
        root="./data",
        train=False,
        download=True
    )

    created = []

    for i in range(20):

        img, label = test[i]

        path = f"{TEST_FOLDER}/digit_{label}_{i}.png"

        if not os.path.exists(path):
            img.save(path)

        created.append(path)

    zip_name = "test_images.zip"

    with zipfile.ZipFile(
        zip_name,
        "w"
    ) as z:

        for file in created:
            z.write(
                file,
                arcname=os.path.basename(file)
            )

    return zip_name



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

        out = model(image)

        probs = torch.softmax(
            out,
            dim=1
        )

        pred = torch.argmax(
            probs
        ).item()

    return pred, probs[0]



left, right = st.columns([6, 1])

with left:
    st.title("Handwritten Digit Classifier")

with right:

    zip_file = create_test_images()

    with open(
        zip_file,
        "rb"
    ) as f:

        st.download_button(
            label="Download Test Images",
            data=f,
            file_name="mnist_test_images.zip",
            mime="application/zip"
        )


st.write(
    "Download sample images or upload your own."
)


uploaded = st.file_uploader(
    "Upload digit image",
    type=["png", "jpg", "jpeg"]
)

if uploaded:

    image = Image.open(uploaded)

    st.image(
        image,
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