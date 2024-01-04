import torch
from torchvision import transforms
from PIL import Image
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision import transforms, models, datasets
from sklearn.model_selection import train_test_split

Model_path = "Resnet_Weights.pth"
Classes = ['Athletic shoes', 'Boat', 'Flats', 'Heels', 'Knee High', 'Loafers', 'Oxford', 'Sneakers']

def predict_image(image_path, model_path=Model_path, class_labels=Classes, device='cpu'):
    # Load and preprocess the image
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
    ])

    image = Image.open(image_path).convert('RGB')  # Ensure the image is in RGB format
    input_tensor = transform(image).unsqueeze(0).to(device)  # Add batch dimension and move to the specified device

    # Load the model state dictionary
    state_dict = torch.load(model_path, map_location=device)

    # Create an instance of the model
    model = models.resnet50(pretrained=False)
    model.fc = nn.Linear(model.fc.in_features, len(class_labels))
    model.load_state_dict(state_dict)
    model = model.to(device)  # Move model to the same device as the input tensor

    # Make prediction
    with torch.no_grad():
        model.eval()
        output = model(input_tensor)

    # Get class probabilities and logits
    probabilities = torch.nn.Softmax(dim=1)(output)[0].tolist()
    logits = output[0].tolist()

    # Get the predicted class index
    predicted_index = torch.argmax(output).item()
    predicted_class = class_labels[predicted_index]

    return {
        'predicted_class': str(predicted_class)
    }

# prediction_result = predict_image("Image_1.jpeg", Model_path, Classes)
# print("Predicted Class:", prediction_result['predicted_class'])
# print("Class Probabilities:", prediction_result['probabilities'])
# print("Logits:", prediction_result['logits'])
