import torch
import torch.nn as nn
from torchvision import models, transforms
from PIL import Image

# 1. Setup Device and Model
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
# Using a standard ResNet50 as the "target" for the attack
model = models.resnet50(pretrained=True).to(device).eval()

# 2. Image Loading & Preprocessing


def load_and_prep(img_path):
    img = Image.open(img_path).convert('RGB')
    # ResNet expects 224x224 and specific normalization
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
    ])
    return transform(img).unsqueeze(0).to(device)

# 3. PGD Attack Logic


def run_pgd(image, eps=0.03, alpha=2/255, steps=40):
    # We create a fake label (e.g., index 1) to "push" the image away from
    target = torch.tensor([1]).to(device)
    adv_image = image.clone().detach()

    # Adding initial random noise within the epsilon ball
    adv_image = adv_image + torch.empty_like(adv_image).uniform_(-eps, eps)
    adv_image = torch.clamp(adv_image, 0, 1).detach()

    for _ in range(steps):
        adv_image.requires_grad = True
        outputs = model(adv_image)
        loss = nn.CrossEntropyLoss()(outputs, target)

        # Calculate gradients
        grad = torch.autograd.grad(loss, adv_image)[0]

        # Iterative update
        adv_image = adv_image.detach() + alpha * grad.sign()

        # Projection step (Keep it within epsilon range of original)
        delta = torch.clamp(adv_image - image, min=-eps, max=eps)
        adv_image = torch.clamp(image + delta, min=0, max=1).detach()

    return adv_image


# 4. Execute and Save
input_tensor = load_and_prep('uploads/uploaded_image.png')
attacked_tensor = run_pgd(input_tensor)

# Convert back to image and save
result_img = transforms.ToPILImage()(attacked_tensor.squeeze(0).cpu())
result_img.save('attacked_image_output.png')

print("Attack complete. Saved as 'attacked_image_output.png'")
