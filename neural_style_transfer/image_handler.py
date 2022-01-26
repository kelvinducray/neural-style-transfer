import matplotlib.pyplot as plt
import torch
from PIL import Image
from torch import Tensor
from torchvision.transforms import Compose, Normalize, Resize, ToTensor

from .config import get_settings

settings = get_settings()


def display_input_images(
    content_image: Image,
    style_image: Image,
) -> None:
    _, axes = plt.subplots(1, 2, figsize=(10, 5))

    axes[0].imshow(content_image)
    axes[0].set_title("Content image", fontsize=18)

    axes[1].imshow(style_image)
    axes[1].set_title("Style image", fontsize=18)

    for ax in axes:
        ax.axis("off")
    plt.tight_layout()
    plt.show()


# Transform the input image and convert it to a tensor
loader = Compose(
    [
        Resize(settings.OUTPUT_IMG_SIZE),  # scale imported image
        ToTensor(),
        Normalize(
            mean=[0.485, 0.456, 0.406],
            std=[0.229, 0.224, 0.225],
        ),
    ],
)


def transform_image(image: Image) -> Tensor:
    # Fake batch dimension required to fit network's input dimensions:
    image = loader(image).unsqueeze(0)
    return image.to(settings.DEVICE, torch.float)
