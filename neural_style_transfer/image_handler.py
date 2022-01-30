from functools import lru_cache

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
    """
    This function is used to easily plot the original input images.
    """

    # Initialise plot configuration
    image_dict = {
        "Content image": content_image,
        "Style image": style_image,
    }

    # Build plot:
    _, axes = plt.subplots(1, 2, figsize=(10, 5))

    for ax, (image_name, image) in zip(axes, image_dict.items()):
        ax.imshow(image)
        ax.set_title(image_name, fontsize=18)
        ax.axis("off")

    plt.tight_layout()
    plt.show()


@lru_cache()  # Do not re-compute
def get_loader() -> Compose:
    """
    This function transforms an input image and converts it to a tensor.
    (with the same pre-processing for a generic model available on
    in torchvision.models that's been pre-trained on ImageNet)
    """
    return Compose(
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
    """
    This function transforms the image, re-shapes it and then attaches it
    to the chosen device (either CPU or GPU) ready for usage in a model.
    """
    loader = get_loader()

    # Fake batch dimension required to fit network's input dimensions:
    image = loader(image).unsqueeze(0)

    return image.to(settings.DEVICE, torch.float)
