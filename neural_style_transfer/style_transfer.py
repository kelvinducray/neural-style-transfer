import matplotlib.pyplot as plt
import torch
from PIL import Image
from pydantic import BaseModel
from torch import nn
from torch.nn.functional import mse_loss
from torch.optim import LBFGS, Optimizer
from torchvision.transforms import ToPILImage

from .config import get_settings

settings = get_settings()


class StyleTransferConfig(BaseModel):
    model: nn.Module
    # Note: Your model should be pre-trained on ImageNet and from torchvision.models
    feature_layers: list[int]

    class Config:
        arbitrary_types_allowed = True


class ContentLoss(nn.Module):
    def __init__(
        self,
        target,
    ):
        super(ContentLoss, self).__init__()
        # We 'detach' the target content from the tree used
        # to dynamically compute the gradient: this is a stated value,
        # not a variable. Otherwise the forward method of the criterion
        # will throw an error
        self.target = target.detach()

    def forward(self, input):
        self.loss = mse_loss(input, self.target)
        return input


class StyleLoss(nn.Module):
    def __init__(self, target_feature):
        super(StyleLoss, self).__init__()
        self.target = self.gram_matrix(target_feature).detach()

    @staticmethod
    def gram_matrix(input):
        a, b, c, d = input.size()
        # a: batch size (= 1)
        # b: number of feature maps
        # c & d: dimensions of a f. map (N=c*d)

        features = input.view(a * b, c * d)  # Resize F_XL into \hat F_XL

        G = torch.mm(features, features.t())  # Compute the gram product

        # We 'normalize' the values of the gram matrix
        # by dividing by the number of element in each feature maps
        return G.div(a * b * c * d)

    def forward(self, input):
        G = self.gram_matrix(input)
        self.loss = mse_loss(G, self.target)
        return input


class StyleTransferModel(nn.Module):
    def __init__(self, style_transfer_config: StyleTransferConfig):
        super(StyleTransferModel, self).__init__()

        # Initialise which layers to use for feature extration
        self.feature_layers = style_transfer_config.feature_layers
        max_layer = max(self.feature_layers)

        # Get model and clip based on selected feature layers
        self.model = style_transfer_config.model.features[: max_layer + 1]

    def forward(self, x):
        features = []

        for (layer_num, layer) in enumerate(self.model):
            if layer_num in self.feature_layers:
                features.append(layer(x))

        return features


class StyleTransfer:
    def __init__(
        self,
        content_image: Image,
        style_image: Image,
        style_transfer_model: StyleTransferModel,
        optimiser: Optimizer = LBFGS,
        style_weight: int = 1000000,
        content_weight: int = 1,
    ) -> None:
        # Initialise all the images
        self.content_image = content_image
        self.style_image = style_image
        self.generated_image = self.content_image.clone()
        self.generated_image.requires_grad_(True)

        # Initialise the model and freeze it
        self.model = style_transfer_model.eval()

        # Initialise the optimiser
        self.optimiser = optimiser([self.generated_image])
        self.total_iterations = 0

        # Initialise the hyperparameters
        self.content_weight = content_weight
        self.style_weight = style_weight

        # Initialise style & content losses
        self.content_losses = self.get_content_losses()
        self.style_losses = self.get_style_losses()

        # Initlise transform back to PIL Image
        # (for displaying generated images)
        self.unloader = ToPILImage()

    def get_content_losses(self):
        content_losses = []

        target = self.model(self.content_image).detach()
        content_loss = ContentLoss(target)
        content_losses.append(content_loss)

        return content_losses

    def get_style_losses(self):
        style_losses = []

        target_feature = self.model(self.style_image).detach()
        style_loss = StyleLoss(target_feature)
        style_losses.append(style_loss)

        return style_losses

    def optimisation_step(self, current_iteration_no: int):
        self.model(self.generated_image)
        style_score = 0
        content_score = 0

        for sl in self.style_losses:
            style_score += sl.loss
        for cl in self.content_losses:
            content_score += cl.loss

        style_score *= self.style_weight
        content_score *= self.content_weight

        loss = style_score + content_score

        self.optimiser.zero_grad()
        loss.backward()
        self.optimiser.step()

        if (current_iteration_no + 1) % 50 == 0:
            print(f"Run #{current_iteration_no + 1}")
            print(
                f"Style loss: {style_score.item():4f}, Content loss: {content_score.item():4f}"
            )

        # Correct the values of updated input image
        with torch.no_grad():
            self.generated_image.clamp_(0, 1)

        # Update the total number of iterations performed
        self.total_iterations += 1

        return style_score + content_score

    def optimise(self, iterations: int = 100):
        for i in range(iterations):
            self.optimisation_step(i)

    def show_generated_image(self):
        image_cloned = self.generated_image.cpu().clone()

        # Remove the fake batch dimension:
        image_squeezed = image_cloned.squeeze(0)

        # Convert back to PIL Image
        image_converted = self.unloader(image_squeezed)

        # Display the image
        plt.imshow(image_converted)
        plt.title("Title")
        plt.show()
