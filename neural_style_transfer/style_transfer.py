import torch
from PIL import Image
from pydantic import BaseModel
from torch import Tensor
from torch.nn import Module, ReLU
from torch.nn.functional import mse_loss
from torch.optim import LBFGS, Optimizer
from torchvision.transforms import ToPILImage

from .config import get_settings
from .helpers import replace_layers

settings = get_settings()


class StyleTransferConfig(BaseModel):
    model: Module
    # Note: Your model should be pre-trained on
    # ImageNet and from torchvision.models
    feature_layers: list[int]

    class Config:
        arbitrary_types_allowed = True


class ContentLoss(Module):
    def __init__(self, target: Tensor) -> None:
        super(ContentLoss, self).__init__()
        self.target = target.detach()

    def forward(self, input):
        self.loss = mse_loss(input, self.target)
        return input


class StyleLoss(Module):
    def __init__(self, target: Tensor) -> None:
        super(StyleLoss, self).__init__()
        self.target = self.gram_matrix(target).detach()

    @staticmethod
    def gram_matrix(input) -> Tensor:
        (batch_size, no_of_feature_maps, x, y) = input.size()
        # Re-size:
        features = input.view(
            batch_size * no_of_feature_maps,
            x * y,
        )
        # Compute the gram product:
        g = torch.mm(features, features.t())
        # 'Normalize' the values of the gram matrix:
        g_norm = g.div(batch_size * no_of_feature_maps * x * y)
        return g_norm

    def forward(self, input: Tensor):
        g_input = self.gram_matrix(input)
        self.loss = mse_loss(g_input, self.target)
        return input


class StyleTransferModel(Module):
    def __init__(
        self,
        style_transfer_config: StyleTransferConfig,
    ) -> None:
        super(StyleTransferModel, self).__init__()

        # Initialise which layers to use for feature extration
        self.feature_layers = style_transfer_config.feature_layers
        max_layer = max(self.feature_layers)

        # Get model and clip based on selected feature layers
        self.model = style_transfer_config.model.features[: max_layer + 1]

        # Modify the ReLU layers
        # replace_layers(self.model, ReLU, ReLU(inplace=False))

        # Turn on evaluation mode:
        self.model.eval()

    def forward(self, x: Tensor) -> list[Tensor]:
        features = []

        for (layer_num, layer) in enumerate(self.model):
            x = layer(x)

            if layer_num in self.feature_layers:
                features.append(x)

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
        # Initialise image for generation
        self.generated_image = content_image.clone()
        self.generated_image.requires_grad_(True)

        # Initialise the model
        self.model = style_transfer_model

        # Initialise the optimiser
        self.optimiser = optimiser([self.generated_image])
        self.total_iterations = 0

        # Initialise the hyperparameters
        self.content_weight = content_weight
        self.style_weight = style_weight

        # Initialise style & content losses
        self.content_losses = self._get_content_losses(content_image)
        self.style_losses = self._get_style_losses(style_image)

        # Initlise transform back to PIL Image
        # (for displaying generated images)
        self.unloader = ToPILImage()

    def _get_content_losses(self, content_image):
        target_features = self.model(content_image)
        content_losses = [ContentLoss(feature.detach()) for feature in target_features]
        return content_losses

    def _get_style_losses(self, style_image):
        target_features = self.model(style_image)
        style_losses = [StyleLoss(feature.detach()) for feature in target_features]
        return style_losses

    def _optimisation_step(self, current_iteration_no: int):
        # Initialise
        self.optimiser.zero_grad()
        self.model(self.generated_image)
        style_score = 0
        content_score = 0

        # Calculate losses:
        for sl in self.style_losses:
            style_score += sl.loss
        for cl in self.content_losses:
            content_score += cl.loss

        style_score *= self.style_weight
        content_score *= self.content_weight

        loss = style_score + content_score
        loss.backward()

        # Report process every 50 steps
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

        return loss

    def optimise(self, iterations: int = 100):
        for i in range(iterations):
            self.optimiser.step(
                self._optimisation_step(i),
            )

    def get_generated_image(self) -> Image:
        image_cloned = self.generated_image.cpu().clone()

        # Remove the fake batch dimension:
        image_squeezed = image_cloned.squeeze(0)

        # Convert back to PIL Image
        image_converted = self.unloader(image_squeezed)

        return image_converted
