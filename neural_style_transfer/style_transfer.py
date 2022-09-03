from typing import Callable, Optional, Union

import torch
from PIL import Image
from torch import Tensor
from torch.nn import Module, ReLU, Sequential
from torch.nn.functional import mse_loss
from torch.optim import LBFGS, Optimizer
from torchvision.models import *
from torchvision.transforms import ToPILImage

from .config import get_settings
from .helpers import replace_layers

# Custom type:
TorchvisionModel = Union[
    ResNet,
    AlexNet,
    VGG,
    SqueezeNet,
    DenseNet,
    Inception3,
    GoogLeNet,
    ShuffleNetV2,
    MobileNetV2,
    MobileNetV3,
    MNASNet,
    EfficientNet,
    RegNet,
]

# Initalise configuration:
settings = get_settings()


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


class FeatureExtractor(Module):
    def __init__(self, model: Module, output_layers: list[int]):
        super().__init__()

        self.model = model
        self.model.requires_grad_(False)  # Disable autograd
        self.output_layers = output_layers
        self.output_features = {}

        for layer_no, layer in enumerate(self.model.features):
            layer.register_forward_hook(self.forward_hook(layer_no))

    def forward_hook(self, layer_no: int) -> Callable:
        def hook(module: Module, input: Tensor, output: Tensor) -> None:
            if layer_no in self.output_layers:
                self.output_features[layer_no] = output

        return hook

    def forward(self, x: Tensor):
        return self.model(x)

    def get_features(self):
        print({k: v.size() for k, v in self.output_features.items()})
        return self.output_features


class StyleTransferModel(Module):
    def __init__(
        self,
        base_model: TorchvisionModel,
        style_image: Image,
        content_image: Image,
        style_feature_layers: list[int],
        content_feature_layers: list[int],
    ) -> None:
        super(StyleTransferModel, self).__init__()

        # Set layer number variables for later use
        self.style_feature_layers = style_feature_layers
        self.content_feature_layers = content_feature_layers

        # Initialise pre-trained model:
        base_model_loaded = base_model(pretrained=True)

        # Get features to initalise StyleLoss and ContentLoss modules
        self._get_features(
            base_model_loaded,
            style_image,
            style_feature_layers,
            content_image,
            content_feature_layers,
        )

        # Initialise model and add StyleLoss and ContentLoss modules to the model:
        max_layer = max(style_feature_layers + content_feature_layers)
        model_clipped = base_model_loaded.features[: max_layer + 1]  # Clip model
        self._build_model(model_clipped)

        # Modify the ReLU layers
        # replace_layers(self.model, ReLU, ReLU(inplace=False))

    def _get_features(
        self,
        base_model,
        style_image: Tensor,
        style_feature_layers: list[int],
        content_image: Tensor,
        content_feature_layers: list[int],
    ) -> None:
        style_extractor = FeatureExtractor(
            base_model,
            style_feature_layers,
        )
        style_extractor(style_image)
        style_features = style_extractor.get_features()

        content_extractor = FeatureExtractor(
            base_model,
            content_feature_layers,
        )
        content_extractor(content_image)
        content_features = content_extractor.get_features()

        self.style_features_extracted = style_features
        self.content_features_extracted = content_features

    def _build_model(self, model_clipped: Module) -> None:
        model_w_loss_layers = Sequential()

        for i, (name, layer) in enumerate(model_clipped.named_modules()):
            model_w_loss_layers.add_module(f"{name}_{i}", layer)
            if i in self.style_feature_layers:
                model_w_loss_layers.add_module(
                    f"style_loss_{i}",
                    StyleLoss(
                        self.style_features_extracted[i],
                    ),
                )
            if i in self.content_feature_layers:
                model_w_loss_layers.add_module(
                    f"content_loss_{i}",
                    ContentLoss(
                        self.content_features_extracted[i],
                    ),
                )

        self.model = model_w_loss_layers

    def forward(self, x: Tensor) -> list[Tensor]:
        style_loss = 0
        content_loss = 0

        for layer in self.model:
            x = layer(x)

            if isinstance(layer, StyleLoss):
                style_loss += x
            if isinstance(layer, ContentLoss):
                content_loss += x

        return style_loss, content_loss


class StyleTransferOptimiser:
    def __init__(
        self,
        # Model
        style_transfer_model: StyleTransferModel,
        # Starting image
        # (or else start with random noise)
        starting_image: Optional[Image.Image],
        # Optimiser for style transfer
        optimiser: Optional[Optimizer] = LBFGS,
        # Hyperparameters
        style_weight: Optional[int] = 1000000,
        content_weight: Optional[int] = 1,
    ) -> None:
        # Initialise image for generation
        if starting_image is None:
            torch.randn(
                settings.OUTPUT_IMG_SIZE,
                device=settings.DEVICE,
            )
        else:
            self.generated_image = starting_image.clone()
            self.generated_image.requires_grad_(True)

        # Initialise the model
        self.style_transfer_model = style_transfer_model

        # Initialise the optimiser
        self.optimiser = optimiser([self.generated_image])
        self.total_iterations = 0

        # Initialise the hyperparameters
        self.content_weight = content_weight
        self.style_weight = style_weight

        # Initlise transform back to PIL Image
        # (for displaying generated images)
        self.unloader = ToPILImage()

    def _optimisation_step(self, current_iteration_no: int):
        # Initialise
        self.optimiser.zero_grad()
        style_score = 0
        content_score = 0

        # Calculate losses:
        style_score, content_score = self.style_transfer_model(
            self.generated_image,
        )

        # Adjust scores based on hyperparams.
        style_score *= self.style_weight
        content_score *= self.content_weight

        # Calculate total loss
        loss = style_score + content_score
        loss.backward()

        # Report process every 50 steps
        if (current_iteration_no + 1) % 50 == 0:
            print(f"Run #{current_iteration_no + 1}")
            print(f"Style loss: {style_score:4f}, Content loss: {content_score:4f}")

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
