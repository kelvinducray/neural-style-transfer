from torch.nn import Module, Sequential


def replace_layers(
    model: Sequential,
    old: Module,
    new: Module,
) -> None:
    """
    Given a model, a specified layer type as well as a new type,
    this function replaces all instances of that layer type with
    the newly specified type.

    Note: This transformation is completed in-place.
    """

    for n, module in model.named_children():
        # If there is a compound module, go inside it:
        if len(list(module.children())) > 0:
            replace_layers(module, old, new)

        # Replace the instance:
        if isinstance(module, old):
            setattr(model, n, new)
