import torch
import torch.nn as nn


def tile_action_into_image(action, image_size):
    """
    :param action: [B, a_dim] tensor
    :param image_size: tuple containing (w, h)
    :return: A tensor of shape [B, 1, w, h], which contains the actions tiled into the image shape
    """
    a_dim = action.shape[1]
    action_tiled = action.unsqueeze(2).unsqueeze(2)  #[B, a_dim, 1, 1]
    # Compute how many times to repeat in each dim
    w, h = image_size
    action_tiled = action_tiled.repeat(1, 1, w, h)
    return action_tiled


def repeat_weights(weights, new_channels):
    prev_channels = weights.shape[1]
    assert prev_channels == 3, "Original weights should have three input channels"
    new_shape = list(weights.shape[:])
    new_shape[1] = new_channels
    new_weights = torch.zeros(new_shape, dtype=weights.dtype, layout=weights.layout, device=weights.device)
    for i in range(0, new_channels):
        new_weights[:, i] = weights[:, i % prev_channels]
    return new_weights


def get_resnet_encoder(resnet_type, num_features_out, channels_in=3, freeze=False):
    model = resnet_type(pretrained=True, progress=True)
    for param in model.parameters():
        param.requires_grad = not freeze
    model.fc = nn.Linear(in_features=model.fc.in_features, out_features=num_features_out)

    if channels_in != 3:
        orig_weights = model.conv1.weight.clone()
        new_weights = repeat_weights(orig_weights, channels_in)
        new_layer = nn.Conv2d(channels_in, orig_weights.shape[0],  kernel_size=(7, 7), stride=(2, 2), padding=(3, 3), bias=False)
        new_layer.weight = nn.Parameter(new_weights)
        model.conv1 = new_layer

    return model
