import matplotlib.pyplot as plt
import torch
import numpy as np
from PIL import Image

def fig2img(fig):
    """Converts a given figure handle to a 3-channel numpy image array."""
    fig.canvas.draw()
    w, h = fig.canvas.get_width_height()
    buf = np.fromstring(fig.canvas.tostring_argb(), dtype=np.uint8)
    buf.shape = (w, h, 4)
    buf = np.roll(buf, 3, axis=2)
    w, h, d = buf.shape
    return np.array(Image.frombytes("RGBA", (w, h), buf.tostring()), dtype=np.float32)[:, :, :3] / 255.

def plot_graph(array, h=400, w=400, dpi=10, linewidth=3.0):
    fig = plt.figure(figsize=(w / dpi, h / dpi), dpi=dpi)
    if isinstance(array, torch.Tensor):
        array = array.cpu().numpy()
    plt.xlim(0, array.shape[0] - 1)
    plt.xticks(fontsize=100)
    plt.yticks(fontsize=100)
    plt.plot(array)
    plt.grid()
    plt.tight_layout()
    fig_img = fig2img(fig)
    plt.close(fig)
    return fig_img



from PIL import Image, ImageDraw

def draw_text_image(text, background_color=(255,255,255), image_size=(30, 64), dtype=np.float32):

    text_image = Image.new('RGB', image_size[::-1], background_color)
    draw = ImageDraw.Draw(text_image)
    if text:
        draw.text((4, 0), text, fill=(0, 0, 0))
    if dtype == np.float32:
        return np.array(text_image).astype(np.float32)/255.
    else:
        return np.array(text_image)


def draw_text_onimage(text, image, color=(255, 0, 0)):
    if image.dtype == np.float32:
        image = (image*255.).astype(np.uint8)
    assert image.dtype == np.uint8
    from PIL import Image, ImageDraw
    text_image = Image.fromarray(image)
    draw = ImageDraw.Draw(text_image)
    draw.text((4, 0), text, fill=color)
    return np.array(text_image).astype(np.float32)/255.