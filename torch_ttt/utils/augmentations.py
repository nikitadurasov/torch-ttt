import random
from torchvision.transforms import functional as F


class RandomResizedCrop:
    def __init__(self, scale=(0.2, 1.0)):
        self.scale = scale

    def __call__(self, img):
        # Dynamically compute the crop size
        original_size = img.shape[-2:]  # H × W
        area = original_size[0] * original_size[1]
        target_area = random.uniform(self.scale[0], self.scale[1]) * area
        aspect_ratio = random.uniform(3 / 4, 4 / 3)

        h = int(round((target_area * aspect_ratio) ** 0.5))
        w = int(round((target_area / aspect_ratio) ** 0.5))

        if random.random() < 0.5:  # Randomly swap h and w
            h, w = w, h

        h = min(h, original_size[0])
        w = min(w, original_size[1])

        return F.resized_crop(img, top=0, left=0, height=h, width=w, size=original_size)
