from PIL import Image


class ColorService:
    @staticmethod
    def dominant_color_hex(image: Image.Image) -> str:
        red, green, blue = ColorService.dominant_color_rgb(image)
        return f"#{red:02x}{green:02x}{blue:02x}"

    @staticmethod
    def dominant_color_rgb(image: Image.Image) -> tuple[int, int, int]:
        import numpy as np

        rgb = image.convert("RGB")
        width, height = rgb.size
        left = round(width * 0.12)
        top = round(height * 0.12)
        right = round(width * 0.88)
        bottom = round(height * 0.88)
        center = rgb.crop((left, top, max(left + 1, right), max(top + 1, bottom))).resize((96, 96))
        full = rgb.resize((96, 96))

        full_array = np.asarray(full, dtype=np.int16)
        center_array = np.asarray(center, dtype=np.int16).reshape(-1, 3)
        corners = np.concatenate(
            [
                full_array[:12, :12].reshape(-1, 3),
                full_array[:12, -12:].reshape(-1, 3),
                full_array[-12:, :12].reshape(-1, 3),
                full_array[-12:, -12:].reshape(-1, 3),
            ],
            axis=0,
        )
        approximate_background = np.median(corners, axis=0)
        distance_from_background = np.linalg.norm(center_array - approximate_background, axis=1)
        foreground_pixels = center_array[distance_from_background > 36]

        # This is an approximate MVP foreground color. If the background and product are similar,
        # the fallback uses the center crop rather than pretending true segmentation happened.
        pixels = foreground_pixels if len(foreground_pixels) >= 100 else center_array
        quantized = (pixels // 24) * 24
        colors, counts = np.unique(quantized, axis=0, return_counts=True)
        dominant = colors[counts.argmax()]
        return tuple(int(max(0, min(255, value))) for value in dominant)

    @staticmethod
    def color_similarity(rgb_a: tuple[int, int, int], rgb_b: tuple[int, int, int]) -> float:
        import math

        distance = math.sqrt(sum((channel_a - channel_b) ** 2 for channel_a, channel_b in zip(rgb_a, rgb_b)))
        max_distance = math.sqrt(3 * (255**2))
        return max(0.0, min(1.0, 1.0 - distance / max_distance))
