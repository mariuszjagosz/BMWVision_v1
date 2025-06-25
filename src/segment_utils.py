import torch
import cv2
import numpy as np
from pathlib import Path
from typing import Tuple, Optional, Any, Union

from models.twinlite.model.model import TwinLiteNetPlus
from models.twinlite.demoDataset import letterbox_for_img as letterbox

# ==========================
# Lokalna klasa konfiguracji
# ==========================

sc_ch_dict = {
    "nano":   {'p': 1, 'q': 1, 'chanels': [4, 8, 16, 32, 64]},
    "small":  {'p': 2, 'q': 3, 'chanels': [8, 16, 32, 64, 128]},
    "medium": {'p': 3, 'q': 5, 'chanels': [16, 32, 64, 128, 256]},
    "large":  {'p': 5, 'q': 7, 'chanels': [32, 64, 128, 256, 512]},
}

class ConfigArgs:
    def __init__(self, config_name: str):
        if config_name not in sc_ch_dict:
            raise ValueError(f"Unknown config name: {config_name}")
        self.config = config_name
        config = sc_ch_dict[config_name]
        self.p = config["p"]
        self.q = config["q"]
        self.chanels = config["chanels"]
        self.chanel_img = 3

# ==========================
# Segmenter
# ==========================

NORMALIZATION_FACTOR = 255.0

class Segmenter:
    def __init__(self,
                 weight_path: Union[str, Path],
                 config_name: str = "large",
                 device: Optional[str] = None,
                 img_size: int = 640):
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.img_size = img_size
        self.half = self.device != "cpu"

        self.args = ConfigArgs(config_name)
        self.model = TwinLiteNetPlus(self.args).to(self.device)

        if self.half:
            self.model.half()

        weight_path = Path(weight_path)
        if not weight_path.exists():
            raise FileNotFoundError(f"Weight file not found: {weight_path}")
        self.model.load_state_dict(torch.load(str(weight_path), map_location=self.device))
        self.model.eval()

        print(f"Segmenter initialized on device: {self.device} with img_size: {self.img_size}")
        if self.half:
            print("Half-precision (FP16) enabled.")

    def preprocess(self, frame: np.ndarray) -> Tuple[torch.Tensor, Tuple[int, ...], int, int]:
        original_shape = frame.shape
        img_resized, _, (pad_w, pad_h) = letterbox(frame, new_shape=self.img_size)

        pad_w, pad_h = int(pad_w), int(pad_h)

        img_rgb_chw = img_resized[:, :, ::-1].transpose(2, 0, 1)
        img_contiguous = np.ascontiguousarray(img_rgb_chw)
        img_tensor = torch.from_numpy(img_contiguous).to(self.device)

        if self.half:
            img_tensor = img_tensor.half() / NORMALIZATION_FACTOR
        else:
            img_tensor = img_tensor.float() / NORMALIZATION_FACTOR

        img_tensor = img_tensor.unsqueeze(0)
        return img_tensor, original_shape, pad_w, pad_h

    def predict(self, frame: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        img_processed, orig_shape, pad_w, pad_h = self.preprocess(frame)
        _, _, pre_height, pre_width = img_processed.shape

        with torch.no_grad():
            da_seg_out, ll_seg_out = self.model(img_processed)

        da_predict = da_seg_out[:, :, pad_h:(pre_height - pad_h), pad_w:(pre_width - pad_w)]
        da_seg_mask_resized = torch.nn.functional.interpolate(
            da_predict, size=orig_shape[:2], mode='bilinear', align_corners=False
        )
        _, da_seg_mask_classes = torch.max(da_seg_mask_resized, 1)
        da_mask = da_seg_mask_classes.squeeze(0).cpu().numpy().astype(np.uint8)

        ll_predict = ll_seg_out[:, :, pad_h:(pre_height - pad_h), pad_w:(pre_width - pad_w)]
        ll_seg_mask_resized = torch.nn.functional.interpolate(
            ll_predict, size=orig_shape[:2], mode='bilinear', align_corners=False
        )
        _, ll_seg_mask_classes = torch.max(ll_seg_mask_resized, 1)
        ll_mask = ll_seg_mask_classes.squeeze(0).cpu().numpy().astype(np.uint8)

        return da_mask, ll_mask

# ==========================
# Test lokalny
# ==========================

if __name__ == "__main__":
    import matplotlib.pyplot as plt

    WEIGHTS_FILE = Path("models/twinlite/pretrained/large.pth")
    IMAGE_FILE = Path("frame.jpg")

    if not WEIGHTS_FILE.exists():
        print(f"Error: Weights file not found at {WEIGHTS_FILE}")
        exit()
    if not IMAGE_FILE.exists():
        print(f"Error: Image file not found at {IMAGE_FILE}")
        exit()

    try:
        segmenter = Segmenter(
            weight_path=WEIGHTS_FILE,
            config_name="large",
            img_size=640
        )

        original_image = cv2.imread(str(IMAGE_FILE))
        if original_image is None:
            print(f"Error: Could not read image from {IMAGE_FILE}")
            exit()

        da_mask, ll_mask = segmenter.predict(original_image)

        overlay = np.zeros_like(original_image)
        overlay[da_mask == 1] = [128, 128, 128]  # Drivable area
        overlay[ll_mask == 1] = [255, 128, 0]    # Lane lines
        blended = cv2.addWeighted(original_image, 0.7, overlay, 0.3, 0)

        plt.figure(figsize=(18, 6))
        plt.subplot(1, 3, 1)
        plt.title("Original Image")
        plt.imshow(cv2.cvtColor(original_image, cv2.COLOR_BGR2RGB))
        plt.axis('off')

        plt.subplot(1, 3, 2)
        plt.title("Overlay")
        plt.imshow(cv2.cvtColor(blended, cv2.COLOR_BGR2RGB))
        plt.axis('off')

        plt.subplot(1, 3, 3)
        plt.title("Lane Lines Mask")
        plt.imshow(ll_mask, cmap="hot")
        plt.axis('off')

        plt.tight_layout()
        output_dir = Path("output")
        output_dir.mkdir(exist_ok=True)
        cv2.imwrite(str(output_dir / "output_image.jpg"), original_image)
        cv2.imwrite(str(output_dir / "output_da_mask.png"), da_mask * 255)
        cv2.imwrite(str(output_dir / "output_ll_mask.png"), ll_mask * 255)
        cv2.imwrite(str(output_dir / "output_overlay.jpg"), blended)
        plt.savefig(output_dir / "output_visualization.png")
        plt.show()

        print("Saved outputs to 'output/' directory.")

    except FileNotFoundError as e:
        print(f"Error: {e}")
    except Exception as e:
        print(f"An unexpected error occurred: {e}")
