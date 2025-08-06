import sys
sys.path.append("../models/")

import os
from PIL import Image
import base64
from io import BytesIO
from typing import Tuple, Annotated

from PIL import Image


def convert_pil_image_to_base64(image):
    buffered = BytesIO()
    image.save(buffered, format="PNG")
    return base64.b64encode(buffered.getvalue()).decode()


def resize(image, base_width=None, base_height=None):
    # Original dimensions
    original_width, original_height = image.size

    # Calculate new dimensions
    if base_width:
        if base_width <= original_width:
            return image
        w_percent = (base_width / float(original_width))
        new_height = int((float(original_height) * float(w_percent)))
        new_size = (base_width, new_height)
    elif base_height:
        if base_height <= original_height:
            return image
        h_percent = (base_height / float(original_height))
        new_width = int((float(original_width) * float(h_percent)))
        new_size = (new_width, base_height)
    else:
        raise ValueError("Either base_width or base_height must be specified")

    # Resize the image
    resized_img = image.resize(new_size, Image.LANCZOS)
    return resized_img


def view_bbox_to_real_bbox(view_bbox, viewport):
    """
    Converts a bounding box defined in a viewport's normalized coordinates (view_bbox) to real normalized coordinates (real_bbox) in the full image.

    Parameters:
        view_bbox (tuple): A tuple (x1, y1, x2, y2) representing the bounding box in normalized coordinates
            relative to the viewport. Each value should be between 0 and 1.
        viewport (tuple): A tuple (x1, y1, x2, y2) representing the viewport's real coordinates in the full image.

    Returns:
        list: A list [real_x1, real_y1, real_x2, real_y2] representing the bounding box in real coordinates
            within the full image.
    """
    view_x1, view_y1, view_x2, view_y2 = view_bbox
    x1, y1, x2, y2 = viewport
    w = x2 - x1
    h = y2 - y1
    real_bbox = [x1 + w * view_x1, y1 + h * view_y1, x1 + w * view_x2, y1 + h * view_y2]
    return real_bbox


def crop_with_padding(
    image: Image.Image, 
    bbox: Tuple[Annotated[float, "Normalized"], Annotated[float, "Normalized"], 
                Annotated[float, "Normalized"], Annotated[float, "Normalized"]], 
    padding: Annotated[float, "Normalized padding"]
) -> Tuple[Tuple[float, float, float, float], Image.Image]:
    """
    Crops an image with specified normalized coordinates and applies padding directly in the normalized scale.

    Parameters:
        image (PIL.Image.Image): The image to crop.
        bbox (Tuple[float, float, float, float]): A tuple (x1, y1, x2, y2) specifying 
            the crop area (normalized between 0 and 1).
        padding (float): Padding in the normalized scale (e.g., 0.1 adds 10% of the image size).

    Returns:
        Tuple[Tuple[float, float, float, float], PIL.Image.Image]: A tuple containing:
            - Tuple[float, float, float, float]: The normalized coordinates of the cropped area with padding applied.
            - PIL.Image.Image: The cropped image.
    """
    # Unpack normalized bounding box
    x1_norm, y1_norm, x2_norm, y2_norm = bbox

    # Apply padding directly in the normalized scale
    x1_norm_padded = max(0.0, x1_norm - padding)
    y1_norm_padded = max(0.0, y1_norm - padding)
    x2_norm_padded = min(1.0, x2_norm + padding)
    y2_norm_padded = min(1.0, y2_norm + padding)

    # Scale normalized coordinates to pixel values
    img_width, img_height = image.size
    x1 = int(x1_norm_padded * img_width)
    y1 = int(y1_norm_padded * img_height)
    x2 = int(x2_norm_padded * img_width)
    y2 = int(y2_norm_padded * img_height)

    # Crop the image
    cropped_box = (x1, y1, x2, y2)
    cropped_image = image.crop(cropped_box)

    # Return normalized box and cropped image
    normalized_cropped_box = (x1_norm_padded, y1_norm_padded, x2_norm_padded, y2_norm_padded)
    return normalized_cropped_box, cropped_image


class IterativeNarrowingMethod:
    def __init__(self, planner=None, grounder=None, configs=None):
        self.planner_model_name = planner
        # self.grounder = grounder
        self.grounder = grounder
        
        self.configs = configs if configs else {
            "max_search_depth": 3,
            "min_crop_size": 768,
        }
        
        self.logs = []
        self.debug_flag = True

    def set_generation_config(self, **kwargs):
        self.override_generation_config.update(kwargs)

    def debug_print(self, string):
        self.logs.append(string)
        if self.debug_flag:
            print(string)

    def ground_only_positive(self, instruction, image):
        """Grounding entry point."""
        self.logs = []
        if isinstance(image, str):
            image_path = image
            assert os.path.exists(image_path) and os.path.isfile(image_path), "Invalid input image path."
            image = Image.open(image_path).convert('RGB')
        assert isinstance(image, Image.Image), "Invalid input image."
        flag, bbox = self.visual_search(instruction, image)

        if flag:
            result_dict = {
                "result": "positive",
                "bbox": bbox,
                "point": [(bbox[0] + bbox[2]) / 2, (bbox[1] + bbox[3]) / 2],
                "raw_response": self.logs
            }
        else:
            result_dict = {
                "result": "negative",
                "bbox": None,
                "point": None,
                "raw_response": self.logs
            }
        return result_dict
    
    def ground_with_grounder(self, instruction, image, viewport):
        result = self.grounder.ground_only_positive(instruction, image)
        view_bbox = result['bbox']  # the bbox in the current viewport
        # self.debug_print(f"Grounded bbox (viewport): {view_bbox}")
        if not view_bbox:
            self.debug_print(f"Grounding failed. Raw response:\n{result}")
            return None, None  # should not happen
        real_bbox = view_bbox_to_real_bbox(view_bbox=view_bbox, viewport=viewport)
        # self.debug_print(f"Grounded bbox (real): {real_bbox}")
        return view_bbox, real_bbox
    
    def visual_search(self, instruction, image):
        img_width, img_height = image.size
        # Define initial viewport covering the entire image
        viewport = (0, 0, 1, 1)

        for iteration in range(self.configs["max_search_depth"]):
            # Perform the search
            view_bbox, real_bbox = self.ground_with_grounder(
                instruction, image, viewport=viewport
            )

            if not real_bbox:
                return False, None

            if iteration < self.configs["max_search_depth"] - 1:  # Split for the first two iterations
                # Define normalized areas (0-1 coordinates) based on current view_bbox
                x1, y1, x2, y2 = view_bbox
                view_center_x = (x1 + x2) / 2
                view_center_y = (y1 + y2) / 2

                ax1 = max(0, view_center_x - 0.25)
                ay1 = max(0, view_center_y - 0.25)
                ax2 = min(1, view_center_x + 0.25)
                ay2 = min(1, view_center_y + 0.25)

                target_area = (ax1, ay1, ax2, ay2)

                if target_area is None:
                    return False, None

                # Update viewport to the identified target area
                _, image = crop_with_padding(image, target_area, padding=0)
                viewport = view_bbox_to_real_bbox(target_area, viewport=viewport)

        return True, real_bbox
