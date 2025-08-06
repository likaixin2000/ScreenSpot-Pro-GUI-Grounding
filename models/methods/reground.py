import sys
sys.path.append("../models/")

import os
from PIL import Image
import base64
from io import BytesIO

from PIL import Image


def convert_pil_image_to_base64(image):
    buffered = BytesIO()
    image.save(buffered, format="PNG")
    return base64.b64encode(buffered.getvalue()).decode()


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


def crop_to_center(img, bbox, crop_size):
    """
    Crop an image around the center of a bounding box.
    
    Args:
        img (PIL.Image.Image): The image to crop.
        bbox (tuple): Bounding box (x_min, y_min, x_max, y_max) in relative coordinates.
        crop_size (tuple): The size of the crop (width, height).
        
    Returns:
        tuple: A tuple containing the cropped bounding box in relative coordinates and the cropped image.
    """
    img_width, img_height = img.size
    crop_width, crop_height = crop_size
    
    x1, y1, x2, y2 = bbox[0] * img_width, bbox[1] * img_height, bbox[2] * img_width, bbox[3] * img_height
    
    # Find the center of the bbox
    cx, cy = (x1 + x2) / 2, (y1 + y2) / 2
    
    half_crop_width = crop_width / 2
    half_crop_height = crop_height / 2
    
    # Define the new left and upper boundaries, ensuring they don't go outside the image
    new_x1_pixel = max(0, cx - half_crop_width)
    new_y1_pixel = max(0, cy - half_crop_height)
    
    # Define the new right and lower boundaries, ensuring they don't exceed the image size
    new_x2_pixel = min(img_width, cx + half_crop_width)
    new_y2_pixel = min(img_height, cy + half_crop_height)
    
    # Adjust boundaries if they meet the image boundaries
    if new_x1_pixel == 0:
        new_x2_pixel = min(img_width, new_x1_pixel + crop_width)  # Extend to the right
    if new_y1_pixel == 0:
        new_y2_pixel = min(img_height, new_y1_pixel + crop_height)  # Extend downward
    
    if new_x2_pixel == img_width:
        new_x1_pixel = max(0, new_x2_pixel - crop_width)  # Extend to the left
    if new_y2_pixel == img_height:
        new_y1_pixel = max(0, new_y2_pixel - crop_height)  # Extend upward
    
    # Crop the image
    cropped_img = img.crop((new_x1_pixel, new_y1_pixel, new_x2_pixel, new_y2_pixel))
    
    # Normalize the cropped box back to relative coordinates
    cropped_box = (new_x1_pixel / img_width, new_y1_pixel / img_height, new_x2_pixel / img_width, new_y2_pixel / img_height)
    
    return cropped_box, cropped_img


class ReGroundMethod:
    def __init__(self, planner="gpt-4o-2024-05-13", grounder=None, configs=None):
        self.planner_model_name = planner
        # self.grounder = grounder
        self.grounder = grounder
        
        self.logs = []

    def debug_print(self, string):
        self.logs.append(string)

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
                "point": [(bbox[0] + bbox[2]) / 2, (bbox[1] + bbox[3]) / 2] if bbox else None,
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
        # self.debug_print(f"Grounded bbox (viewport): {view_bbox}")
        
        if result['bbox']:
            view_bbox = result['bbox']  # the bbox in the current viewport
        elif result['point']:
            point = result['point']
            view_bbox = [point[0], point[1], point[0], point[1]]  # Fake bounding box
        else:
            self.debug_print(f"Grounding failed. Raw response:\n{result}")
            return None, None  # should not happen
        real_bbox = view_bbox_to_real_bbox(view_bbox=view_bbox, viewport=viewport)
        # self.debug_print(f"Grounded bbox (real): {real_bbox}")
        return view_bbox, real_bbox
    
    # def rephrase_target(self, instruction, image):


    def visual_search(self, instruction, image):
        view_bbox, real_bbox = self.ground_with_grounder(instruction, image, viewport=(0, 0, 1, 1))
        if not real_bbox:
            return False, None
    
        cropped_box, focused_image = crop_to_center(image, real_bbox, crop_size=[1024, 1024])
        view_bbox, real_bbox = self.ground_with_grounder(instruction, focused_image, viewport=cropped_box)
        return True, real_bbox
