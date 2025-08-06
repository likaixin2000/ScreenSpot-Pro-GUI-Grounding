import sys
sys.path.append("../models/")

import os
# API_KEY = os.environ['OPENAI_API_KEY']
import re
import io
import math
import json
from collections import defaultdict
import itertools
import base64
# import svgwrite
# import cairosvg
import numpy as np
from tqdm import tqdm
from io import BytesIO
from PIL import Image, ImageDraw, ImageFont, ImageColor
from typing import Tuple, Annotated

import openai


POSITION_INFERENCE_PROMPT_TOP_LEVEL = (
"""I want to identify a UI element that best matches my instruction. Please help me determine which region(s) of the screenshot to focus on and list the UI elements that might appear next to the target.
If the target does not exist in the screenshot, please output "No target".

**Output Requirements:**
1. List the possible regions in descending order of probability. 
2. **Always make specific, clear and unique references to avoid ambiguity**. References such as "Other icons" and "window" are NOT allowed, because they don't refer to a specific UI element.
3. Use the following XML tags to describe items in the screenshot:
   - `<element>`: Wrap a specific UI element.
   - `<area>`: Describe an area of the UI containing multiple elements.
   - `<neighbor>`: Describe a UI element that may appear around the target, to help anchor its location.

**Example Output (No need to follow the strict sentence format):**
1. The <element>shortcut link</element> is most likely to be found in the <area>Settings window</area>, in the <area>tools panel in settings window</area> next to the <neighbor>Search button in the settings window</neighbor>. There is also a <neighbor>update button in the settings window</neighbor> nearby.
2. The target may also appear in the <area>Web Browser</area>, in the <area>Bookmark bar in browser</area>, next to the <neighbor>Search button in the bookmark bar in web browser</neighbor> and the <neighbor>open bookmark collection button in the browser</neighbor>.
...

**Important Notes:**
- The target UI element is guaranteed to be present in the screenshot. Do **not** speculate about any operations that could change the screenshot, such as navigating to another page or opening a menu.

**Instruction:**
{instruction}
"""
)


EXISTENCE_PROMPT = """
You are given a cropped screenshot. Your task is to evaluate whether the marked element in red box matches the target described in my instruction.

Please follow these steps:
1. Analyze the screenshot by describing its visible content and functionalities.
2. Determine which of the following applies:
    - 'is_target': The marked element is the target.
    - 'target_elsewhere': The marked element is not the target, but the target exists elsewhere in the screenshot.
    - 'target_not_found': The marked element is not the target, and the target does not exist in the screenshot.
3. If the target exists, rewrite the instruction to make it more specific and clear. This should include unambiguous details like labels, text, or position.

After your analysis, provide the result in JSON format with the following fields:
- "result": (str) One of 'is_target', 'target_elsewhere', or 'target_not_found'.
- "new_instruction": (str, default null) A clearer, more specific version of the instruction, if applicable.

Here is my instruction:
{instruction}
"""


def convert_pil_image_to_base64(image):
    buffered = BytesIO()
    image.save(buffered, format="PNG")
    return base64.b64encode(buffered.getvalue()).decode()


# def get_text_image(image, text, text_height, text_color="white", bg_color="blue"):
#     # Get the image dimensions (needed for compatibility, even if not used directly)
#     image_width, image_height = image.size

#     # Estimate the width of the text based on its length and desired height
#     estimated_width = len(text) * (text_height // 2) * 2  # Approximation for text width

#     # Create an SVG drawing
#     dwg = svgwrite.Drawing(size=(estimated_width, text_height))

#     # Add a rectangle for the background
#     dwg.add(dwg.rect(
#         insert=(0, 0),
#         size=(estimated_width, text_height),
#         fill=bg_color
#     ))

#     # Add the text to the SVG
#     dwg.add(dwg.text(
#         text,
#         insert=(0.2 * text_height, text_height * 0.9),  # Position (x, y) with baseline adjustment
#         font_size=f"{text_height * 0.9}px",
#         font_family="SimSun",
#         fill=text_color
#     ))

#     # Save the SVG as a string
#     svg_data = dwg.tostring()

#     # Convert the SVG string to a PNG using CairoSVG
#     png_data = cairosvg.svg2png(bytestring=svg_data)

#     # Load the PNG data into a Pillow image object
#     png_image = Image.open(io.BytesIO(png_data))

#     # Return the size of the text and the bitmap image
#     return (png_image.width, png_image.height), png_image

# def draw_alpha_box(image, rect, color, alpha, line_width):
#     x1, y1, x2, y2 = rect
#     rect_color_rgb = ImageColor.getrgb(color)
#     draw = ImageDraw.Draw(image)
#     draw.rectangle([x1, y1, x2, y2], outline=color, width=line_width)

#     for x in range(x1, x2):
#         for y in range(y1, y2):
#             background_color = image.getpixel((x, y))
#             blended_color = tuple(
#                 int((alpha * rect_color_rgb[i] + (1 - alpha) * background_color[i])) for i in range(3)
#             )
#             draw.point((x, y), fill=blended_color)

#     return image



# def draw_bounding_boxes(image, bboxes, alpha=0.2):
#     image = image.copy().convert("RGBA")
#     draw = ImageDraw.Draw(image)

#     width, height = image.size

#     box_cache = []

#     for i, (name, coords) in enumerate(reversed(list(bboxes.items()))):  # Draw from the last one to the first, to make sure high-priority boxes are complete
#         name = str(name)
#         x1, y1, x2, y2 = coords
#         color = BOX_COLORS[(len(BOX_COLORS) - i - 1) % len(BOX_COLORS)]
#         # color = "red"
        
#         # Convert normalized coordinates to pixel values, and draw the line outside the target (do not pad inside)
#         line_width = 5
#         x1 = max(0, int(x1 * width - line_width))
#         y1 = max(0, int(y1 * height - line_width))
#         x2 = min(width, int(x2 * width + line_width))
#         y2 = min(height, int(y2 * height + line_width))

#         # draw.rectangle([x1, y1, x2, y2], outline=color, width=line_width, fill=rect_color_rgba)
#         image = draw_alpha_box(image, (x1, y1, x2, y2), color, alpha, line_width)
#         font_pixel_height = max(12, int(0.03 * height))
#         (label_width, label_height), text_image = get_text_image(image, name, font_pixel_height, bg_color=color)

#         # Position the label at the top-left corner of the bounding box
#         label_x1 = x1
#         label_y1 = y1 - label_height  # Default position: above the bounding box

#         # Sanity check: Adjust the label position if it goes beyond the image
#         if label_y1 < 0:  # If label goes above the image
#             label_y1 = y1  # Move it below the bounding box
#         if label_x1 + label_width > width:  # If label exceeds the image width
#             label_x1 = width - label_width  # Shift it left to fit
#         if label_x1 < 0:  # If label exceeds the left boundary
#             label_x1 = 0  # Align it to the left edge
        
#         # Put the text image into cache
#         box_cache.append((label_x1, label_y1, text_image))

#     # Loop over the cache and paste the images, to make sure texts are above boxes.
#     # Assume boxes with smaller orders have priority
#     pasted_boxes = []
#     for x, y, text_image in box_cache:
#         tx1, ty1, tx2, ty2 = x, y, x + text_image.width, y + text_image.height

#         # Check if the position is occupied (overlap check)
#         is_occupied = False
#         for (bx1, by1, bx2, by2) in pasted_boxes:
#             # Check for overlap (if any part of the new image intersects with a pasted box)
#             if not (tx2 <= bx1 or tx1 >= bx2 or ty2 <= by1 or ty1 >= by2):
#                 is_occupied = True
#                 break

#         # If the position is occupied, move horizontally to find a place
#         while is_occupied:
#             x += text_image.width  # Move horizontally by the width of the text image
#             tx1, ty1, tx2, ty2 = x, y, x + text_image.width, y + text_image.height
#             is_occupied = False
#             for (bx1, by1, bx2, by2) in pasted_boxes:
#                 if not (tx2 <= bx1 or tx1 >= bx2 or ty2 <= by1 or ty1 >= by2):
#                     is_occupied = True
#                     break

#         # Paste the text image at the position
#         image.paste(text_image, (x, y))
#         pasted_boxes.append((tx1, ty1, tx2, ty2))

#     return image.convert("RGB")


def plot_annotated_image(image: Image, point=None, bbox=None, label: str = None):
    if point is None and bbox is None:
        raise ValueError("At least one of 'point' or 'bbox' must be provided.")
    
    img_width, img_height = image.size
    annotated_image = image.copy()
    draw = ImageDraw.Draw(annotated_image)
    
    if point is not None:
        # Scale the point
        x, y = point
        x_scaled = int(x * img_width)
        y_scaled = int(y * img_height)
        
        # Draw point
        radius = 10
        draw.ellipse((x_scaled - radius, y_scaled - radius, x_scaled + radius, y_scaled + radius), fill='red')
        
        if label:
            draw.text((x_scaled + radius + 5, y_scaled - radius), label, fill='white')
    
    if bbox is not None:
        # Scale the bounding box
        x_min, y_min, x_max, y_max = bbox
        x_min_scaled = int(x_min * img_width)
        y_min_scaled = int(y_min * img_height)
        x_max_scaled = int(x_max * img_width)
        y_max_scaled = int(y_max * img_height)
        
        # Draw bounding box with a transparent mask
        draw.rectangle([(x_min_scaled, y_min_scaled), (x_max_scaled, y_max_scaled)], outline='red', width=5, )
        
        if label:
            draw.text((x_min_scaled, y_min_scaled - 10), label, fill='white')
    
    return annotated_image


def plot_debug_image(image: Image, bboxes=None, labels = None):
    img_width, img_height = image.size
    annotated_image = image.copy()
    draw = ImageDraw.Draw(annotated_image)
    
    if bboxes is not None:
        for i, bbox in enumerate(bboxes):
            
            # Scale the bounding box
            x_min, y_min, x_max, y_max = bbox
            x_min_scaled = int(x_min * img_width)
            y_min_scaled = int(y_min * img_height)
            x_max_scaled = int(x_max * img_width)
            y_max_scaled = int(y_max * img_height)
            
            # Draw bounding box with a transparent mask
            draw.rectangle([(x_min_scaled, y_min_scaled), (x_max_scaled, y_max_scaled)], outline='red', width=5, )
            
            if labels:
                label = str(labels[i])
                draw.text((x_min_scaled, y_min_scaled - 10), label, fill='white')
    
    return annotated_image


def non_maximum_suppression(boxes, iou_threshold=0.5, keep="order"):
    """
    Perform Non-Maximum Suppression to remove overlapping bounding boxes.

    Args:
        boxes (list of lists or np.ndarray): Bounding boxes in the format [x1, y1, x2, y2].
        iou_threshold (float): IoU threshold to determine overlap (default: 0.5).
        keep (str): Determines which box to keep when overlap occurs:
                    - "largest": Keep the largest box by area.
                    - "order": Keep boxes based on their original order.

    Returns:
        list: Indices of the boxes to keep.
    """
    if len(boxes) == 0:
        return []

    # Convert to numpy array for convenience
    boxes = np.array(boxes)

    # Compute the area of each box
    areas = (boxes[:, 2] - boxes[:, 0]) * (boxes[:, 3] - boxes[:, 1])

    # Determine processing order
    if keep == "largest":
        # Sort boxes by area in descending order
        order = areas.argsort()[::-1]
    elif keep == "order":
        # Keep the original order
        order = np.arange(len(boxes))
    else:
        raise ValueError("Invalid value for 'keep'. Choose 'largest' or 'order'.")

    keep_indices = []
    while len(order) > 0:
        # Always keep the first box in the current order
        current = order[0]
        keep_indices.append(current)

        # Compute IoU with the current box and the rest
        x1 = np.maximum(boxes[current, 0], boxes[order[1:], 0])
        y1 = np.maximum(boxes[current, 1], boxes[order[1:], 1])
        x2 = np.minimum(boxes[current, 2], boxes[order[1:], 2])
        y2 = np.minimum(boxes[current, 3], boxes[order[1:], 3])

        # Compute intersection area
        inter_area = np.maximum(0, x2 - x1) * np.maximum(0, y2 - y1)

        # Compute IoU
        iou = inter_area / (areas[current] + areas[order[1:]] - inter_area)

        # Keep boxes with IoU less than the threshold
        remaining = np.where(iou < iou_threshold)[0]
        order = order[remaining + 1]
    # return the boxes
    return [boxes[i] for i in keep_indices]


def resize(image, target_width=None, target_height=None):
    # Original dimensions
    original_width, original_height = image.size

    # Calculate new dimensions
    if target_width:
        if target_width >= original_width:
            return image
        w_percent = (target_width / float(original_width))
        new_height = int((float(original_height) * float(w_percent)))
        new_size = (target_width, new_height)
    elif target_height:
        if target_height >= original_height:
            return image
        h_percent = (target_height / float(original_height))
        new_width = int((float(original_width) * float(h_percent)))
        new_size = (new_width, target_height)
    else:
        raise ValueError("Either target_width or target_height must be specified")

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


def get_sys_prompt_msg(sys_prompt):
    history = [
        {
            "role": "system",
            "content": [
                {"type": "text", "text": sys_prompt}
            ],
        }
    ]
    return history


def dilate_box(image_size: list, bbox: list, dilation_size: list):
    dilation_size_x, dilation_size_y = dilation_size
    img_width, img_height = image_size
    x1, y1, x2, y2 = bbox
    cx, cy = (x1 + x2) / 2, (y1 + y2) / 2
    new_x1_pixel = max(0, cx * img_width - dilation_size_x / 2)
    new_y1_pixel = max(0, cy * img_height - dilation_size_y / 2)
    new_x2_pixel = min(img_width, cx * img_width + dilation_size_x / 2)
    new_y2_pixel = min(img_height, cy * img_height + dilation_size_y / 2)

    # Ensure the result has the size of dilation_size
    # If a side of the box is at the image boundary, extend it in the opposite direction
    if new_x1_pixel == 0:
        new_x2_pixel = min(img_width, new_x2_pixel + dilation_size_x / 2)
    if new_y1_pixel == 0:
        new_y2_pixel = min(img_height, new_y2_pixel + dilation_size_y / 2)
    if new_x2_pixel == img_width:
        new_x1_pixel = max(0, new_x1_pixel - dilation_size_x / 2)
    if new_y2_pixel == img_height:
        new_y1_pixel = max(0, new_y1_pixel - dilation_size_y / 2)

    new_x1 = new_x1_pixel / img_width
    new_y1 = new_y1_pixel / img_height
    new_x2 = new_x2_pixel / img_width
    new_y2 = new_y2_pixel / img_height
    return [new_x1, new_y1, new_x2, new_y2]


class ScreenSeekeRMethod:
    def __init__(self, planner="gpt-4o-2024-05-13", grounder=None, configs=None):
        self.planner_model_name = planner
        # self.grounder = grounder
        self.grounder = grounder
        
        self.configs = configs if configs else {
            "max_search_depth": 3,
            "max_desicion_size": [1280, 768],
            # "min_crop_size": [960, 540],
            "min_crop_size": [1280, 768],
            "max_aspect_ratio": 2.5,
        }
        
        self.client = openai.OpenAI()

        self.override_generation_config = {
            "temperature": 0.0,
            "max_tokens": 2048,
        }
        self.logs = []
        self.debug_flag = True

    def set_generation_config(self, **kwargs):
        self.override_generation_config.update(kwargs)

    def debug_print(self, string):
        self.logs.append(string)
        if self.debug_flag:
            print(string)

    # Ablation: Simple patch scoring
    # def score_patch(self, patch, point):
    #     x1, y1, x2, y2 = patch
    #     x, y = point
    #     if x1 <= x <= x2 and y1 <= y <= y2:
    #         return 1.0
    #     else:
    #         return 0.0


    def score_patch(self, patch, point, mu=0.5, sigma=0.3):
        """
        Computes the Gaussian-weighted score of a point relative to a patch.

        Parameters:
            patch (tuple): A tuple (x1, y1, x2, y2) representing the bounding box of the patch.
            point (tuple): A tuple (x, y) representing the coordinates of the point.
            mu (float): The mean value (center of the Gaussian distribution) in normalized space. Default is 0.5.
            sigma (float): The standard deviation of the Gaussian distribution. Default is 0.3.

        Returns:
            float: The computed Gaussian score for the point.
        """
        if len(patch) != 4:
            raise ValueError("Patch must be a tuple or list with exactly four elements (x1, y1, x2, y2).")
        
        if len(point) != 2:
            raise ValueError("Point must be a tuple or list with exactly two elements (x, y).")
        
        x1, y1, x2, y2 = patch
        x, y = point

        if x2 <= x1 or y2 <= y1:
            raise ValueError("Invalid patch coordinates: (x2, y2) must be greater than (x1, y1).")

        width = x2 - x1
        height = y2 - y1

        relative_x = (x - x1) / width
        relative_y = (y - y1) / height

        if not (0 <= relative_x <= 1 and 0 <= relative_y <= 1):
            return 0.

        # Calculate the Euclidean distance between the point and the center (mu, mu)
        distance = math.sqrt((relative_x - mu) ** 2 + (relative_y - mu) ** 2)
        
        # Apply the Gaussian function
        gaussian_value = math.exp(-distance**2 / (2 * sigma**2))  # sigma controls the spread of the Gaussian
        
        return gaussian_value


    def smart_crop(self, image, bbox, padding, max_ratio=3, n_crops=3):
        """
        Crops an image, ensuring the aspect ratio is within the specified limits.
        If the aspect ratio is too extreme, splits the cropped area into `n_crops` pieces along the longer axis.

        Parameters:
            image (PIL.Image): The image to crop.
            bbox (tuple): A tuple (x1, y1, x2, y2) specifying the crop area (normalized between 0 and 1).
            padding (float): Padding ratio (e.g., 0.1 for 10% padding).
            max_ratio (float): The maximum allowed aspect ratio (either width/height or height/width).
            n_crops (int): The number of pieces to split the image into if the aspect ratio is too extreme.

        Returns:
            list: A list containing tuples: (bbox, cropped_image)
        """
        self.debug_print(f"smart cropping bbox={bbox}, padding={padding}, n_crops={n_crops}")
        cropped_box, cropped_image = crop_with_padding(image, bbox, padding)

        width, height = cropped_image.size
        aspect_ratio = width / height

        cropped_boxes = []
        cropped_images = []
        # If the aspect ratio is too extreme, split the image into `n_crops` pieces
        if aspect_ratio > max_ratio:
            # Split along the width (horizontal split)
            crop_width = 1 / n_crops
            for i in range(n_crops):
                left = i * crop_width
                right = min((i + 1) * crop_width, width)
                box_to_crop = (left, 0, right, 1)
                sub_cropped_box, sub_cropped_image = crop_with_padding(cropped_image, box_to_crop, padding=padding)
                # Convert the coordinates in the cropped image to the whole image passed in the parameter
                cropped_boxes.append(view_bbox_to_real_bbox(view_bbox=sub_cropped_box, viewport=cropped_box))
                cropped_images.append(sub_cropped_image)
        elif aspect_ratio < 1 / max_ratio:
            # Split along the height (vertical split)
            crop_height = 1 / n_crops
            for i in range(n_crops):
                top = i * crop_height
                bottom = min((i + 1) * crop_height, height)
                box_to_crop = (0, top, 1, bottom)
                sub_cropped_box, sub_cropped_image = crop_with_padding(cropped_image, box_to_crop, padding=padding)
                # Convert the coordinates in the cropped image to the whole image passed in the parameter
                cropped_boxes.append(view_bbox_to_real_bbox(view_bbox=sub_cropped_box, viewport=cropped_box))
                cropped_images.append(sub_cropped_image)
        else:
            # If the aspect ratio is within acceptable limits, return the single cropped image
            cropped_boxes = [cropped_box]
            cropped_images = [cropped_image]

        return list(zip(cropped_boxes, cropped_images))


    def extract_ui_ref_in_response(self, text):
        # Regular expressions to match <element>, <area>, <neighbor>, and <loc> patterns
        element_pattern = r'<element>(.*?)</element>'
        area_pattern = r'<area>(.*?)</area>'
        neighbor_pattern = r'<neighbor>(.*?)</neighbor>'
        loc_pattern = r'<loc>\((.*?)\)</loc>'

        # Helper function to extract tag content and corresponding locations
        def extract_tags_and_locations(tag_pattern, text):
            matches = re.findall(f'({tag_pattern})', text)
            extracted_data = []
            for match in matches:
                tag_content = match[0]
                extracted_data.append(tag_content)
            return extracted_data

        # Extract data for each tag type
        elements = extract_tags_and_locations(element_pattern, text)
        areas = extract_tags_and_locations(area_pattern, text)
        neighbors = extract_tags_and_locations(neighbor_pattern, text)

        # Combine all extracted data into a unified structure
        extracted_info = {
            "elements": elements,
            "areas": areas,
            "neighbors": neighbors
        }

        return extracted_info

    def extract_existence_decision(self, output):
        """
        Parses the JSON response from the EXISTENCE_PROMPT output.

        Args:
            output (str): The raw output string containing the JSON block.

        Returns:
            dict: A dictionary containing parsed data with keys `result` and optionally `new_instruction`.
            None: If parsing fails or required fields are missing.
        """
        try:
            # Extract JSON block
            start_idx = output.find("```json")
            end_idx = output.rfind("```")

            if start_idx == -1 or end_idx == -1 or start_idx >= end_idx:
                raise ValueError("No JSON block found in the output.")

            json_block = output[start_idx + len("```json"):end_idx].strip()

            # Parse JSON
            data = json.loads(json_block)

            # Validate required fields
            required_fields = ["result"]
            for field in required_fields:
                if field not in data:
                    raise ValueError(f"Missing required field: {field}")

            # Validate result field
            valid_results = ["is_target", "target_elsewhere", "target_not_found"]
            if data["result"] not in valid_results:
                raise ValueError(f"Invalid result value: {data['result']}. Must be one of {valid_results}.")

            # Optional field handling
            data["new_instruction"] = data.get("new_instruction", None)

            return data

        except (ValueError, json.JSONDecodeError) as e:
            print(f"Error parsing output: {e}")
            return None



    def chat(self, prompt, image=None, history=None):
        # Build the current message
        if image:
            current_msg = {
                "role": "user",
                "content": [
                    {
                        "type": "image_url",
                        "image_url": {
                            # "url": "data:image/png;base64," + convert_pil_image_to_base64(resize(image, target_height=768, target_width=768)),
                            "url": "data:image/png;base64," + convert_pil_image_to_base64(image),
                        }
                    },
                    {
                        "type": "text", "text": prompt
                    }
                ]
            }
        else:
            current_msg = {
                "role": "user",
                "content": [
                    {
                        "type": "text", "text": prompt
                    }
                ]
            }
        
        history = history if history else [] 
        messages = history + [current_msg]

        # make the request
        response = self.client.chat.completions.create(
            model=self.planner_model_name,
            messages=messages,
            temperature=self.override_generation_config['temperature'],
            max_tokens=self.override_generation_config['max_tokens'],
        )
        response_text = response.choices[0].message.content
        messages.append({"role": "assistant", "content": response_text})
        return response_text, messages
    
    def ground_only_positive(self, instruction, image):
        """Grounding entry point."""
        num_trail = 0
        while num_trail < 3:
            try:
                self.logs = []
                if isinstance(image, str):
                    image_path = image
                    assert os.path.exists(image_path) and os.path.isfile(image_path), "Invalid input image path."
                    image = Image.open(image_path).convert('RGB')
                assert isinstance(image, Image.Image), "Invalid input image."

                self.debug_print(f"Grounding instruction: {instruction}")
                
                flag, bbox = self.visual_search(instruction, image)

                if flag and bbox:
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
                self.debug_print(f"Grounding result: {result_dict}")
                return result_dict
            except openai.BadRequestError as e:
                self.debug_print(f"Bad request: {e}")
        # Error out of 3 trials
        return {
            "result": "negative",
            "bbox": None,
            "point": None,
            "raw_response": self.logs
        }
    
    def ground_with_grounder(self, instruction, image, viewport):
        result = self.grounder.ground_only_positive(instruction, image)
        view_bbox = result['bbox']  # the bbox in the current viewport
        # self.debug_print(f"Grounded bbox (viewport): {view_bbox}")
        if not view_bbox:
            # self.debug_print(f"Grounding failed. Raw response:\n{result}")
            return None, None  # should not happen
        real_bbox = view_bbox_to_real_bbox(view_bbox=view_bbox, viewport=viewport)
        # self.debug_print(f"Grounded bbox (real): {real_bbox}")
        return view_bbox, real_bbox
    

    def check_grounding_result(self, image, instruction, view_bbox):
        if view_bbox is None or view_bbox[2] <= view_bbox[0] or view_bbox[3] <= view_bbox[1]:
            self.debug_print(f"Invalid bbox: {view_bbox}. Skipping box drawing.")
            annotated_image = image
        else:
            annotated_image = plot_annotated_image(image, bbox=view_bbox)
        # annotated_image.save("tmp_annotated_image.png")
        history = get_sys_prompt_msg("You are an expert in using electronic devices and interacting with graphic interfaces.")
        position_inference_prompt = EXISTENCE_PROMPT.format(instruction=instruction)
        response, _ = self.chat(position_inference_prompt, annotated_image, history)
        self.debug_print(f"Result checking response: \n{response}")
        result_json = self.extract_existence_decision(response)
        return result_json

    def try_ground_in_patch(self, instruction, image, viewport, depth=0, area_desc=None):
        img_width, img_height = image.size
        view_direct_bbox, real_direct_bbox = self.ground_with_grounder(instruction, image, viewport)
        self.debug_print(f"Direct grounding result: view_bbox={view_direct_bbox}, real_bbox={real_direct_bbox}")

        # Ablation: No iterative search
        # if not view_direct_bbox:
        #     return False, None, None
        # else:
        #     return True, view_direct_bbox, real_direct_bbox
        
        
        # Call the planner model to check the result
        result = self.check_grounding_result(image, instruction, view_direct_bbox)

        if not result:
            return False, None, None

        if result['result'] == 'is_target':
            return True, view_direct_bbox, real_direct_bbox
        elif result['result'] == 'target_elsewhere':
            # Check if the target is in the current viewport
            view_direct_bbox, real_direct_bbox = self.ground_with_grounder(result["new_instruction"], image, viewport)
            return True, view_direct_bbox, real_direct_bbox
            # new_result = self.check_grounding_result(image, instruction, view_direct_bbox)
            # if new_result["result"] != "is_target":
            #     self.debug_print("Still wasn't able to ground the correct UI.")
            #     return False, None
        else:
            return False, None, None


    def score_patches(self, patches: dict, votes: list, weights=None):
        if weights:
            assert len(votes) == len(weights)
        else:
            weights = [1] * len(patches)

        scores = defaultdict(float)
        for (patch_key, patch), weight in zip(patches.items(), weights):
            for vote in votes:
                point_vote = [(vote[0] + vote[2]) / 2, (vote[1] + vote[3]) / 2]
                scores[patch_key] += self.score_patch(patch, point_vote) * weight
        return scores

    def auto_dilate_patch(self, image, patch, min_patch_size=(960, 540), max_ratio=2.5, target_ratio=16/9, num_splits=2):
        """
        Auto-dilate a patch to ensure it's within the specified limits.
        If an edge is greater than the minimum patch size and the aspect ratio is too extreme, split and pad to the minimum size.
        If only one edge is greater than the minimum size but the aspect ratio is not extreme, keep it as is.
        If neither edge is greater than the minimum size, pad to the minimum size.

        Parameters:
            patch (tuple): A tuple (x1, y1, x2, y2), 0-1 normalized, representing the patch.
            min_patch_size (tuple): A tuple (width, height) specifying the minimum size of result patches (in pixels).
            max_ratio (float): The maximum allowed aspect ratio (either width/height or height/width).
            target_ratio (float): The target aspect ratio for the result patch.
            num_splits (int): The number of splits to perform if the aspect ratio is too extreme.
        
        Returns:
            list: A list of tuples: (bbox), where bbox is the normalized (x1, y1, x2, y2).
        """

        def pad_patch(x1, y1, x2, y2, img_width, img_height, min_patch_size):
            """
            Helper function to pad a patch if it's smaller than the minimum size.
            Pads the patch in the normalized space (0-1).
            """
            pixel_width = (x2 - x1) * img_width
            pixel_height = (y2 - y1) * img_height

            width_deficit = max(0, min_patch_size[0] - pixel_width)
            height_deficit = max(0, min_patch_size[1] - pixel_height)

            # Pad horizontally
            if width_deficit > 0:
                delta = width_deficit / (2 * img_width)  # Normalize the padding
                if x1 - delta < 0:  # Patch is at the left edge
                    x1 = 0
                    x2 = min(1, x2 + width_deficit / img_width)
                elif x2 + delta > 1:  # Patch is at the right edge
                    x2 = 1
                    x1 = max(0, x1 - width_deficit / img_width)
                else:  # Centered padding
                    x1 = max(0, x1 - delta)
                    x2 = min(1, x2 + delta)

            # Pad vertically
            if height_deficit > 0:
                delta = height_deficit / (2 * img_height)  # Normalize the padding
                if y1 - delta < 0:  # Patch is at the top edge
                    y1 = 0
                    y2 = min(1, y2 + height_deficit / img_height)
                elif y2 + delta > 1:  # Patch is at the bottom edge
                    y2 = 1
                    y1 = max(0, y1 - height_deficit / img_height)
                else:  # Centered padding
                    y1 = max(0, y1 - delta)
                    y2 = min(1, y2 + delta)

            return x1, y1, x2, y2

        img_width, img_height = image.size
        x1, y1, x2, y2 = patch
        patch_width = x2 - x1
        patch_height = y2 - y1

        result = []

        # Convert normalized dimensions to pixels for comparison
        pixel_width = patch_width * img_width
        pixel_height = patch_height * img_height

        # Case 1: Neither edge is greater than the minimum size
        if pixel_width < min_patch_size[0] and pixel_height < min_patch_size[1]:
            # Pad the patch
            x1, y1, x2, y2 = pad_patch(x1, y1, x2, y2, img_width, img_height, min_patch_size)
            result.append((x1, y1, x2, y2))
            return result

        # Case 2: One edge is greater than the minimum size and aspect ratio is not extreme
        if (pixel_width >= min_patch_size[0] or pixel_height >= min_patch_size[1]) and \
        (pixel_width / pixel_height <= max_ratio and pixel_height / pixel_width <= max_ratio):
            result.append((x1, y1, x2, y2))
            return result

        # Case 3: One edge is greater than the minimum size and aspect ratio is too extreme
        if pixel_width / pixel_height > max_ratio:  # Width is too large compared to height
            split_width = patch_width / num_splits
            for i in range(num_splits):
                split_x1 = x1 + i * split_width
                split_x2 = split_x1 + split_width

                # Pad each split if needed
                split_x1, split_y1, split_x2, split_y2 = pad_patch(split_x1, y1, split_x2, y2, img_width, img_height, min_patch_size)

                result.append((split_x1, split_y1, split_x2, split_y2))
            return result

        elif pixel_height / pixel_width > max_ratio:  # Height is too large compared to width
            split_height = patch_height / num_splits
            for i in range(num_splits):
                split_y1 = y1 + i * split_height
                split_y2 = split_y1 + split_height

                # Pad each split if needed
                split_x1, split_y1, split_x2, split_y2 = pad_patch(x1, split_y1, x2, split_y2, img_width, img_height, min_patch_size)

                result.append((split_x1, split_y1, split_x2, split_y2))
            return result

        # Default case: Both edges are large.
        # Append the original patch.
        result.append((x1, y1, x2, y2))
        return result


    def visual_search(self, instruction, image, viewport=(0,0,1,1), depth=0, area_desc=None):
        img_width, img_height = image.size

        self.debug_print(f"Running visual search in viewport {viewport}, depth={depth}")
        if depth >= self.configs["max_search_depth"]:
            self.debug_print(f"Maximum depth of search reached. Grounding directly for {instruction}.")
            # Directly ground the target using the grounder
            flag, view_bbox, real_bbox = self.try_ground_in_patch(instruction, image, viewport)
            if flag:
                return True, real_bbox
            else:
                return False, None
        # Check if the image size is smaller than the minimum crop size. Note some extreme aspect ratio will not go into this if clause.
        if img_width <= self.configs["min_crop_size"][0] and img_height <= self.configs["min_crop_size"][1]:
            self.debug_print(f"Minimum crop size reached. Grounding directly for {instruction}..")
            flag, view_bbox, real_bbox = self.try_ground_in_patch(instruction, image, viewport)
            if flag:
                return True, real_bbox
            else:
                return False, None

        history = get_sys_prompt_msg("You are an expert in using electronic devices and interacting with graphic interfaces.")
        position_inference_prompt = POSITION_INFERENCE_PROMPT_TOP_LEVEL.format(instruction=instruction)

        position_inference_response, history = self.chat(position_inference_prompt, image, history)
        self.debug_print("**Position inference response**:\n" + position_inference_response)

        candidates = self.extract_ui_ref_in_response(position_inference_response)
        if len(candidates) == 0:
            self.debug_print("No candidates found. Target is not in currect patch.")
            return False, None

        # Collect possible areas and elements
        groundings = []
        patches = {}
        # 1. Direct Grounding
        view_direct_bbox, real_direct_bbox = self.ground_with_grounder(instruction, image, viewport)
        if view_direct_bbox:
            patches[1] = view_direct_bbox
            groundings.append(view_direct_bbox)
            self.debug_print(f"Direct grounding result: view_bbox={view_direct_bbox}, real_bbox={real_direct_bbox}")
        else:
            self.debug_print(f"Direct grounding failed.")

        # 2. Build candidate boxes by parsing the planner response
        for i, obj_ref in enumerate(itertools.chain(
            candidates["elements"], 
            candidates["areas"], 
            candidates["neighbors"]
        ), start=2):
            self.debug_print(f"Calling the grounder to search for '{obj_ref}'.")
            view_bbox, real_bbox = self.ground_with_grounder(obj_ref, image, viewport)
            if view_bbox:
                patches[i] = view_bbox
                groundings.append(view_bbox)
                # self.debug_print(f"Grounder result: view_bbox={view_bbox}, real_bbox={real_bbox}")
            else:
                self.debug_print(f"Grounding failed for {obj_ref}.")
                continue
        self.debug_print(f"Patches to select: {patches}")

        # !!! DEBUG !!!
        # Draw all regions on the image for debugging
        # debug_image = plot_debug_image(image, bboxes=list(patches.values()), labels=list(patches.keys()))
        # debug_image.save(f"debug_grounder_pos_depth{depth}.png")
        # !!! DEBUG !!!

        # Filter boxes
        updated_patches = {}
        counter = 1
        # The image size is not enough to make a final desicion. Dilate small grounding results.
        for patch_id, patch in patches.items():
            dilated_patches = self.auto_dilate_patch(image, patch, min_patch_size=self.configs["min_crop_size"], max_ratio=self.configs["max_aspect_ratio"], target_ratio=16/9, num_splits=2)
            for new_patch in dilated_patches:
                updated_patches[counter] = new_patch
                counter += 1
        

        # Add 4-split patches
        # updated_patches[len(updated_patches) + 1] = [0, 0, 0.5, 0.5]  # Top-left
        # updated_patches[len(updated_patches) + 1] = [0.5, 0, 1, 0.5]  # Top-right
        # updated_patches[len(updated_patches) + 1] = [0, 0.5, 0.5, 1]  # Bottom-left
        # updated_patches[len(updated_patches) + 1] = [0.5, 0.5, 1, 1]  # Bottom-right

        # !!! DEBUG !!!
        # Draw all regions on the image for debugging
        # debug_image = plot_debug_image(image, bboxes=list(updated_patches.values()), labels=list(updated_patches.keys()))
        # debug_image.save(f"debug_candidate_patch_pos_depth{depth}.png")
        # !!! DEBUG !!!

        
        patches = updated_patches
        # votes = [item[1] for item in list(candidates["elements"]) + list(candidates["areas"]) + list(candidates["neighbors"])]
        # TODO: FINE-GRAINED VOTING
        patch_scores = self.score_patches(patches, votes=groundings)

        # Sort patches by their scores
        sorted_patches = dict(sorted(patches.items(), key=lambda x: patch_scores[x[0]], reverse=True))

        # Perform NMS
        nms_patches = non_maximum_suppression(list(sorted_patches.values()), iou_threshold=0.5, keep="order")
        self.debug_print(f"Non-maximum suppression: {len(sorted_patches)} -> {len(nms_patches)}")

        self.debug_print(f"NMS patches: {nms_patches}")

        # Iterate through the sorted patches to search
        for selected_patch in nms_patches:
            sub_real_viewport = view_bbox_to_real_bbox(view_bbox=selected_patch, viewport=viewport)
            self.debug_print(f"Searching in viewport {sub_real_viewport}, view_bbox={selected_patch}")
            _, sub_image = crop_with_padding(image, selected_patch, padding=0)
            terminate_flag, bbox = self.visual_search(instruction, sub_image, depth=depth + 1, viewport=sub_real_viewport, area_desc=None)
            if terminate_flag:
                return terminate_flag, bbox
        
        # This really should not happen
        return False, None
