"""
duvo-eye with a zoom second pass (MAI-UI protocol, arXiv 2512.22047): ground on the full screenshot, crop a window
of half the width and height centred on that point (clamped inside the image), resize the crop back to the original
size, ground again on the crop, and map the point back. If the second point lands within 4% of the crop edge, the
target is probably outside the crop and the first point is kept. Serve the model as for duvo_eye_1.
"""
from models.duvo_eye_1 import DuvoEye1Model, PROMPT_TEMPLATE, convert_pil_image_to_base64
from PIL import Image
import os


class DuvoEyeZoomModel(DuvoEye1Model):
    def _ask(self, image, instruction):
        response = self.client.chat.completions.create(
            model=self.model_name,
            messages=[{"role": "user", "content": [
                {"type": "image_url", "image_url": {"url": f"data:image/png;base64,{convert_pil_image_to_base64(image)}"}},
                {"type": "text", "text": PROMPT_TEMPLATE.format(instruction=instruction)}]}],
            temperature=self.override_generation_config["temperature"],
            max_tokens=64,
            extra_body={"chat_template_kwargs": {"enable_thinking": False}},
        )
        text = response.choices[0].message.content
        return self._parse_point(text), text

    def ground_only_positive(self, instruction, image):
        if isinstance(image, str):
            assert os.path.isfile(image), "Invalid input image path."
            image = Image.open(image).convert("RGB")
        W, H = image.size
        coarse, text = self._ask(image, instruction)
        point, raw = None, [text]
        if coarse is not None:
            point = [coarse[0] / 1000.0, coarse[1] / 1000.0]
            cw, ch = W / 2, H / 2
            x0 = min(max(0, point[0] * W - cw / 2), W - cw)
            y0 = min(max(0, point[1] * H - ch / 2), H - ch)
            crop = image.crop((int(x0), int(y0), int(x0 + cw), int(y0 + ch))).resize((W, H), Image.LANCZOS)
            fine, text2 = self._ask(crop, instruction)
            raw.append(text2)
            if fine is not None and 40 <= fine[0] <= 960 and 40 <= fine[1] <= 960:
                point = [(x0 + fine[0] / 1000.0 * cw) / W, (y0 + fine[1] / 1000.0 * ch) / H]
        return {"result": "positive", "format": "x1y1x2y2", "raw_response": raw, "bbox": None, "point": point}
