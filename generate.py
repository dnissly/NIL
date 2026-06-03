# Jersey Generation Script - Rally House
#
# This script generates jersey images based on a CSV input file and team asset folders.
# It processes front and back jersey images, composites numbers, and renders nameplates.
# It also creates a combined image for each player, overlaying the front jersey on the back.
# Generates based on netsuite saved search from NIL buyer
# Asset Folders are named as "{Team}-{Color List}" and contain:
# - "blanks" folder with front.png and back.png
# - "number_front", "number_back", and "number_shoulder" folders with digit pngs (0-9) approx 2880 px tall
# - "fonts" folder with NamePlate.otf
# - "coords.json" file with bounding box coordinates for various elements and color hex for nameplate
# - examples folder with example jersey images for reference (not required for generation)

import os
import pandas as pd
import json
from PIL import Image, ImageDraw, ImageFont
import re
import shutil
import numpy as np
import easygui

# Paths
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
CSV_PATH = os.path.join(BASE_DIR, "jerseystocreate.csv")
OUTPUT_DIR = os.path.join(BASE_DIR, "output")

def load_coords_json(team_folder):
    coords_path = os.path.join(team_folder, "coords.json")
    with open(coords_path, "r") as f:
        coords = json.load(f)
    return coords

def composite_numbers(number_str, number_folder, target_box):
    digits = list(str(number_str))
    digit_imgs = [Image.open(os.path.join(number_folder, f"{d}.png")).convert("RGBA") for d in digits]
    widths, heights = zip(*(img.size for img in digit_imgs))

    x0, y0, x1, y1 = target_box
    box_width = int(round(x1 - x0))
    box_height = int(round(y1 - y0))

    # Special case: single digit "1"
    if len(digits) == 1 and digits[0] == '1':
        # Only scale vertically, keep aspect ratio for width, and make 10% smaller
        orig_img = digit_imgs[0]
        scale = box_height / heights[0] * 0.9  # 10% smaller
        new_width = int(widths[0] * scale)
        new_height = int(heights[0] * scale)
        scaled = orig_img.resize((new_width, new_height), Image.LANCZOS)
        final = Image.new("RGBA", (box_width, box_height), (0,0,0,0))
        offset_x = (box_width - new_width) // 2
        offset_y = (box_height - new_height) // 2
        final.paste(scaled, (offset_x, offset_y), scaled)
        return final

    # For single digit (not "1"), treat as if it's two digits for scaling, but only render one
    elif len(digits) == 1:
        composite_width = widths[0] * 2
        composite_height = heights[0]
        composite = Image.new("RGBA", (composite_width, composite_height), (0,0,0,0))
        offset_x = (composite_width - widths[0]) // 2
        composite.paste(digit_imgs[0], (offset_x, 0), digit_imgs[0])
        scale = box_height / composite_height
        new_width = int(composite_width * scale)
        new_height = box_height
        scaled = composite.resize((new_width, new_height), Image.LANCZOS)
        final = Image.new("RGBA", (box_width, box_height), (0,0,0,0))
        offset_x = (box_width - new_width) // 2
        final.paste(scaled, (offset_x, 0), scaled)
        return final

    elif len(digits) == 2 and digits[0] == '1' and digits[1] == '1':
        # Special case for "11": no overlap, keep aspect ratio, small gap
        gap = int(widths[0] * 0.2)
        composite_width = widths[0] + widths[1] + gap
        composite_height = max(heights)
        composite = Image.new("RGBA", (composite_width, composite_height), (0,0,0,0))
        # Paste first "1"
        composite.paste(digit_imgs[0], (0, (composite_height - heights[0]) // 2), digit_imgs[0])
        # Paste second "1" with gap
        composite.paste(digit_imgs[1], (widths[0] + gap, (composite_height - heights[1]) // 2), digit_imgs[1])
        # Scale only vertically, keep original width
        x0, y0, x1, y1 = target_box
        box_width = int(round(x1 - x0))
        box_height = int(round(y1 - y0))
        scale = box_height / composite_height
        new_width = int(composite_width * scale)
        new_height = box_height
        scaled = composite.resize((new_width, new_height), Image.LANCZOS)
        # Center horizontally in the bounding box
        final = Image.new("RGBA", (box_width, box_height), (0,0,0,0))
        offset_x = (box_width - new_width) // 2
        final.paste(scaled, (offset_x, 0), scaled)
        return final

    elif len(digits) == 2 and ('1' in digits):
        # Two-digit case with exactly one '1': cap horizontal stretch of the other digit at 1.4x
        # 1) Scale both digits to fit box height while preserving aspect
        scaled = []
        base_widths = []
        for i, img in enumerate(digit_imgs):
            s = box_height / float(img.size[1])
            w = max(1, int(round(img.size[0] * s)))
            base_widths.append(w)
            scaled.append(img.resize((w, box_height), Image.LANCZOS))

        base_total = sum(base_widths)
        final = Image.new("RGBA", (box_width, box_height), (0,0,0,0))

        if base_total <= box_width:
            # 2) Add extra width only to the non-1 digit, capped at +40% (1.4x total)
            idx_non1 = 0 if digits[0] != '1' else 1
            non1_w = base_widths[idx_non1]
            max_increase = int(round(non1_w * 0.1))  # CHANGE CAP HERE
            extra_needed = box_width - base_total
            increase = max(0, min(extra_needed, max_increase))

            # Resize only the non-1 digit wider (others unchanged)
            if increase > 0:
                new_w = non1_w + increase
                scaled[idx_non1] = scaled[idx_non1].resize((new_w, box_height), Image.LANCZOS)
                base_widths[idx_non1] = new_w

            comp_w = sum(base_widths)
            composite = Image.new("RGBA", (comp_w, box_height), (0,0,0,0))
            x = 0
            for img in scaled:
                composite.paste(img, (x, 0), img)
                x += img.size[0]

            # Center composite in box; leftover width becomes side padding
            offset_x = (box_width - comp_w) // 2
            final.paste(composite, (offset_x, 0), composite)
            return final
        else:
            # 3) If too wide, uniformly compress horizontally to fit box width
            comp_w = base_total
            composite = Image.new("RGBA", (comp_w, box_height), (0,0,0,0))
            x = 0
            for img in scaled:
                composite.paste(img, (x, 0), img)
                x += img.size[0]
            squeezed = composite.resize((box_width, box_height), Image.LANCZOS)
            return squeezed

    else:
        composite_width = sum(widths)
        composite_height = max(heights)
        composite = Image.new("RGBA", (composite_width, composite_height), (0,0,0,0))
        x = 0
        for img in digit_imgs:
            y = (composite_height - img.size[1]) // 2
            composite.paste(img, (x, y), img)
            x += img.size[0]

        # Stretch the composite image to exactly fit the bounding box (ignore aspect ratio)
        stretched = composite.resize((box_width, box_height), Image.LANCZOS)
        return stretched

def fit_text_to_box(text, font_path, box_width, box_height, spacing_factor, word_spacing_factor=0.33, max_font_size=400, min_font_size=10):
    # Binary search for best font size to fill the box, with a little margin for descenders
    best_font = None
    best_size = None
    margin = int(box_height * 0.08)  # 8% margin at the bottom

    def _advance(font_obj, s):
        # Prefer accurate advance width when available
        if hasattr(font_obj, "getlength"):
            return font_obj.getlength(s)
        bbox = font_obj.getbbox(s)
        return bbox[2] - bbox[0]

    while min_font_size <= max_font_size:
        mid = (min_font_size + max_font_size) // 2
        font = ImageFont.truetype(font_path, mid)

        spacing = int(mid * spacing_factor)          # letter-to-letter spacing
        word_spacing = int(mid * word_spacing_factor)  # space between words (single increment)

        # Measure char advances; treat spaces as 0 here (we add word_spacing separately)
        char_advances = [(_advance(font, ch) if ch != ' ' else 0) for ch in text]

        # Compute total width using the same rules we use when drawing
        total_width = 0
        for i, ch in enumerate(text):
            if ch == ' ':
                total_width += word_spacing
            else:
                total_width += char_advances[i]
                # add letter-spacing only if next char is not a space
                if i < len(text) - 1 and text[i + 1] != ' ':
                    total_width += spacing

        bbox = font.getbbox(text)
        h = bbox[3] - bbox[1]
        if total_width <= box_width and h <= (box_height - margin):
            best_font = font
            best_size = (total_width, h, bbox, spacing, char_advances, word_spacing)
            min_font_size = mid + 1
        else:
            max_font_size = mid - 1
    return best_font, best_size

def hex_to_rgba(hex_color):
    hex_color = hex_color.lstrip('#')
    lv = len(hex_color)
    if lv == 6:
        return tuple(int(hex_color[i:i+2], 16) for i in (0, 2, 4)) + (255,)
    elif lv == 8:
        return tuple(int(hex_color[i:i+2], 16) for i in (0, 2, 4, 6))
    else:
        raise ValueError("Invalid hex color")

def _srgb_to_linear(c):  # c in [0..1]
    return np.where(c <= 0.04045, c / 12.92, ((c + 0.055) / 1.055) ** 2.4)

def _linear_to_srgb(c):  # c in [0..1]
    return np.where(c <= 0.0031308, 12.92 * c, 1.055 * (c ** (1 / 2.4)) - 0.055)

def resize_rgba_linear_pm(img, size, resample=Image.LANCZOS):
    if img.mode != "RGBA":
        return img.resize(size, resample)

    # Split channels
    r, g, b, a = img.split()
    r = np.asarray(r, dtype=np.float32) / 255.0
    g = np.asarray(g, dtype=np.float32) / 255.0
    b = np.asarray(b, dtype=np.float32) / 255.0
    a = np.asarray(a, dtype=np.float32) / 255.0

    # Convert to linear light, premultiply by alpha
    r_lin = _srgb_to_linear(r)
    g_lin = _srgb_to_linear(g)
    b_lin = _srgb_to_linear(b)
    r_pm = r_lin * a
    g_pm = g_lin * a
    b_pm = b_lin * a

    # Helper to resize float planes with Pillow
    def _resize_plane(arrf):
        pilf = Image.fromarray(arrf, mode="F")
        return np.asarray(pilf.resize(size, resample), dtype=np.float32)

    r_pm_rs = _resize_plane(r_pm)
    g_pm_rs = _resize_plane(g_pm)
    b_pm_rs = _resize_plane(b_pm)
    a_rs    = _resize_plane(a)

    # Unpremultiply (avoid div-by-zero), convert back to sRGB
    eps = 1e-6
    a_safe = np.maximum(a_rs, eps)
    r_lin_rs = r_pm_rs / a_safe
    g_lin_rs = g_pm_rs / a_safe
    b_lin_rs = b_pm_rs / a_safe

    r_srgb = _linear_to_srgb(np.clip(r_lin_rs, 0.0, 1.0))
    g_srgb = _linear_to_srgb(np.clip(g_lin_rs, 0.0, 1.0))
    b_srgb = _linear_to_srgb(np.clip(b_lin_rs, 0.0, 1.0))

    r8 = np.clip(r_srgb * 255.0 + 0.5, 0, 255).astype(np.uint8)
    g8 = np.clip(g_srgb * 255.0 + 0.5, 0, 255).astype(np.uint8)
    b8 = np.clip(b_srgb * 255.0 + 0.5, 0, 255).astype(np.uint8)
    a8 = np.clip(a_rs * 255.0 + 0.5, 0, 255).astype(np.uint8)

    return Image.merge("RGBA", (Image.fromarray(r8, "L"),
                                Image.fromarray(g8, "L"),
                                Image.fromarray(b8, "L"),
                                Image.fromarray(a8, "L")))

def render_nameplate(text, font_path, nameplate_obj, rotation_angle=0, y_offset_extra=0):
    coords = nameplate_obj["coords"]
    color = nameplate_obj.get("color", "#FFFFFF")
    spacing_factor = nameplate_obj.get("spacing_factor", 0.06)  # Default to 0.06 if not present
    word_spacing_factor = nameplate_obj.get("word_spacing_factor", 0.33)  # 1/3 em default
    v_align = (nameplate_obj.get("vertical_align") or "top").lower()  # top | center | bottom
    x0, y0, x1, y1 = coords
    box_width = int(round(x1 - x0))
    box_height = int(round(y1 - y0))
    font, (total_width, h, bbox, spacing, char_widths, word_spacing) = fit_text_to_box(
        text, font_path, box_width, box_height, spacing_factor, word_spacing_factor
    )
    img = Image.new("RGBA", (box_width, box_height), (0,0,0,0))
    draw = ImageDraw.Draw(img)
    fill_color = hex_to_rgba(color)

    # Horizontal center
    x_cursor = (box_width - total_width) // 2

    # Vertical alignment
    if v_align == "center":
        y_offset = (box_height - h) // 2 - bbox[1] + y_offset_extra
    elif v_align == "bottom":
        y_offset = box_height - h - bbox[1] + y_offset_extra
    else:  # top (default)
        y_offset = -bbox[1] + y_offset_extra

    # Draw with special handling for spaces
    for i, char in enumerate(text):
        if char == ' ':
            x_cursor += word_spacing  # single increment; no letter-spacing around spaces
            continue

        draw.text((x_cursor, y_offset), char, font=font, fill=fill_color)
        x_cursor += char_widths[i]
        if i < len(text) - 1 and text[i + 1] != ' ':
            x_cursor += spacing

    if "rotation" in nameplate_obj:
        rotation_angle = nameplate_obj["rotation"]
    if rotation_angle != 0:
        img = img.rotate(rotation_angle, expand=True, resample=Image.BICUBIC, fillcolor=(0,0,0,0))
        # Crop transparent padding so top alignment remains true after rotation
        bbox_img = img.getbbox()
        if bbox_img:
            img = img.crop(bbox_img)
    else:
        # Also crop for consistency when not rotated
        bbox_img = img.getbbox()
        if bbox_img:
            img = img.crop(bbox_img)
    return img

def process_front(row, team_folder, coords):
    player_name, player_number = extract_name_and_number(row["Jersey Characters"])
    blanks_folder = os.path.join(team_folder, "blanks")
    number_folder = os.path.join(team_folder, "number_front")
    blank_front_path = os.path.join(blanks_folder, "front.png")
    blank_img = Image.open(blank_front_path).convert("RGBA")

    # --- Handle both dict and list formats for FrontNumber ---
    front_number_obj = coords["FrontNumber"]
    if isinstance(front_number_obj, dict):
        front_number_coords = front_number_obj["coords"]
        front_number_rotation = front_number_obj.get("rotation", coords.get("NamePlate", {}).get("rotation", 0))
    else:
        front_number_coords = front_number_obj
        front_number_rotation = coords.get("NamePlate", {}).get("rotation", 0)

    temp = blank_img.copy()
    if str(player_number).strip():
        number_img = composite_numbers(player_number, number_folder, front_number_coords)

        if front_number_rotation != 0:
            number_img = number_img.rotate(front_number_rotation, expand=True, resample=Image.BICUBIC, fillcolor=(0,0,0,0))
        x0, y0, x1, y1 = [int(round(c)) for c in front_number_coords]
        # Center if rotated
        if front_number_rotation != 0:
            num_w, num_h = number_img.size
            box_w = int(round(x1 - x0))
            box_h = int(round(y1 - y0))
            paste_x = x0 + (box_w - num_w) // 2
            paste_y = y0 + (box_h - num_h) // 2
        else:
            paste_x, paste_y = x0, y0

        # Special case: shift single-digit '4' left by 30px on front
        if str(player_number).strip() == '4':
            paste_x -= 30
        # Special case: shift single-digit '1' left by 5px on front
        elif str(player_number).strip() == '1':
            paste_x -= 5

        temp.paste(number_img, (paste_x, paste_y), number_img)
        # Add front shoulder numbers
        add_shoulder_number(temp, player_number, number_folder, coords["FLShoulder"])
        add_shoulder_number(temp, player_number, number_folder, coords["FRShoulder"])
    alpha = blank_img.split()[-1]
    temp.putalpha(alpha)
    out_name = f"{row['Name']}-3.png"
    out_path = os.path.join(OUTPUT_DIR, out_name)
    temp.save(out_path)
    print(f"Saved {out_path}")

def process_back(row, team_folder, coords):
    player_name, player_number = extract_name_and_number(row["Jersey Characters"])
    blanks_folder = os.path.join(team_folder, "blanks")
    number_folder = os.path.join(team_folder, "number_back")
    fonts_folder = os.path.join(team_folder, "fonts")
    font_path = os.path.join(fonts_folder, "NamePlate.otf")
    blank_back_path = os.path.join(blanks_folder, "back.png")
    blank_img = Image.open(blank_back_path).convert("RGBA")

    # Respect coords.json exactly (no Y shifting for long names)
    rotation_angle = coords.get("NamePlate", {}).get("rotation", 0)
    nameplate_obj = dict(coords["NamePlate"])  # do not modify coords
    formatted_name = apply_nameplate_casing(player_name)
    nameplate_img = render_nameplate(formatted_name, font_path, nameplate_obj, rotation_angle, 0)

    x0, y0, x1, y1 = [int(round(c)) for c in coords["NamePlate"]["coords"]]
    box_width = int(round(x1 - x0))
    box_height = int(round(y1 - y0))

    # Top-align inside the box; center horizontally regardless of rotation
    np_w, np_h = nameplate_img.size
    paste_x = x0 + (box_width - np_w) // 2
    paste_y = y0  # top aligned to the box

    temp = blank_img.copy()
    temp.paste(nameplate_img, (paste_x, paste_y), nameplate_img)

    # --- Handle both dict and list formats for BackNumber ---
    back_number_obj = coords["BackNumber"]
    if isinstance(back_number_obj, dict):
        back_number_coords = back_number_obj["coords"]
        back_number_rotation = back_number_obj.get("rotation", coords.get("NamePlate", {}).get("rotation", 0))
    else:
        back_number_coords = back_number_obj
        back_number_rotation = coords.get("NamePlate", {}).get("rotation", 0)

    if str(player_number).strip():
        number_img = composite_numbers(player_number, number_folder, back_number_coords)
        if back_number_rotation != 0:
            number_img = number_img.rotate(back_number_rotation, expand=True, resample=Image.BICUBIC, fillcolor=(0,0,0,0))
        x0, y0, x1, y1 = [int(round(c)) for c in back_number_coords]
        if back_number_rotation != 0:
            num_w, num_h = number_img.size
            box_w = int(round(x1 - x0))
            box_h = int(round(y1 - y0))
            paste_x_num = x0 + (box_w - num_w) // 2
            paste_y_num = y0 + (box_h - num_h) // 2
        else:
            paste_x_num, paste_y_num = x0, y0

        # Special cases
        if str(player_number).strip() == '4':
            paste_x_num -= 25
        elif str(player_number).strip() == '1':
            paste_x_num -= 0

        temp.paste(number_img, (paste_x_num, paste_y_num), number_img)
        add_shoulder_number(temp, player_number, number_folder, coords["BLShoulder"])
        add_shoulder_number(temp, player_number, number_folder, coords["BRShoulder"])
    alpha = blank_img.split()[-1]
    temp.putalpha(alpha)
    out_name = f"{row['Name']}-2.png"
    out_path = os.path.join(OUTPUT_DIR, out_name)
    temp.save(out_path)
    print(f"Saved {out_path}")

def add_shoulder_number(base_img, number_str, number_folder, shoulder_obj):
    if not str(number_str).strip():
        return
    coords = shoulder_obj["coords"]
    rotation = shoulder_obj.get("rotation", 0)
    x0, y0, x1, y1 = [int(round(c)) for c in coords]
    box_width = int(round(x1 - x0))
    box_height = int(round(y1 - y0))

    # Prepare digit images
    digits = list(str(number_str))
    digit_imgs = [Image.open(os.path.join(number_folder, f"{d}.png")).convert("RGBA") for d in digits]
    widths, heights = zip(*(img.size for img in digit_imgs))

    # For single digit, treat as if it's two digits for scaling, but only render one
    if len(digits) == 1:
        composite_width = widths[0] * 2
        composite_height = heights[0]
        composite = Image.new("RGBA", (composite_width, composite_height), (0,0,0,0))
        offset_x = (composite_width - widths[0]) // 2
        composite.paste(digit_imgs[0], (offset_x, 0), digit_imgs[0])
    elif len(digits) == 2 and digits[0] == '1' and digits[1] == '1':
        # Special case for "11": no overlap, keep aspect ratio, small gap
        gap = int(widths[0] * 0.10)
        composite_width = widths[0] + widths[1] + gap
        composite_height = max(heights)
        composite = Image.new("RGBA", (composite_width, composite_height), (0,0,0,0))
        composite.paste(digit_imgs[0], (0, (composite_height - heights[0]) // 2), digit_imgs[0])
        composite.paste(digit_imgs[1], (widths[0] + gap, (composite_height - heights[1]) // 2), digit_imgs[1])
    else:
        composite_width = sum(widths)
        composite_height = max(heights)
        composite = Image.new("RGBA", (composite_width, composite_height), (0,0,0,0))
        x = 0
        for img in digit_imgs:
            y = (composite_height - img.size[1]) // 2
            composite.paste(img, (x, y), img)
            x += img.size[0]

    # Scale so the composite fills the bounding box vertically, then squish horizontally
    scale = box_height / composite_height
    shoulder_squish = 0.65  # 65% of the normal width
    new_width = int(composite_width * scale * shoulder_squish)
    new_height = box_height
    scaled = composite.resize((max(1, new_width), max(1, new_height)), Image.LANCZOS)

    # Center horizontally in the bounding box
    final = Image.new("RGBA", (box_width, box_height), (0,0,0,0))
    offset_x = (box_width - new_width) // 2
    final.paste(scaled, (offset_x, 0), scaled)

    # Rotate the number image
    rotated_number = final.rotate(rotation, expand=True, resample=Image.BICUBIC, fillcolor=(0,0,0,0))
    # Paste the rotated number at the top-left of the bounding box
    base_img.paste(rotated_number, (x0, y0), rotated_number)

def process_combo(row, front_path, back_path):
    combo_width, combo_height = 700, 1000
    scale = 0.68

    # Load images
    front_img = Image.open(front_path).convert("RGBA")
    back_img = Image.open(back_path).convert("RGBA")

    # Scale images (linear-light premultiplied alpha)
    front_scaled = resize_rgba_linear_pm(
        front_img, (int(front_img.width * scale), int(front_img.height * scale)), Image.LANCZOS
    )
    back_scaled = resize_rgba_linear_pm(
        back_img, (int(back_img.width * scale), int(back_img.height * scale)), Image.LANCZOS
    )

    # Create blank canvas
    combo_img = Image.new("RGBA", (combo_width, combo_height), (0, 0, 0, 0))

    # Place back jersey at top-left
    combo_img.paste(back_scaled, (0, 90), back_scaled)

    # Place front jersey 40% down and 40% right, overlayed
    front_x = int(combo_width * 0.30) + 13
    front_y = int(combo_height * 0.18) + 70
    combo_img.paste(front_scaled, (front_x, front_y), front_scaled)

    # Save combo image (preserve ICC if available)
    icc = front_img.info.get("icc_profile") or back_img.info.get("icc_profile")
    out_name = f"{row['Name']}-1.png"
    out_path = os.path.join(OUTPUT_DIR, out_name)
    if icc:
        combo_img.save(out_path, icc_profile=icc)
    else:
        combo_img.save(out_path)
    print(f"Saved {out_path}")

def apply_overlay_to_file(result_path, overlay_img):
    try:
        base_pil = Image.open(result_path)
        icc = base_pil.info.get("icc_profile")
        base = base_pil.convert("RGBA")

        ov = overlay_img
        if ov.size != base.size:
            ov = ov.resize(base.size, Image.LANCZOS)

        # Overlay on top
        base.paste(ov, (0, 0), ov)

        if icc:
            base.save(result_path, icc_profile=icc)
        else:
            base.save(result_path)
    except Exception as e:
        print(f"[WARN] Failed to overlay youth.png on {result_path}: {e}")

def extract_last_name_and_suffix(full_name):
    # Remove nicknames in quotes
    name = re.sub(r'"[^"]*"', '', full_name).strip()
    # Split by spaces
    parts = name.split()
    # List of common suffixes
    suffixes = {"Jr.", "Sr.", "II", "III", "IV", "V"}
    last_name = ""
    suffix = ""
    if len(parts) >= 2:
        # Check if last part is a suffix
        if parts[-1] in suffixes:
            last_name = parts[-2]
            suffix = parts[-1]
        else:
            last_name = parts[-1]
    return f"{last_name} {suffix}".strip()


def apply_nameplate_casing(name: str) -> str:
    """Uppercase the name while keeping known lowercase patterns (e.g., Mc)."""
    if not name:
        return ""

    def _format_token(token: str) -> str:
        # Ignore tokens that are purely separators or non-alpha
        if not any(ch.isalpha() for ch in token):
            return token

        upper = token.upper()

        # Handle Mc* prefix: keep the 'c' lowercase, rest uppercase
        if upper.startswith("MC") and len(upper) >= 3 and upper[2].isalpha():
            return "Mc" + upper[2:]

        return upper

    # Split on spaces, hyphens, or apostrophes but keep separators
    parts = re.split(r"(\s+|[-']+)", name.strip())
    return "".join(_format_token(p) if p and not p.isspace() else p for p in parts)

def extract_name_and_number(jersey_characters):
    """Extract alphabetic name and numeric jersey value from the mixed field."""
    if pd.isna(jersey_characters):
        return "", ""

    # Normalize to string so downstream .strip() calls do not fail on numeric input
    if isinstance(jersey_characters, (int, np.integer)):
        raw_value = str(jersey_characters)
    elif isinstance(jersey_characters, float):
        raw_value = str(int(jersey_characters)) if jersey_characters.is_integer() else str(jersey_characters)
    else:
        raw_value = str(jersey_characters)

    value = raw_value.strip()
    if not value:
        return "", ""

    # If the field only contains numbers, treat it as "no name" by request
    digits_only = ''.join(ch for ch in value if ch.isdigit())
    has_letters = any(ch.isalpha() for ch in value)
    if digits_only and not has_letters:
        return "", digits_only

    match = re.match(r"([A-Za-z\s]+)\s*(\d+)$", value)
    if match:
        name = match.group(1).strip()
        number = match.group(2)
        return name, number

    # fallback: treat all non-digits as name, remaining digits as number
    name = ''.join([c for c in value if not c.isdigit()]).strip()
    number = digits_only
    return name, number

def read_csv_with_fallback(path):
    for enc in ("utf-8-sig", "cp1252", "latin1"):
        try:
            df = pd.read_csv(path, encoding=enc)
            print(f"[INFO] Loaded CSV with encoding: {enc}")
            return df
        except UnicodeDecodeError:
            continue
    raise UnicodeDecodeError("Could not decode CSV with common encodings.")

def get_csv_path():
    file_path = easygui.fileopenbox(
        title="Select jersey input CSV file",
        default="*.csv",
        filetypes=["*.csv", "*.*"]
    )
    return file_path

def main():
    if os.path.exists(OUTPUT_DIR):
        shutil.rmtree(OUTPUT_DIR)
    if not os.path.exists(OUTPUT_DIR):
        os.makedirs(OUTPUT_DIR)

    assets_root = os.path.join(BASE_DIR, "bin")

    # Prompt user for CSV file
    csv_path = get_csv_path()
    if not csv_path:
        print("[ERROR] No CSV file selected. Exiting.")
        return

    youth_overlay_img = None
    youth_overlay_path = os.path.join(assets_root, "youth.png")

    df = read_csv_with_fallback(csv_path)

    sport_column = None
    for candidate in ("Sport Specific", "Sport"):
        if candidate in df.columns:
            sport_column = candidate
            break

    if sport_column is None:
        print("[ERROR] Input CSV must include a 'Sport Specific' or 'Sport' column. Exiting.")
        return

    for idx, row in df.iterrows():
        # Identify sport folder first, then locate specific team/color assets within it
        sport_value = str(row.get(sport_column, "")).strip()
        if not sport_value or sport_value.lower() == "nan":
            print(f"[WARN] Missing sport for row {idx}. Skipping.")
            continue

        sport_folder = os.path.join(assets_root, sport_value)
        if not os.path.isdir(sport_folder):
            print(f"[WARN] Sport assets not found for '{sport_value}' in {assets_root}. Skipping.")
            continue

        team_name = str(row.get("Team", "")).strip()
        color_name = str(row.get("Color List", "")).strip()
        if not team_name or not color_name:
            print(f"[WARN] Missing team or color on row {idx}. Skipping.")
            continue

        team_folder_name = f"{team_name}-{color_name}"
        team_folder = os.path.join(sport_folder, team_folder_name)

        if not os.path.isdir(team_folder):
            print(f"[WARN] Assets not found for '{team_folder_name}' in {sport_folder}. Skipping.")
            continue

        coords = load_coords_json(team_folder)
        process_front(row, team_folder, coords)
        process_back(row, team_folder, coords)
        front_path = os.path.join(OUTPUT_DIR, f"{row['Name']}-3.png")
        back_path = os.path.join(OUTPUT_DIR, f"{row['Name']}-2.png")
        process_combo(row, front_path, back_path)

        # Youth overlay on all three outputs if needed
        is_youth = str(row.get("Mens or Youth", "")).strip().lower() == "youth"
        if is_youth:
            if youth_overlay_img is None:
                if os.path.exists(youth_overlay_path):
                    youth_overlay_img = Image.open(youth_overlay_path).convert("RGBA")
                else:
                    print(f"[WARN] youth.png not found at {youth_overlay_path}. Skipping youth overlay.")
                    youth_overlay_img = False  # mark as unavailable

            if youth_overlay_img:
                combo_path = os.path.join(OUTPUT_DIR, f"{row['Name']}-1.png")
                # Overlay on top of each result image
                apply_overlay_to_file(front_path, youth_overlay_img)
                apply_overlay_to_file(back_path, youth_overlay_img)
                apply_overlay_to_file(combo_path, youth_overlay_img)

if __name__ == "__main__":
    main()