# X-Ray Image Validation System

## Overview

The TB Detection system now includes a **robust, multi-heuristic X-ray validation system** that reliably differentiates chest X-ray images from non-medical photos.

## Problem Statement

### Previous Issues

1. **Rejected valid X-rays with color tint** - Images with blueish, yellowish, or greenish tints (from DICOM conversions or institutional systems) were incorrectly rejected due to colorfulness thresholds.
2. **Accepted grayscale non-medical images** - Photos converted to grayscale (e.g., apple, animals, documents) were incorrectly accepted as valid X-rays.
3. **Over-reliance on simple color detection** - Colorfulness metrics alone cannot differentiate medical from non-medical images.

## Solution: Multi-Heuristic Validation

The new `is_xray_image()` function uses **5 sophisticated feature checks** to validate images:

### Feature 1: Intensity Distribution Analysis

**What it checks:** Whether pixel intensity values fall within typical X-ray ranges.

- **X-ray characteristic:** Mean intensity 30-220, Standard deviation ≥15
- **Why it works:** X-rays have moderate brightness and good contrast. Very dark/bright or uniform images are rejected.
- **Score:** 1.5 points if passed

### Feature 2: Texture Analysis (Local Binary Pattern - LBP)

**What it checks:** Local texture patterns characteristic of X-ray images.

- **X-ray characteristic:** LBP entropy between 3.5-7.5
  - X-rays have fine, structured texture (organs, ribs, cardiac silhouette)
  - Natural photos have chaotic, varied texture patterns
- **Why it works:** Medical images have different texture signatures than photographs of objects.
- **Score:** 1.5 points if passed
- **Dependency:** Requires `scikit-image` library

### Feature 3: Frequency Domain Analysis (FFT)

**What it checks:** How image energy is distributed across frequency bands.

- **X-ray characteristic:** Balanced ratio of high to low frequencies (0.3-2.5)
- **Why it works:**
  - X-rays: Fine anatomical details + smooth background = balanced spectrum
  - Natural photos: Either high-frequency clutter (leaves, textures) or very different patterns
- **Score:** 1.5 points if passed

### Feature 4: Medical Structure Detection

**What it checks:** Presence of anatomical structures typical of chest X-rays.

- **X-ray characteristic:**
  - **Vertical symmetry:** >0.50 (left-right symmetry of chest)
  - **Horizontal edge density:** >0.01 (mediastinum, cardiac silhouette)
- **Why it works:** Chest X-rays are naturally symmetric (bilateral anatomy) with strong horizontal structures. Random photos lack these properties.
- **Score:** 1.5 points if passed

### Feature 5: Edge and Contrast Analysis

**What it checks:** Edge density within expected range for X-rays.

- **X-ray characteristic:** Edge ratio between 0.005-0.20
  - Blank/simple images: <0.003
  - X-rays: 0.005-0.20 (anatomical edges, rib details)
  - Detailed natural photos: >0.25
- **Why it works:** Different image types have characteristic edge densities.
- **Score:** 1.5 points if passed

## Scoring System

- **Total Score:** Sum of points across all features (max 7.5 points)
- **Confidence:** Score / Total (0-100%)
- **Decision Threshold:** ≥60% confidence → Accept image as likely X-ray

Each failed feature contributes a specific rejection reason logged to the user.

## Implementation Details

### Function Signature

```python
is_xray_image(img_color) -> (bool, str, float)
```

**Parameters:**

- `img_color` (numpy.ndarray): Color image (BGR format from OpenCV)

**Returns:**

- `bool`: True if image passes validation, False otherwise
- `str`: Detailed message explaining acceptance or rejection reasons
- `float`: Confidence score (0.0-1.0)

### Example Usage

```python
is_valid, message, confidence = is_xray_image(img_bgr)
if is_valid:
    print(f"Accepted X-ray (confidence: {confidence:.1%})")
    # Process image with CNN model
else:
    print(f"Rejected: {message}")
    # Show error to user
```

## Advantages Over Previous Approach

| Aspect                  | Old Method  | New Method            |
| ----------------------- | ----------- | --------------------- |
| **Colored X-rays**      | ❌ Rejected | ✅ Accepted           |
| **Grayscale photos**    | ❌ Accepted | ✅ Rejected           |
| **Texture analysis**    | ❌ None     | ✅ LBP entropy        |
| **Structure detection** | ❌ None     | ✅ Symmetry + anatomy |
| **Frequency analysis**  | ❌ None     | ✅ FFT spectrum       |
| **Logging**             | ⚠ Generic   | ✅ Detailed reasons   |
| **Robustness**          | Low         | High                  |

## Testing Recommendations

### Test Cases

1. **Real Chest X-rays (should ACCEPT)**

   - Standard PA (posteroanterior) view
   - Lateral view
   - X-rays with color tint (blue, yellow institutional processing)
   - Low-contrast X-rays
   - High-contrast X-rays

2. **Grayscale Non-Medical (should REJECT)**

   - Apple photo (grayscale)
   - Landscape (grayscale)
   - Document scan (grayscale)
   - Simple shapes/geometric patterns

3. **Color Non-Medical (should REJECT)**

   - Fruit/food photos
   - Nature scenes
   - Objects
   - Selfies

4. **Edge Cases**
   - Very small/blurry X-rays
   - Rotated/skewed X-rays
   - Partially visible chest X-rays
   - Chest CT images (may be rejected - designed for X-rays)

## Tuning Parameters

If validation is too strict or too lenient, adjust these thresholds in `is_xray_image()`:

### Intensity Distribution

```python
if 30 <= mean_val <= 220 and std_dev >= 15:  # Adjust ranges
```

### LBP Entropy

```python
if 3.5 <= lbp_entropy <= 7.5:  # Adjust entropy bounds
```

### Frequency Ratio

```python
if 0.3 <= freq_ratio <= 2.5:  # Adjust frequency balance
```

### Medical Structure

```python
if symmetry > 0.5 and h_edge_density > 0.01:  # Adjust thresholds
```

### Edge Density

```python
if 0.005 <= edge_ratio <= 0.20:  # Adjust edge ratio bounds
```

### Final Confidence Threshold

```python
if confidence >= 0.60:  # Adjust acceptance threshold (0.50-0.70 typical)
```

## Computational Cost

- **Processing time:** ~200-500ms per image on typical CPU (224x224 resized)
- **Memory:** ~50-100MB for FFT operations
- **Bottleneck:** FFT frequency domain analysis (can be optimized)

## Dependencies

- `opencv-python`: Image processing (Canny, resize, etc.)
- `numpy`: Numerical operations (FFT, statistics)
- `scikit-image`: Texture analysis (LBP)

All dependencies listed in `requirements.txt`.

## Future Improvements

1. **Deep Learning Classifier** - Fine-tune a lightweight model (MobileNet) on X-ray vs. non-X-ray dataset for even higher accuracy
2. **DICOM Support** - Detect DICOM metadata directly if file contains it
3. **Modality Detection** - Distinguish X-ray from CT, ultrasound, MRI, etc.
4. **Cached Validation** - Cache validation results for repeated uploads
5. **Interactive Feedback** - Show validation heatmaps to users for debugging

## Error Messages

Users see these messages when images are rejected:

```
❌ This image does not appear to be a chest X-ray. [REASON]. Please upload a chest X-ray image (JPG or PNG).
```

Possible reasons:

- "Insufficient contrast (likely uniform image)"
- "Unusual brightness levels for X-ray"
- "Image has too much textural complexity (likely natural photo)"
- "Image has too much high-frequency detail (likely natural photo)"
- "No anatomical structure detected (not a medical image)"
- "Excessive edge density (likely detailed natural photo)"
- "Insufficient edges (likely blank or very simple image)"

## References

- Hasler, D., & Süsstrunk, S. E. (2003). Measuring colorfulness in natural images. _IS&T/SPIE Electronic Imaging_
- Ojala, T., Pietikäinen, M., & Mäenpää, T. (2002). Multiresolution gray-scale and rotation invariant texture classification with local binary patterns. _IEEE TPAMI_
- FFT-based texture analysis techniques in medical imaging

---

**Last Updated:** November 17, 2025
**Version:** 2.0 (Multi-heuristic validation)
