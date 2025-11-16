"""
Siamese-aware augmentation for SIAM-YOLO
On-the-fly mosaic augmentation that respects query-support pairing
"""

import random
import cv2
import numpy as np
from typing import Dict, List, Any
from ultralytics.data.augment import BaseMixTransform


class SiameseMosaic(BaseMixTransform):
    """
    Siamese-aware Mosaic augmentation for SIAM-YOLO.
    
    Creates 2x2 mosaics from 4 images with random placement (Ultralytics style).
    Only includes bboxes from images that match the base object category.
    This enables hard negative mining while preserving the query-support structure.
    
    Key differences from standard Mosaic:
    - Only adds labels for images matching the base object
    - Preserves query-support image pairing
    - Creates diverse negative samples for better discrimination
    """
    
    def __init__(self, dataset, imgsz: int = 640, p: float = 0.5, distractor_handler=None):
        """
        Initialize Siamese Mosaic augmentation.
        
        Args:
            dataset: The SiamDataset instance
            imgsz: Target image size
            p: Probability of applying mosaic
        """
        super().__init__(dataset=dataset, p=p)
        self.imgsz = imgsz
        self.border = (-imgsz // 2, -imgsz // 2)
        # Optional handler that can apply distractor copy-paste to each source image
        # before it is cropped/stiched into the mosaic. Accepts a transform-like
        # object (callable) such as DistractorCopyPaste.
        self.distractor_handler = distractor_handler
        
    def get_indexes(self, buffer: bool = True) -> List[int]:
        """Return 3 random dataset indexes for mosaic (need 4 total including base)."""
        return [random.randint(0, len(self.dataset) - 1) for _ in range(3)]
    
    def _extract_object_name(self, img_path: str) -> str:
        """Extract object category name from image path."""
        import re
        from pathlib import Path
        
        name = Path(img_path).stem
        
        # Match support images: object_name_support_N
        if '_support_' in name:
            match = re.match(r'(.+)_support_\d+', name)
            if match:
                return match.group(1)
        
        # Match query images: object_name_image_NNNNN
        match = re.match(r'(.+)_image_\d+', name)
        if match:
            return match.group(1)
        
        return name
    
    def _crop_for_mosaic(
        self,
        img: np.ndarray,
        instances,
        target_ratio: float = 0.70
    ):
        """Smart crop around objects for mosaic placement."""
        import os
        debug = os.environ.get('SIAM_DEBUG', '').lower() == 'true'
        
        if len(instances) == 0:
            return img, instances
        
        h, w = img.shape[:2]
        
        # Get bbox bounds - convert to xyxy format if needed
        if instances._bboxes.format != 'xyxy':
            instances._bboxes.convert('xyxy')
        
        bboxes = instances.bboxes  # Now in xyxy format
        if len(bboxes) == 0:
            return img, instances
        
        if debug:
            print(f"[_crop_for_mosaic] Input: img={w}x{h}, bboxes[0]={bboxes[0]}, format={instances._bboxes.format}")
        
        # Denormalize if needed
        if bboxes.max() <= 1.0:
            bboxes_pixel = bboxes.copy()
            bboxes_pixel[:, [0, 2]] *= w
            bboxes_pixel[:, [1, 3]] *= h
        else:
            bboxes_pixel = bboxes
        
        if debug:
            print(f"[_crop_for_mosaic] bboxes_pixel[0]={bboxes_pixel[0]}")
        
        # Find encompassing bbox
        x1_min = bboxes_pixel[:, 0].min()
        y1_min = bboxes_pixel[:, 1].min()
        x2_max = bboxes_pixel[:, 2].max()
        y2_max = bboxes_pixel[:, 3].max()
        
        # Calculate crop size
        bbox_w = x2_max - x1_min
        bbox_h = y2_max - y1_min
        
        crop_w = max(target_ratio * w, bbox_w * 1.5)
        crop_h = max(target_ratio * h, bbox_h * 1.5)
        crop_w = min(crop_w, w)
        crop_h = min(crop_h, h)
        
        # Random crop position that includes all bboxes
        max_x1 = min(x1_min, w - crop_w)
        min_x1 = max(0, x2_max - crop_w)
        max_y1 = min(y1_min, h - crop_h)
        min_y1 = max(0, y2_max - crop_h)
        
        if min_x1 <= max_x1:
            crop_x1 = random.uniform(min_x1, max_x1)
        else:
            crop_x1 = max(0, (x1_min + x2_max - crop_w) / 2)
        
        if min_y1 <= max_y1:
            crop_y1 = random.uniform(min_y1, max_y1)
        else:
            crop_y1 = max(0, (y1_min + y2_max - crop_h) / 2)
        
        crop_x1 = int(max(0, crop_x1))
        crop_y1 = int(max(0, crop_y1))
        crop_x2 = int(min(w, crop_x1 + crop_w))
        crop_y2 = int(min(h, crop_y1 + crop_h))
        
        if debug:
            print(f"[_crop_for_mosaic] Crop region: [{crop_x1}:{crop_x2}, {crop_y1}:{crop_y2}]")
        
        # Crop image
        img_cropped = img[crop_y1:crop_y2, crop_x1:crop_x2]
        
        # Create new instances for cropped image
        from ultralytics.utils.instance import Instances
        
        # Adjust bboxes to pixel coords
        bboxes_adjusted = bboxes_pixel.copy()
        bboxes_adjusted[:, [0, 2]] -= crop_x1
        bboxes_adjusted[:, [1, 3]] -= crop_y1
        
        if debug:
            print(f"[_crop_for_mosaic] After offset: bboxes_adjusted[0]={bboxes_adjusted[0]}")
        
        # Clip to crop bounds - must maintain x1 <= x2 and y1 <= y2
        crop_w = crop_x2 - crop_x1
        crop_h = crop_y2 - crop_y1
        
        # First check: filter out boxes that are completely outside the crop region or have negative dimensions
        # A bbox is invalid if x1 >= crop_w or x2 <= 0 or y1 >= crop_h or y2 <= 0
        # Or if it has negative width/height (which indicates it's outside the crop)
        bbox_widths_before = bboxes_adjusted[:, 2] - bboxes_adjusted[:, 0]
        bbox_heights_before = bboxes_adjusted[:, 3] - bboxes_adjusted[:, 1]
        
        # Keep only boxes that: have positive dimensions AND have some overlap with crop region
        valid_mask_before_clip = (bbox_widths_before > 0) & (bbox_heights_before > 0) & \
                                 (bboxes_adjusted[:, 0] < crop_w) & (bboxes_adjusted[:, 2] > 0) & \
                                 (bboxes_adjusted[:, 1] < crop_h) & (bboxes_adjusted[:, 3] > 0)
        
        bboxes_adjusted = bboxes_adjusted[valid_mask_before_clip]
        
        if debug:
            print(f"[_crop_for_mosaic] Valid boxes before clip: {valid_mask_before_clip.sum()}/{len(valid_mask_before_clip)}")
        
        if len(bboxes_adjusted) == 0:
            if debug:
                print(f"[_crop_for_mosaic] After clip: all bboxes filtered out (outside crop region), crop_size={crop_w}x{crop_h}")
            # Return empty instances
            instances_cropped = Instances(
                np.zeros((0, 4), dtype=np.float32),
                bbox_format='xyxy',
                normalized=True
            )
            return img_cropped, instances_cropped
        
        # Clip all corners to crop bounds
        bboxes_adjusted[:, 0] = np.clip(bboxes_adjusted[:, 0], 0, crop_w)  # x1
        bboxes_adjusted[:, 1] = np.clip(bboxes_adjusted[:, 1], 0, crop_h)  # y1
        bboxes_adjusted[:, 2] = np.clip(bboxes_adjusted[:, 2], 0, crop_w)  # x2
        bboxes_adjusted[:, 3] = np.clip(bboxes_adjusted[:, 3], 0, crop_h)  # y2
        
        # Ensure x2 >= x1 and y2 >= y1
        bboxes_adjusted[:, 0] = np.minimum(bboxes_adjusted[:, 0], bboxes_adjusted[:, 2])
        bboxes_adjusted[:, 1] = np.minimum(bboxes_adjusted[:, 1], bboxes_adjusted[:, 3])
        
        # Filter again after clipping to ensure minimum size
        bbox_widths = bboxes_adjusted[:, 2] - bboxes_adjusted[:, 0]
        bbox_heights = bboxes_adjusted[:, 3] - bboxes_adjusted[:, 1]
        valid_mask_after_clip = (bbox_widths >= 1.0) & (bbox_heights >= 1.0)
        
        # Store indices of originally valid boxes for segment filtering later
        original_valid_indices = np.where(valid_mask_before_clip)[0]
        final_valid_indices = original_valid_indices[valid_mask_after_clip]
        
        bboxes_adjusted = bboxes_adjusted[valid_mask_after_clip]
        
        if debug:
            if len(bboxes_adjusted) > 0:
                print(f"[_crop_for_mosaic] After clip: bboxes_adjusted[0]={bboxes_adjusted[0]}, crop_size={crop_w}x{crop_h}")
            else:
                print(f"[_crop_for_mosaic] After clip: all bboxes filtered out (too small), crop_size={crop_w}x{crop_h}")
        
        # Normalize
        if len(bboxes_adjusted) > 0:
            bboxes_adjusted[:, [0, 2]] /= crop_w
            bboxes_adjusted[:, [1, 3]] /= crop_h
        
        if debug:
            if len(bboxes_adjusted) > 0:
                print(f"[_crop_for_mosaic] After normalize: bboxes_adjusted[0]={bboxes_adjusted[0]}")
            else:
                print(f"[_crop_for_mosaic] After normalize: no valid bboxes")
        
        # Create new Instances object
        instances_cropped = Instances(
            bboxes_adjusted.astype(np.float32),
            bbox_format='xyxy',
            normalized=True
        )
        
        # Handle segments if they exist - also filter them with the same mask
        if hasattr(instances, 'segments') and instances.segments is not None and len(final_valid_indices) > 0:
            # Adjust segments similarly and apply the same filter mask
            segments_all = []
            for i, seg in enumerate(instances.segments):
                if seg is not None and len(seg) > 0:
                    seg_adj = seg.copy()
                    seg_adj[:, 0] = (seg_adj[:, 0] * w - crop_x1) / (crop_x2 - crop_x1)
                    seg_adj[:, 1] = (seg_adj[:, 1] * h - crop_y1) / (crop_y2 - crop_y1)
                    segments_all.append(seg_adj)
                else:
                    segments_all.append([])
            # Apply the same filter - keep only segments corresponding to final_valid_indices
            if len(segments_all) > 0:
                segments_adjusted = [segments_all[i] for i in final_valid_indices if i < len(segments_all)]
                instances_cropped.segments = segments_adjusted
        
        # Copy other attributes if they exist
        if hasattr(instances, 'keypoints') and instances.keypoints is not None:
            instances_cropped.keypoints = instances.keypoints
        
        return img_cropped, instances_cropped
    
    def _mix_transform(self, labels: Dict[str, Any]) -> Dict[str, Any]:
        """
        Apply Siamese-aware mosaic to create 2x2 grid with hard negative mining.
        
        Args:
            labels: Base image labels dict
            
        Returns:
            Augmented labels dict with mosaic image
        """
        import os
        debug = os.environ.get('SIAM_DEBUG', '').lower() == 'true'
        
        assert len(labels.get("mix_labels", [])) >= 3, "Need 3 additional images for mosaic"
        
        if debug:
            print(f"[SiameseMosaic] Starting _mix_transform")
            print(f"  - Base instances: {len(labels.get('instances', []))}")
            print(f"  - Mix labels: {len(labels.get('mix_labels', []))}")
        
        # Extract base object name
        base_img_path = str(labels.get("im_file", ""))
        base_object = self._extract_object_name(base_img_path)
        
        s = self.imgsz
        mosaic4 = np.full((s * 2, s * 2, 3), 114, dtype=np.uint8)
        
        # Random mosaic center
        yc = int(random.uniform(s * 0.5, s * 1.5))
        xc = int(random.uniform(s * 0.5, s * 1.5))
        
        # Collect all 4 images with their object names
        all_labels = [labels] + labels["mix_labels"][:3]
        all_bboxes_mosaic = []
        
        for i, labels_patch in enumerate(all_labels):
            # Apply distractor handler directly to the source patch BEFORE any
            # cropping/stitching. Using the handler as a callable respects its own
            # probability `p` and internal logic (alpha blending, collision checks).
            if getattr(self, 'distractor_handler', None) is not None:
                try:
                    labels_patch = self.distractor_handler(labels_patch)
                except Exception:
                    # Fail-safe: if distractor handler errors, continue with original patch
                    pass
            # Get image and instances
            img = labels_patch["img"]
            instances = labels_patch["instances"]
            cls_labels = labels_patch.get("cls", np.array([]))
            h, w = img.shape[:2]
            
            if debug:
                print(f"[SiameseMosaic] Processing patch {i}: instances={len(instances)}, cls={len(cls_labels)}")
            
            # Crop for mosaic - further increased to 0.98 to minimize resizing
            img_cropped, instances_cropped = self._crop_for_mosaic(
                img, instances, target_ratio=0.98
            )
            h_crop, w_crop = img_cropped.shape[:2]
            
            if debug:
                print(f"[SiameseMosaic] After crop patch {i}: instances_cropped={len(instances_cropped)}")
            
            # Determine if same object
            patch_img_path = str(labels_patch.get("im_file", ""))
            patch_object = self._extract_object_name(patch_img_path)
            is_same_object = (patch_object == base_object) or i == 0  # Always include base
            
            if debug:
                print(f"[SiameseMosaic] Patch {i}: is_same_object={is_same_object}, patch_object={patch_object}, base_object={base_object}")
            
            # Calculate placement (Ultralytics style)
            if i == 0:  # top left
                x1a, y1a, x2a, y2a = max(xc - w_crop, 0), max(yc - h_crop, 0), xc, yc
                x1b, y1b, x2b, y2b = w_crop - (x2a - x1a), h_crop - (y2a - y1a), w_crop, h_crop
            elif i == 1:  # top right
                x1a, y1a, x2a, y2a = xc, max(yc - h_crop, 0), min(xc + w_crop, s * 2), yc
                x1b, y1b, x2b, y2b = 0, h_crop - (y2a - y1a), min(w_crop, x2a - x1a), h_crop
            elif i == 2:  # bottom left
                x1a, y1a, x2a, y2a = max(xc - w_crop, 0), yc, xc, min(s * 2, yc + h_crop)
                x1b, y1b, x2b, y2b = w_crop - (x2a - x1a), 0, w_crop, min(y2a - y1a, h_crop)
            else:  # bottom right
                x1a, y1a, x2a, y2a = xc, yc, min(xc + w_crop, s * 2), min(s * 2, yc + h_crop)
                x1b, y1b, x2b, y2b = 0, 0, min(w_crop, x2a - x1a), min(y2a - y1a, h_crop)
            
            # Place image
            mosaic4[y1a:y2a, x1a:x2a] = img_cropped[y1b:y2b, x1b:x2b]
            
            if debug:
                print(f"[SiameseMosaic] Patch {i}: is_same_object={is_same_object}, len(instances_cropped)={len(instances_cropped)}")
            
            # Adjust bboxes only if same object
            if is_same_object and len(instances_cropped) > 0:
                # Get normalized bboxes from cropped instances
                bboxes = instances_cropped.bboxes.copy()
                
                if debug and i == 0:
                    print(f"[SiameseMosaic] Patch {i} bbox transform:")
                    print(f"  - Original bbox (normalized in cropped): {bboxes[0]}")
                    print(f"  - Cropped image size: w={w_crop}, h={h_crop}")
                    print(f"  - Placement: mosaic[{y1a}:{y2a}, {x1a}:{x2a}] = crop[{y1b}:{y2b}, {x1b}:{x2b}]")
                
                # Convert to pixel coords in cropped image
                bboxes[:, [0, 2]] *= w_crop
                bboxes[:, [1, 3]] *= h_crop
                
                if debug and i == 0:
                    print(f"  - After scaling to crop pixels: {bboxes[0]}")
                
                # The bboxes are in the coordinate system of img_cropped (0 to w_crop, 0 to h_crop)
                # But we're only using img_cropped[y1b:y2b, x1b:x2b] which goes to mosaic[y1a:y2a, x1a:x2a]
                # So we need to:
                # 1. Subtract (x1b, y1b) to get coords relative to the used portion
                # 2. Add (x1a, y1a) to place in mosaic
                
                bboxes[:, [0, 2]] -= x1b
                bboxes[:, [1, 3]] -= y1b
                
                if debug and i == 0:
                    print(f"  - After subtracting crop offset ({x1b}, {y1b}): {bboxes[0]}")
                
                bboxes[:, [0, 2]] += x1a
                bboxes[:, [1, 3]] += y1a
                
                if debug and i == 0:
                    print(f"  - After adding mosaic offset ({x1a}, {y1a}): {bboxes[0]}")
                
                # Normalize to 2s x 2s mosaic
                bboxes[:, [0, 2]] /= (s * 2)
                bboxes[:, [1, 3]] /= (s * 2)
                
                if debug and i == 0:
                    print(f"  - After normalizing to 2s grid (size={s*2}): {bboxes[0]}")
                
                # Store for later concatenation
                for j, bbox in enumerate(bboxes):
                    # Get class label - handle case where cls might be shorter than instances
                    if j < len(cls_labels):
                        cls_val = cls_labels[j]
                    else:
                        cls_val = 0  # Default class
                    
                    all_bboxes_mosaic.append({
                        'bbox': bbox,
                        'cls': cls_val,
                        'segment': instances_cropped.segments[j] if hasattr(instances_cropped, 'segments') and instances_cropped.segments is not None and j < len(instances_cropped.segments) else [],
                        'format': instances_cropped._bboxes.format  # Store the bbox format
                    })
        

        # --- Ensure main object is inside crop ---
        # Find the first bbox (main object)
        crop_top = s // 2
        crop_left = s // 2
        if len(all_bboxes_mosaic) > 0:
            # Use the first bbox (main object)
            main_bbox = all_bboxes_mosaic[0]['bbox'].copy()
            bbox_fmt = all_bboxes_mosaic[0].get('format', 'xyxy')
            # Convert normalized bbox to pixel coordinates in 2s x 2s mosaic
            if bbox_fmt == 'xyxy':
                main_bbox[[0, 2]] *= (s * 2)
                main_bbox[[1, 3]] *= (s * 2)
                x1, y1, x2, y2 = main_bbox
            elif bbox_fmt == 'xywh':
                main_bbox *= (s * 2)
                cx, cy, w, h = main_bbox
                x1 = cx - w / 2
                y1 = cy - h / 2
                x2 = cx + w / 2
                y2 = cy + h / 2
            else:
                x1 = y1 = x2 = y2 = s
            # Adjust crop so bbox is inside
            crop_left = int(np.clip((x1 + x2) / 2 - s / 2, 0, s))
            crop_top = int(np.clip((y1 + y2) / 2 - s / 2, 0, s))
        mosaic4 = mosaic4[crop_top:crop_top + s, crop_left:crop_left + s]

        if debug:
            print(f"[SiameseMosaic] After placement: {len(all_bboxes_mosaic)} bboxes collected")
            if len(all_bboxes_mosaic) > 0:
                print(f"[SiameseMosaic]   - First bbox (normalized in 2s grid): {all_bboxes_mosaic[0]['bbox']}")
                print(f"[SiameseMosaic]   - Will crop 2s grid ({s*2}x{s*2}) from [{crop_top}:{crop_top+s}, {crop_left}:{crop_left+s}] to final {s}x{s}")
        
        # Adjust all bboxes for center crop
        final_bboxes = []
        final_cls = []
        final_segments = []
        
        for bbox_idx, bbox_data in enumerate(all_bboxes_mosaic):
            bbox = bbox_data['bbox'].copy()
            bbox_fmt = bbox_data.get('format', 'xyxy')  # Default to xyxy
            
            if debug and bbox_idx == 0:
                print(f"[SiameseMosaic] Processing final bbox:")
                print(f"  - Input (normalized in 2s grid): {bbox}")
            
            # Convert normalized bbox to pixel coordinates in the 2s x 2s mosaic
            if bbox_fmt == 'xyxy':
                # xyxy: x1, y1, x2, y2
                bbox[[0, 2]] *= (s * 2)
                bbox[[1, 3]] *= (s * 2)
            elif bbox_fmt == 'xywh':
                # xywh: center_x, center_y, width, height
                # First scale all components to pixel coords in 2s x 2s grid
                bbox *= (s * 2)
            else:
                raise ValueError(f"Unknown bbox format: {bbox_fmt}")
            
            if debug and bbox_idx == 0:
                print(f"  - After scaling to 2s pixels: {bbox}")
            
            # Adjust for crop (only x1/y1 for xyxy, or center for xywh)
            if bbox_fmt == 'xyxy':
                bbox[[0, 2]] -= crop_left
                bbox[[1, 3]] -= crop_top
                
                if debug and bbox_idx == 0:
                    print(f"  - After subtracting crop offset ({crop_left}, {crop_top}): {bbox}")
                
                # Clip to final image bounds
                bbox[[0, 2]] = np.clip(bbox[[0, 2]], 0, s)
                bbox[[1, 3]] = np.clip(bbox[[1, 3]], 0, s)
                
                if debug and bbox_idx == 0:
                    print(f"  - After clipping to final bounds [0:{s}, 0:{s}]: {bbox}")
                
                # Check if still visible (at least 0.5 pixel in each dimension)
                w_box = bbox[2] - bbox[0]
                h_box = bbox[3] - bbox[1]
                
                if debug and bbox_idx == 0:
                    print(f"  - Final size: w={w_box:.2f}, h={h_box:.2f}, threshold=0.5")
                
            else:  # xywh
                # For xywh, adjust center coordinates and clip box region
                center_x = bbox[0]
                center_y = bbox[1]
                width = bbox[2]
                height = bbox[3]
                
                # Adjust center
                center_x -= crop_left
                center_y -= crop_top
                
                # Clip center to final bounds, and adjust for out-of-bounds centers
                x1 = center_x - width / 2
                y1 = center_y - height / 2
                x2 = center_x + width / 2
                y2 = center_y + height / 2
                
                # Clip bounding box to image
                x1 = np.clip(x1, 0, s)
                y1 = np.clip(y1, 0, s)
                x2 = np.clip(x2, 0, s)
                y2 = np.clip(y2, 0, s)
                
                # Recompute from clipped corners
                w_box = x2 - x1
                h_box = y2 - y1
                
                # Recompute center and dimensions
                bbox[0] = x1 + w_box / 2  # center_x
                bbox[1] = y1 + h_box / 2  # center_y
                bbox[2] = w_box
                bbox[3] = h_box
            
            if debug and bbox_idx == 0:
                print(f"[SiameseMosaic] First bbox check: w={w_box:.2f}, h={h_box:.2f}, threshold=0.5")
            
            if w_box >= 0.5 and h_box >= 0.5:
                # Normalize to final size
                if bbox_fmt == 'xyxy':
                    bbox[[0, 2]] /= s
                    bbox[[1, 3]] /= s
                else:  # xywh
                    bbox[:] /= s
                
                final_bboxes.append(bbox)
                final_cls.append(bbox_data['cls'])
                final_segments.append(bbox_data['segment'])
            elif debug and bbox_idx == 0:
                print(f"[SiameseMosaic] ✗ First bbox filtered out!")
        
        # Update labels
        if len(final_bboxes) > 0:
            from ultralytics.utils.instance import Instances
            
            final_bboxes = np.array(final_bboxes, dtype=np.float32)
            final_cls = np.array(final_cls, dtype=np.float32)
            if final_cls.ndim == 1:
                final_cls = final_cls.reshape(-1, 1)  # Ensure shape is (N, 1)
            
            if debug:
                print(f"[SiameseMosaic] Final result: {len(final_bboxes)} bboxes")
                print(f"  - bboxes shape: {final_bboxes.shape}, dtype: {final_bboxes.dtype}")
                print(f"  - cls shape: {final_cls.shape}, dtype: {final_cls.dtype}")
            
            # Bboxes are already in normalized xyxy format, create instances properly
            # Keep them in xyxy normalized format - this is what Ultralytics expects after augmentation
            instances_new = Instances(
                final_bboxes,
                bbox_format='xyxy',
                normalized=True  # Explicitly set normalized=True since bboxes are normalized
            )
            
            labels["instances"] = instances_new
            labels["cls"] = final_cls
            
            if final_segments and any(len(seg) > 0 for seg in final_segments):
                instances_new.segments = final_segments
        else:
            if debug:
                print(f"[SiameseMosaic] ✗ NO bboxes in final result!")
            # No valid instances after mosaic
            from ultralytics.utils.instance import Instances
            labels["instances"] = Instances(
                np.zeros((0, 4), dtype=np.float32),
                bbox_format='xyxy',
                normalized=True
            )
            labels["cls"] = np.array([], dtype=np.float32).reshape(0, 1)
        
        labels["img"] = mosaic4
        labels.pop("mix_labels", None)
        
        return labels


class DistractorCopyPaste(BaseMixTransform):
    """
    Paste small distractor objects (transparent RGBA images) onto training images
    without adding labels. Distractors are preloaded from a support folder whose
    filenames begin with `dis_`.

    Behavior:
    - Preloads all `dis_*` images with alpha during initialization.
    - On each call to `_mix_transform`, with probability `p` pastes between
      `min_paste` and `max_paste` distractors scaled to a random height in
      `size_range` (fraction of image height).
    - Ensures pasted distractors do NOT overlap any ground truth boxes.
    - Blends using the alpha channel: out = a*fg + (1-a)*bg
    - Does not modify `labels['instances']` or add boxes.
    """

    def __init__(
        self,
        dataset,
        folder: str = "/mlcv2/WorkingSpace/Personal/nguyenmv/ZaloAI/Repo/maibel/ZaloAIC2025/copy_paste/support",
        p: float = 0.5,
        min_paste: int = 1,
        max_paste: int = 3,
        size_range: tuple[float, float] = (0.05, 0.10),
        max_retries: int = 10,
    ):
        super().__init__(dataset=dataset, p=p)
        from pathlib import Path

        self.folder = Path(folder)
        self.min_paste = int(min_paste)
        self.max_paste = int(max_paste)
        self.size_range = size_range
        self.max_retries = int(max_retries)

        # Preload distractor images (RGBA) whose filename starts with 'dis_'
        self.distractors: List[np.ndarray] = []
        if self.folder.exists() and self.folder.is_dir():
            for pth in sorted(self.folder.glob("dis_*")):
                try:
                    img = cv2.imread(str(pth), cv2.IMREAD_UNCHANGED)  # Load with alpha if present (BGRA)
                    if img is None:
                        continue
                    # Keep the OpenCV channel order (B,G,R,Alpha) to match pipeline images (BGR)
                    # If image has no alpha channel, synthesize a fully-opaque alpha channel.
                    if img.ndim == 2:
                        # grayscale -> convert to BGR then add alpha
                        img = cv2.cvtColor(img, cv2.COLOR_GRAY2BGRA)
                    elif img.ndim == 3 and img.shape[2] == 3:
                        # BGR -> add alpha channel
                        alpha_ch = np.full((img.shape[0], img.shape[1], 1), 255, dtype=np.uint8)
                        img = np.concatenate([img, alpha_ch], axis=2)
                    elif img.ndim == 3 and img.shape[2] == 4:
                        # already BGRA, keep as-is
                        pass
                    else:
                        # unexpected shape, skip
                        continue

                    # store BGRA image (OpenCV order) so blending uses same BGR channel ordering
                    self.distractors.append(img)
                except Exception:
                    continue

        if len(self.distractors) == 0:
            import warnings

            warnings.warn(f"DistractorCopyPaste: no 'dis_*' RGBA images found in {self.folder}")

    def __call__(self, labels: Dict[str, Any]) -> Dict[str, Any]:
        """
        Custom __call__ to allow applying distractors even when `cls` is empty.

        BaseMixTransform.__call__ skips mixing when `labels['cls']` is empty (useful
        for MixUp/Mosaic). DistractorCopyPaste is intended to add background noise
        even to images without target boxes, so we bypass that check here.
        """
        # Probability check
        if random.uniform(0, 1) > self.p:
            return labels

        # Directly call _mix_transform; we don't need additional image indexes
        try:
            return self._mix_transform(labels)
        except Exception:
            # On any unexpected error, return original labels to avoid breaking pipeline
            return labels

    @staticmethod
    def _intersects(box_a, box_b) -> bool:
        """Simple intersection test for normalized xyxy boxes.

        box: [x1, y1, x2, y2]
        Returns True if intersection area > 0
        """
        # Works for either normalized (0..1) or absolute pixel coords as long as
        # both boxes are in the same coordinate space. After refactor we use
        # absolute pixel coordinates for both candidate and GT boxes.
        x1 = max(box_a[0], box_b[0])
        y1 = max(box_a[1], box_b[1])
        x2 = min(box_a[2], box_b[2])
        y2 = min(box_a[3], box_b[3])
        return (x2 > x1) and (y2 > y1)

    def _mix_transform(self, labels: Dict[str, Any]) -> Dict[str, Any]:
        """Apply distractor copy-paste to the provided labels dict.

        Expects `labels['img']` to be a HxWx3 uint8 RGB image and
        `labels['instances']` to be an Instances object with normalized bboxes.
        """
        import os
        debug = os.environ.get('SIAM_DEBUG', '').lower() == 'true'

        if not self.distractors:
            if debug:
                print("[DistractorCopyPaste] no distractors loaded, skipping")
            return labels

        img = labels.get("img")
        if img is None:
            return labels

        h_img, w_img = img.shape[:2]

        n_to_paste = random.randint(self.min_paste, self.max_paste)

        # Ensure instances are in xyxy normalized coordinates
        instances = labels.get("instances")
        gt_boxes = np.zeros((0, 4), dtype=np.float32)
        if instances is not None:
            try:
                # Ensure bbox format is xyxy for intersection checks; this mutates the Instances' internal
                # Bboxes format, which is ok since format conversion is a lightweight operation.
                if instances._bboxes.format != 'xyxy':
                    instances._bboxes.convert('xyxy')

                # Copy GT boxes so we don't modify the original Instances object
                gt_boxes = instances.bboxes.copy()

                # Convert GT boxes to absolute pixel coordinates if they are normalized.
                # We avoid mutating `instances.normalized` or the Instances object itself by working on a copy.
                if getattr(instances, 'normalized', False):
                    # gt_boxes are in [x1, y1, x2, y2] normalized (0..1); convert to pixels
                    gt_boxes = gt_boxes.astype(np.float32).copy()
                    gt_boxes[:, [0, 2]] *= float(w_img)
                    gt_boxes[:, [1, 3]] *= float(h_img)
                else:
                    # Already in absolute pixel coordinates
                    gt_boxes = gt_boxes.astype(np.float32).copy()
            except Exception:
                gt_boxes = np.zeros((0, 4), dtype=np.float32)

        for _ in range(n_to_paste):
            # choose a distractor image
            dis = random.choice(self.distractors)
            dh, dw = dis.shape[:2]
            # Skip invalid images
            if dh == 0 or dw == 0:
                continue

            # choose target height as fraction of image height
            frac = random.uniform(self.size_range[0], self.size_range[1])
            target_h = max(1, int(round(frac * h_img)))
            scale = target_h / float(dh)
            target_w = max(1, int(round(dw * scale)))

            # Resize distractor (RGBA)
            try:
                dis_resized = cv2.resize(dis, (target_w, target_h), interpolation=cv2.INTER_AREA)
            except Exception:
                continue

            # Prepare array view. Note: distractors are kept in OpenCV channel order (B,G,R,A)
            fg_rgb = dis_resized[..., :3].astype(np.float32)  # BGR ordering to match bg
            alpha = dis_resized[..., 3].astype(np.float32) / 255.0
            alpha = np.expand_dims(alpha, axis=2)

            placed = False
            retries = 0
            while not placed and retries < self.max_retries:
                retries += 1
                # Random top-left such that the paste fits
                x1 = random.randint(0, max(0, w_img - target_w))
                y1 = random.randint(0, max(0, h_img - target_h))
                x2 = x1 + target_w
                y2 = y1 + target_h

                # Build candidate box in absolute pixel coordinates to match gt_boxes
                cand_box = np.array([x1, y1, x2, y2], dtype=np.float32)

                # Check collision with any GT box (both in pixel coords)
                collision = False
                for gt in gt_boxes:
                    if self._intersects(cand_box, gt):
                        collision = True
                        break

                if collision:
                    if debug:
                        print(f"[DistractorCopyPaste] collision, retry {retries}")
                    continue

                # Paste: blend fg into background region
                bg_patch = img[y1:y2, x1:x2]

                # If pipeline images are float tensors or normalized, attempt to convert to uint8 BGR for blending
                if not (isinstance(bg_patch, np.ndarray) and bg_patch.dtype == np.uint8):
                    try:
                        # handle torch tensors or float arrays in 0..1 range
                        import torch

                        if isinstance(bg_patch, torch.Tensor):
                            bg_patch = bg_patch.cpu().numpy()
                        if bg_patch.dtype != np.uint8:
                            # assume float in 0..1
                            bg_patch = (bg_patch * 255.0).astype(np.uint8)
                    except Exception:
                        # fallback: convert to uint8 without scaling
                        bg_patch = bg_patch.astype(np.uint8)

                bg_patch = bg_patch.astype(np.float32)
                if bg_patch.shape[0] != fg_rgb.shape[0] or bg_patch.shape[1] != fg_rgb.shape[1]:
                    # shape mismatch, skip
                    continue

                # fg_rgb is in BGR channel order (we preserved OpenCV ordering on load)
                out = alpha * fg_rgb + (1.0 - alpha) * bg_patch
                out = np.clip(out, 0, 255).astype(np.uint8)
                img[y1:y2, x1:x2] = out
                placed = True

            if debug and not placed:
                print("[DistractorCopyPaste] failed to place distractor after retries")

        labels["img"] = img
        return labels
