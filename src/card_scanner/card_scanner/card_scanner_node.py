#!/usr/bin/env python3
"""
Live card scanner using Snehil's detection logic.
Shows live feed with green boxes around detected cards.
"""

import rclpy
from rclpy.node import Node
from sensor_msgs.msg import Image
from cv_bridge import CvBridge

import cv2
import numpy as np
import threading
import time


class CardDetector:
    """
    White cards on dark carpet — uses Otsu's automatic threshold.
    Otsu finds the optimal split between the dark floor and bright card
    pixel distributions without any hardcoded brightness value.

    Pipeline:
      1. Grayscale + Gaussian blur  — remove carpet texture noise (low-pass)
      2. Otsu threshold             — auto-separate bright card / dark floor
      3. Morphological close        — fill small holes inside the card blob
      4. Morphological open         — remove tiny noise specks
      5. Contour filter             — area, aspect ratio, solidity
    """

    MIN_CARD_AREA      = 300
    MAX_CARD_AREA_FRAC = 0.50    # up to half the frame (very close card)
    MIN_SOLIDITY       = 0.35
    MIN_ASPECT         = 0.08    # very flat card seen nearly edge-on
    MAX_ASPECT         = 0.98

    # Fraction of frame height from the bottom used as the scan ROI
    SCAN_ROI_FRAC = 0.35

    def _build_mask(self, frame):
        gray    = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        # Gaussian blur — low-pass filter suppresses carpet grain noise
        blurred = cv2.GaussianBlur(gray, (7, 7), 0)
        # Otsu automatically finds the threshold between dark floor & white card
        _, mask = cv2.threshold(
            blurred, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        # Closing — dilation then erosion — fills gaps inside the card blob
        k    = cv2.getStructuringElement(cv2.MORPH_RECT, (9, 9))
        mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, k, iterations=2)
        # Opening — erosion then dilation — removes small noise specks
        mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN,  k, iterations=1)
        return mask

    def get_scan_roi(self, frame):
        """Returns (roi_y, roi_h) — the bottom strip used for scanning."""
        H = frame.shape[0]
        roi_h = int(H * self.SCAN_ROI_FRAC)
        roi_y = H - roi_h
        return roi_y, roi_h

    def find_cards(self, frame):
        """Returns list of (x, y, w, h) sorted left to right, within bottom ROI only."""
        H, W   = frame.shape[:2]
        roi_y, roi_h = self.get_scan_roi(frame)

        # Crop to bottom ROI for detection
        roi_frame = frame[roi_y:roi_y + roi_h, 0:W]
        mask      = self._build_mask(roi_frame)

        contours, _ = cv2.findContours(
            mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        boxes    = []
        max_area = self.MAX_CARD_AREA_FRAC * roi_h * W

        for cnt in contours:
            area = cv2.contourArea(cnt)
            if area < self.MIN_CARD_AREA or area > max_area:
                continue

            x, y, w, h = cv2.boundingRect(cnt)
            if w < 10 or h < 5:
                continue

            aspect = min(w, h) / max(w, h)
            if aspect < self.MIN_ASPECT or aspect > self.MAX_ASPECT:
                continue

            solidity = area / float(w * h)
            if solidity < self.MIN_SOLIDITY:
                continue

            # Offset y back to full-frame coordinates
            boxes.append((x, y + roi_y, w, h))

        boxes.sort(key=lambda b: b[0])
        return boxes

    def get_debug_mask(self, frame):
        """Full-frame mask with the scan ROI rectangle drawn in white."""
        mask  = self._build_mask(frame)
        debug = cv2.cvtColor(mask, cv2.COLOR_GRAY2BGR)
        H, W  = frame.shape[:2]
        roi_y, roi_h = self.get_scan_roi(frame)
        cv2.rectangle(debug, (0, roi_y), (W - 1, roi_y + roi_h - 1), (0, 255, 255), 2)
        return debug


class CardScannerNode(Node):
    WINDOW = "Card Scanner"

    def __init__(self):
        super().__init__("card_scanner_node")
        self.declare_parameter("camera_topic", "/camera/image_raw")
        cam = self.get_parameter("camera_topic").value

        self.bridge   = CvBridge()
        self.detector = CardDetector()
        self.lock     = threading.Lock()
        self.frame    = None
        self._running = True

        self.img_sub = self.create_subscription(
            Image, cam, self._img_cb, 10)
        self.get_logger().info("Waiting for camera on {} ...".format(cam))

    def _img_cb(self, msg):
        try:
            f = self.bridge.imgmsg_to_cv2(msg, "bgr8")
            with self.lock:
                self.frame = f
        except Exception as e:
            self.get_logger().error(str(e))

    # ── Design template comparison ────────────────────────────────────────────
    #
    # Core idea: save the card's visual design as a template on Scan 1.
    # When a physical card is flipped 180°, its printed design also rotates 180°.
    # So we compare:
    #   normal_score  = NCC( saved_design,           current_crop )
    #   flipped_score = NCC( rotate180(saved_design), current_crop )
    # If flipped_score > normal_score  →  the card was flipped.

    SIZE = (100, 150)   # width × height to resize all crops before comparing

    def _crop(self, frame, box):
        x, y, w, h = box
        return cv2.resize(frame[y:y+h, x:x+w], self.SIZE)

    def _diff(self, a, b):
        """MAD on histogram-equalized grayscale. Equalization removes lighting
        differences so only the card's spatial design drives the score."""
        g1 = cv2.equalizeHist(cv2.cvtColor(a, cv2.COLOR_BGR2GRAY)).astype(np.float32)
        g2 = cv2.equalizeHist(cv2.cvtColor(b, cv2.COLOR_BGR2GRAY)).astype(np.float32)
        return float(np.mean(np.abs(g1 - g2)))

    def _best_flip_diff(self, saved, current):
        """Try rotate180, flip-horizontal, flip-vertical. Return the lowest diff."""
        return min(
            self._diff(cv2.rotate(saved, cv2.ROTATE_180), current),
            self._diff(cv2.flip(saved, 1),                current),   # left-right
            self._diff(cv2.flip(saved, 0),                current),   # top-bottom
        )

    def _compare(self, frame1, boxes1, frame2, boxes2):
        """
        Match each saved card design to the nearest current card by x-centre.
        diff_normal  = MAD(saved_design, current)          — lower → not flipped
        diff_flipped = best of rotate180/flip-h/flip-v     — lower → flipped
        gap = diff_normal - diff_flipped  (positive → FLIPPED)
        Returns [] if card counts differ.
        """
        if len(boxes1) != len(boxes2):
            return []

        results = []
        for i, b1 in enumerate(boxes1):
            cx1 = b1[0] + b1[2] // 2
            b2  = min(boxes2, key=lambda b: abs((b[0] + b[2]//2) - cx1))

            saved_design = self._crop(frame1, b1)
            current_crop = self._crop(frame2, b2)

            cv2.imwrite("/tmp/card{}_design.jpg".format(i+1),  saved_design)
            cv2.imwrite("/tmp/card{}_current.jpg".format(i+1), current_crop)

            diff_n = round(self._diff(saved_design, current_crop), 2)
            diff_f = round(self._best_flip_diff(saved_design, current_crop), 2)
            gap    = round(diff_n - diff_f, 2)

            results.append({
                "card":         i + 1,
                "box":          b2,
                "diff_normal":  diff_n,
                "diff_flipped": diff_f,
                "gap":          gap,
                "flipped":      diff_f < diff_n,
            })

        results.sort(key=lambda r: r["gap"], reverse=True)
        return results

    # ── Display loop ──────────────────────────────────────────────────────────

    def display_loop(self):
        """
        Live feed with card detection.
        Keys:
          S — save current frame as Scan 1 (before)
          C — compare current frame with Scan 1, highlight changed card
          R — reset
          Q — quit
        """
        cv2.namedWindow(self.WINDOW, cv2.WINDOW_NORMAL)
        cv2.resizeWindow(self.WINDOW, 800, 600)
        cv2.namedWindow("Mask Debug", cv2.WINDOW_NORMAL)
        cv2.resizeWindow("Mask Debug", 400, 300)

        scan1_frame   = None
        scan1_boxes   = None
        current_crops = None   # card crops from last compare (shown as F1, F2...)
        result_text   = "S=Save  C=Compare  R=Reset  Q=Quit"
        result_color  = (0, 220, 255)
        changed_box   = None
        prev_count    = 0

        while self._running:
            with self.lock:
                frame = self.frame.copy() if self.frame is not None else None

            if frame is None:
                cv2.waitKey(1)
                time.sleep(0.033)
                continue

            # ── Mask Debug window ─────────────────────────────────────────────
            mask_debug = self.detector.get_debug_mask(frame)

            # Top strip: C1=saved design (green), F1=actual current card (red/cyan)
            if scan1_frame is not None and scan1_boxes:
                thumb_w, thumb_h = 80, 60
                pad = 4
                for idx, b in enumerate(scan1_boxes):
                    tx = pad + idx * (thumb_w * 2 + pad * 3)
                    ty = pad
                    if tx + thumb_w * 2 + pad >= mask_debug.shape[1]:
                        break

                    # C = saved design
                    saved_thumb = cv2.resize(self._crop(scan1_frame, b), (thumb_w, thumb_h))
                    mask_debug[ty:ty+thumb_h, tx:tx+thumb_w] = saved_thumb
                    cv2.rectangle(mask_debug, (tx, ty), (tx+thumb_w, ty+thumb_h), (0, 255, 0), 1)
                    cv2.putText(mask_debug, "C{}".format(idx+1),
                                (tx+2, ty+12), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 255, 0), 1)

                    # F = actual current card crop (only for the changed card)
                    fx = tx + thumb_w + pad
                    if current_crops is not None and idx in current_crops:
                        cur_thumb = cv2.resize(current_crops[idx], (thumb_w, thumb_h))
                        border_color = (0, 0, 255)   # red = this card was changed
                    else:
                        # No compare yet — show live crop from current frame if card visible
                        live_boxes = self.detector.find_cards(frame)
                        if idx < len(live_boxes):
                            cur_thumb = cv2.resize(self._crop(frame, live_boxes[idx]), (thumb_w, thumb_h))
                        else:
                            cur_thumb = np.zeros((thumb_h, thumb_w, 3), dtype=np.uint8)
                        border_color = (0, 255, 255)   # cyan = live preview
                    mask_debug[ty:ty+thumb_h, fx:fx+thumb_w] = cur_thumb
                    cv2.rectangle(mask_debug, (fx, ty), (fx+thumb_w, ty+thumb_h), border_color, 1)
                    cv2.putText(mask_debug, "F{}".format(idx+1),
                                (fx+2, ty+12), cv2.FONT_HERSHEY_SIMPLEX, 0.4, border_color, 1)

            cv2.imshow("Mask Debug", mask_debug)

            boxes = self.detector.find_cards(frame)
            disp  = frame.copy()

            # Draw all detected cards in green
            for i, (x, y, w, h) in enumerate(boxes, start=1):
                cv2.rectangle(disp, (x, y), (x+w, y+h), (0, 255, 0), 3)
                label = "CARD {}".format(i)
                (tw, th), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.8, 2)
                ty = max(y - 4, th + 8)
                cv2.rectangle(disp, (x, ty-th-6), (x+tw+6, ty+2), (0, 180, 0), -1)
                cv2.putText(disp, label, (x+3, ty),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)

            # Highlight changed card in red with card number
            if changed_box is not None:
                x, y, w, h = changed_box
                cv2.rectangle(disp, (x-6, y-6), (x+w+6, y+h+6), (0, 0, 255), 5)
                label = "CARD {} CHANGED!".format(
                    next((i for i, b in enumerate(boxes, 1) if b == changed_box), "?"))
                cv2.putText(disp, label, (x, max(y - 14, 30)),
                            cv2.FONT_HERSHEY_SIMPLEX, 1.1, (0, 0, 255), 3)

            # Scan1 indicator
            if scan1_frame is not None:
                cv2.putText(disp, "DESIGN SAVED ({} cards)".format(len(scan1_boxes)),
                            (disp.shape[1]-340, 26),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2)

            # Status bar
            cv2.rectangle(disp, (0, 0), (disp.shape[1], 36), (0, 0, 0), -1)
            cv2.putText(disp, result_text, (8, 26),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, result_color, 2)

            cv2.imshow(self.WINDOW, disp)

            if len(boxes) != prev_count:
                print("  Detected {} card(s)".format(len(boxes)) if boxes else "  No card")
                prev_count = len(boxes)

            key = cv2.waitKey(1) & 0xFF

            # ── S: save scan 1 ────────────────────────────────────────────────
            if key == ord('s'):
                if not boxes:
                    result_text  = "No cards visible — cannot save!"
                    result_color = (0, 0, 255)
                else:
                    scan1_frame   = frame.copy()
                    scan1_boxes   = boxes[:]
                    current_crops = None
                    changed_box   = None
                    result_text  = "Design saved ({} cards) — flip a card, then press C.".format(len(boxes))
                    result_color = (0, 255, 0)
                    print("\n  [SAVED] {} card design(s) remembered. Flip a card then press C.\n".format(len(boxes)))

            # ── C: compare with scan 1 ────────────────────────────────────────
            elif key == ord('c'):
                if scan1_frame is None:
                    result_text  = "Press S first to save Scan 1!"
                    result_color = (0, 0, 255)
                elif not boxes:
                    result_text  = "No cards visible in current frame!"
                    result_color = (0, 0, 255)
                elif len(boxes) != len(scan1_boxes):
                    result_text  = "Need {} cards, seeing {} — show all cards!".format(
                        len(scan1_boxes), len(boxes))
                    result_color = (0, 165, 255)
                    print("  Wrong card count: need {}, got {}.".format(
                        len(scan1_boxes), len(boxes)))
                else:
                    results = self._compare(scan1_frame, scan1_boxes, frame, boxes)

                    # Pick card with highest gap (flip_score - normal_score)
                    best = max(results, key=lambda r: r["gap"])

                    # Only store crop for the card detected as changed/flipped
                    current_crops = current_crops or {}
                    current_crops[best["card"] - 1] = self._crop(frame, best["box"])

                    print()
                    for r in results:
                        tag = " <<< CHANGED" if r["card"] == best["card"] else ""
                        print("  Card {}: normal={:.1f}  flipped={:.1f}  gap={:+.1f}{}".format(
                            r["card"], r["diff_normal"], r["diff_flipped"],
                            r["gap"], tag))
                    print()

                    # Always pick the card with the highest gap — user confirmed
                    # the changed card always has the clearly highest gap.
                    changed_box  = best["box"]
                    result_text  = ">>> CARD {} CHANGED!  gap={:+.1f}".format(
                        best["card"], best["gap"])
                    result_color = (0, 0, 255)
                    print("  >>> CARD {} CHANGED! (gap={:+.1f})\n".format(
                        best["card"], best["gap"]))

            # ── R: reset ──────────────────────────────────────────────────────
            elif key == ord('r'):
                scan1_frame   = None
                scan1_boxes   = None
                current_crops = None
                changed_box   = None
                result_text   = "Reset. Press S to save Scan 1."
                result_color = (0, 220, 255)
                print("  Reset.")

            # ── Q: quit ───────────────────────────────────────────────────────
            elif key == ord('q'):
                self._running = False
                break

            time.sleep(0.033)


def main(args=None):
    rclpy.init(args=args)
    node = CardScannerNode()

    spin_thread = threading.Thread(
        target=rclpy.spin, args=(node,), daemon=True)
    spin_thread.start()

    try:
        node.display_loop()
    except KeyboardInterrupt:
        pass
    finally:
        node._running = False
        node.destroy_node()
        rclpy.shutdown()
        cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
