#!/usr/bin/env python3
"""
Live card scanner using Snehil's detection logic.
Shows live feed with green boxes around detected cards.
"""

import rclpy
from rclpy.node import Node
from sensor_msgs.msg import Image
from cv_bridge import CvBridge
from geometry_msgs.msg import Twist

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

    # ── Scan zone (fixed pixels from bottom) ─────────────────────────────────
    SCAN_ROI_HEIGHT = 150   # px — height of the scan strip at the bottom

    # ── Fixed card size bounds (pixels inside the scan ROI) ───────────────────
    # Tune these to match the actual card size seen by your camera.
    CARD_MIN_W   = 40
    CARD_MAX_W   = 280
    CARD_MIN_H   = 25
    CARD_MAX_H   = 140
    MIN_SOLIDITY = 0.45

    def _build_mask(self, frame):
        gray    = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        blurred = cv2.GaussianBlur(gray, (7, 7), 0)
        _, mask = cv2.threshold(
            blurred, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        k    = cv2.getStructuringElement(cv2.MORPH_RECT, (9, 9))
        mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, k, iterations=2)
        mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN,  k, iterations=1)
        return mask

    def get_scan_roi(self, frame):
        """Returns (roi_y, roi_h) — fixed-height bottom strip."""
        H     = frame.shape[0]
        roi_h = min(self.SCAN_ROI_HEIGHT, H)
        roi_y = H - roi_h
        return roi_y, roi_h

    def find_cards(self, frame):
        """Returns list of (x, y, w, h) sorted left-to-right, within scan ROI only."""
        H, W             = frame.shape[:2]
        roi_y, roi_h     = self.get_scan_roi(frame)
        roi_frame        = frame[roi_y:roi_y + roi_h, 0:W]
        mask             = self._build_mask(roi_frame)

        contours, _ = cv2.findContours(
            mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        boxes = []
        for cnt in contours:
            x, y, w, h = cv2.boundingRect(cnt)

            # Fixed size gate — reject anything too small or too large
            if not (self.CARD_MIN_W <= w <= self.CARD_MAX_W):
                continue
            if not (self.CARD_MIN_H <= h <= self.CARD_MAX_H):
                continue

            area     = cv2.contourArea(cnt)
            solidity = area / float(w * h)
            if solidity < self.MIN_SOLIDITY:
                continue

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

    # ── Navigation constants (tune if robot over/under-shoots) ────────────────
    _TURN_SPEED      = 0.5              # rad/s
    _FWD_SPEED       = 0.15            # m/s
    _TURN_90_SECS    = (np.pi / 2) / _TURN_SPEED   # ≈ 3.14 s for 90°
    _MIN_DRIVE_SECS  = 1.5             # ← TUNE: min drive time to clear current card
    _FWD_TIMEOUT     = 12.0            # max seconds before giving up

    # Navigation states
    _NAV_IDLE        = 0
    _NAV_TURN_RIGHT  = 1   # step 1: rotate CW 90°
    _NAV_DRIVE_FWD   = 2   # step 2: drive forward calculated distance
    _NAV_TURN_LEFT   = 3   # step 3: rotate CCW 90°
    _NAV_ALIGN       = 4   # step 4: visual servo fine-alignment
    _NAV_LINE_UP     = 5   # L key: rotate only until card is on centre line

    def __init__(self):
        super().__init__("card_scanner_node")
        self.declare_parameter("camera_topic", "/camera/image_raw")
        cam = self.get_parameter("camera_topic").value

        self.bridge        = CvBridge()
        self.detector      = CardDetector()
        self.lock          = threading.Lock()
        self.frame         = None
        self._running      = True
        self._nav_state      = self._NAV_IDLE
        self._nav_start_t    = 0.0
        self._nav_lost_card  = False   # True once current card leaves view during drive
        _FWD_TIMEOUT         = 12.0    # max seconds driving before giving up

        self.img_sub  = self.create_subscription(Image, cam, self._img_cb, 10)
        self.vel_pub  = self.create_publisher(Twist, '/cmd_vel', 10)
        self.get_logger().info("Waiting for camera on {} ...".format(cam))

    def _img_cb(self, msg):
        try:
            f = self.bridge.imgmsg_to_cv2(msg, "bgr8")
            with self.lock:
                self.frame = f
        except Exception as e:
            self.get_logger().error(str(e))

    # ── Stable card numbering ────────────────────────────────────────────────
    @staticmethod
    def _stable_map(boxes, ref_cx):
        """
        Match each detected box to the nearest reference x-centre.
        Returns {card_num (1-based): box} with fixed card numbers.
        If no ref_cx yet, numbers left-to-right as usual.
        """
        if not ref_cx or not boxes:
            return {i+1: b for i, b in enumerate(sorted(boxes, key=lambda b: b[0]))}
        result   = {}
        assigned = set()
        for box in boxes:
            cx = box[0] + box[2] // 2
            candidates = [i for i in range(len(ref_cx)) if i not in assigned]
            if not candidates:
                break
            best = min(candidates, key=lambda i: abs(ref_cx[i] - cx))
            assigned.add(best)
            result[best + 1] = box
        return result

    # ── Navigation state machine ─────────────────────────────────────────────
    def _nav_step(self, smap, target_num, cx_win, threshold):
        """
        One frame tick of the turn→drive→turn→align state machine.
        smap       — {card_num: box}  stable-numbered current detections
        target_num — the FIXED card number we are navigating toward
        Returns (Twist cmd, status string, done flag).
        """
        cmd  = Twist()
        now  = time.time()
        done = False

        if self._nav_state == self._NAV_TURN_RIGHT:
            elapsed = now - self._nav_start_t
            if elapsed < self._TURN_90_SECS:
                cmd.angular.z = -self._TURN_SPEED
                status = "Step 1/3: Turning RIGHT 90°  ({:.1f} s)".format(
                    self._TURN_90_SECS - elapsed)
            else:
                self._nav_state      = self._NAV_DRIVE_FWD
                self._nav_start_t    = now
                self._nav_lost_card  = False   # reset: current card still visible
                status = "Step 2/3: Driving — waiting to pass Card {}...".format(target_num - 1)
                print("  [NAV] Turn right done — driving forward.")

        elif self._nav_state == self._NAV_DRIVE_FWD:
            elapsed = now - self._nav_start_t
            cmd.linear.x = self._FWD_SPEED

            if elapsed >= self._MIN_DRIVE_SECS and smap:
                # Minimum clearing distance covered AND next card is visible → stop
                cmd.linear.x    = 0.0
                self._nav_state   = self._NAV_TURN_LEFT
                self._nav_start_t = now
                status = "Step 3/3: Card {} found! Turning LEFT 90°...".format(target_num)
                print("  [NAV] Card {} detected at {:.1f}s — turning left.".format(
                    target_num, elapsed))
            elif elapsed > self._FWD_TIMEOUT:
                cmd.linear.x    = 0.0
                self._nav_state   = self._NAV_TURN_LEFT
                self._nav_start_t = now
                status = "Step 3/3: Timeout — Turning LEFT 90°..."
                print("  [NAV] Drive timeout {:.1f}s — turning left anyway.".format(elapsed))
            elif elapsed < self._MIN_DRIVE_SECS:
                status = "Step 2/3: Clearing Card {}...  ({:.1f}s / {:.1f}s)".format(
                    target_num - 1, elapsed, self._MIN_DRIVE_SECS)
            else:
                status = "Step 2/3: Searching for Card {}...  ({:.1f}s)".format(
                    target_num, elapsed)

        elif self._nav_state == self._NAV_TURN_LEFT:
            elapsed = now - self._nav_start_t
            if elapsed < self._TURN_90_SECS:
                cmd.angular.z = +self._TURN_SPEED
                status = "Step 3/3: Turning LEFT 90°  ({:.1f} s)".format(
                    self._TURN_90_SECS - elapsed)
            else:
                self._nav_state = self._NAV_ALIGN
                status = "Fine-aligning to Card {}...".format(target_num)
                print("  [NAV] Turn left done — fine aligning.")

        elif self._nav_state == self._NAV_ALIGN:
            # Find target card by its stable number; fall back to nearest card
            tgt_box = smap.get(target_num) or (
                min(smap.values(),
                    key=lambda b: abs((b[0]+b[2]//2) - cx_win))
                if smap else None)
            if tgt_box:
                offset = (tgt_box[0] + tgt_box[2]//2) - cx_win
                if abs(offset) <= threshold:
                    self._nav_state = self._NAV_IDLE
                    done   = True
                    status = "Card {} aligned! Press N=next  A=go again.".format(target_num)
                    print("  [NAV] Card {} aligned — done.".format(target_num))
                else:
                    cmd.angular.z = -0.35 * (offset / float(cx_win))
                    status = "Aligning Card {}...".format(target_num)
            else:
                status = "Waiting for Card {} in view...".format(target_num)

        elif self._nav_state == self._NAV_LINE_UP:
            # Rotate in place until target card is on the centre line, then stop
            tgt_box = smap.get(target_num) or (
                min(smap.values(),
                    key=lambda b: abs((b[0]+b[2]//2) - cx_win))
                if smap else None)
            if tgt_box:
                offset = (tgt_box[0] + tgt_box[2]//2) - cx_win
                if abs(offset) <= threshold:
                    self._nav_state = self._NAV_IDLE
                    done   = True
                    status = "Lined up to Card {}!".format(target_num)
                    print("  [LINE UP] Card {} on centre line — stopped.".format(target_num))
                else:
                    cmd.angular.z = -0.35 * (offset / float(cx_win))
                    status = "Lining up Card {}...".format(target_num)
            else:
                status = "No card visible — waiting..."

        else:
            status = ""
            done   = True

        return cmd, status, done

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

        # Per-card design memory  {card_num: BGR crop (SIZE)}
        designs       = {}
        # Per-card check results  {card_num: {"flipped": bool, "gap": float}}
        check_results = {}

        result_text   = "S=SaveDesign  C=Check  N=Next  L=LineUp  A=AutoNav  R=Reset  Q=Quit"
        result_color  = (0, 220, 255)
        changed_box   = None
        prev_count    = 0
        target_num    = 1
        ref_cx        = None

        while self._running:
            with self.lock:
                frame = self.frame.copy() if self.frame is not None else None

            if frame is None:
                cv2.waitKey(1)
                time.sleep(0.033)
                continue

            boxes = self.detector.find_cards(frame)
            # Only one card visible at a time — assign it to target_num.
            # If multiple blobs slip through, take the one nearest to centre.
            H_f, W_f = frame.shape[:2]
            if boxes:
                best_box = min(boxes, key=lambda b: abs((b[0]+b[2]//2) - W_f//2))
                smap  = {target_num: best_box}
            else:
                smap  = {}
            snums = list(smap.items())

            # ── Mask Debug window ─────────────────────────────────────────────
            mask_debug = self.detector.get_debug_mask(frame)
            H_d, W_d   = mask_debug.shape[:2]
            roi_y_g, roi_h_g = self.detector.get_scan_roi(frame)
            cx_win   = W_d // 2
            guide_y  = roi_y_g + roi_h_g // 2

            # Top strip: one thumbnail per saved card design
            # Green border = design saved  |  Red border + label = FLIPPED  |  Grey = not yet checked
            if designs:
                thumb_w, thumb_h = 70, 55
                pad = 3
                for cn in sorted(designs.keys()):
                    tx = pad + (cn - 1) * (thumb_w + pad)
                    ty = pad
                    if tx + thumb_w >= W_d:
                        break
                    thumb = cv2.resize(designs[cn], (thumb_w, thumb_h))
                    mask_debug[ty:ty+thumb_h, tx:tx+thumb_w] = thumb

                    res = check_results.get(cn)
                    if res is None:
                        border = (0, 255, 0)      # green = saved, not yet checked
                        label  = "C{}".format(cn)
                    elif res["flipped"]:
                        border = (0, 0, 255)      # red = FLIPPED
                        label  = "C{}!".format(cn)
                    else:
                        border = (0, 200, 0)      # dark green = OK
                        label  = "C{}OK".format(cn)

                    # Highlight target card with brighter border
                    thick = 2 if cn == target_num else 1
                    cv2.rectangle(mask_debug, (tx, ty), (tx+thumb_w, ty+thumb_h), border, thick)
                    cv2.putText(mask_debug, label,
                                (tx+2, ty+12), cv2.FONT_HERSHEY_SIMPLEX, 0.38, border, 1)

            # ── Navigation guide + auto-centre control ────────────────────────
            # Dashed vertical centre line
            for yy in range(roi_y_g, roi_y_g + roi_h_g, 10):
                cv2.line(mask_debug,
                         (cx_win, yy), (cx_win, min(yy + 5, roi_y_g + roi_h_g)),
                         (180, 180, 180), 1)

            threshold     = W_d // 8   # ±12.5 % of width = aligned
            is_navigating = self._nav_state != self._NAV_IDLE

            if smap:
                # Grey dots for all cards, green dot for the TARGET card
                for cn, b in snums:
                    cv2.circle(mask_debug,
                               (b[0]+b[2]//2, guide_y), 5, (160, 160, 160), -1)

                target_box = smap.get(target_num) or list(smap.values())[0]
                target_cx  = target_box[0] + target_box[2] // 2
                offset     = target_cx - cx_win

                cv2.circle(mask_debug, (target_cx, guide_y), 10, (0, 255, 0), -1)
                cv2.putText(mask_debug, "CARD {}".format(target_num),
                            (max(target_cx - 30, 0), guide_y - 16),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 255, 0), 2)

                if not is_navigating and abs(offset) > threshold:
                    col = (0, 165, 255)
                    cv2.arrowedLine(mask_debug,
                                    (target_cx, guide_y), (cx_win, guide_y),
                                    col, 3, tipLength=0.25)
                    direction = "MOVE RIGHT >" if offset > 0 else "< MOVE LEFT"
                    cv2.putText(mask_debug, direction,
                                (W_d - 200 if offset > 0 else 10, guide_y - 14),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.75, col, 2)

            # Run state-machine step and publish velocity
            if is_navigating:
                nav_cmd, nav_status, nav_done = self._nav_step(
                    smap, target_num, cx_win, threshold)
                self.vel_pub.publish(nav_cmd)
                if nav_done:
                    self.vel_pub.publish(Twist())
                    result_text  = nav_status
                    result_color = (0, 255, 0)
                else:
                    result_text  = nav_status
                    result_color = (0, 255, 255)
                cv2.putText(mask_debug, nav_status[:40],
                            (4, roi_y_g + roi_h_g - 8),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 255, 255), 1)

            # Navigation state indicator
            if is_navigating:
                cv2.putText(mask_debug, "AUTO NAV",
                            (W_d - 120, roi_y_g - 6),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2)

            cv2.imshow("Mask Debug", mask_debug)

            disp = frame.copy()

            # Draw all detected cards with STABLE numbers + check result overlay
            for card_num, (x, y, w, h) in snums:
                res = check_results.get(card_num)
                if res is None:
                    box_color  = (0, 255, 0)    # green = not yet checked
                    box_thick  = 2
                elif res["flipped"]:
                    box_color  = (0, 0, 255)    # red = FLIPPED
                    box_thick  = 5
                else:
                    box_color  = (0, 200, 0)    # dark green = OK
                    box_thick  = 2

                cv2.rectangle(disp, (x, y), (x+w, y+h), box_color, box_thick)

                if res and res["flipped"]:
                    label = "CARD {} FLIPPED!".format(card_num)
                elif res:
                    label = "CARD {} OK".format(card_num)
                else:
                    label = "CARD {}{}".format(
                        card_num, " [saved]" if card_num in designs else "")

                (tw, th), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.7, 2)
                ty = max(y - 4, th + 8)
                cv2.rectangle(disp, (x, ty-th-6), (x+tw+6, ty+2), box_color, -1)
                cv2.putText(disp, label, (x+3, ty),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)

            # Designs saved indicator
            if designs:
                cv2.putText(disp, "Designs saved: {}".format(sorted(designs.keys())),
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

            # ── S: save design for the current target card ────────────────────
            if key == ord('s'):
                tgt_box = smap.get(target_num)
                if tgt_box is None:
                    result_text  = "Card {} not visible!".format(target_num)
                    result_color = (0, 0, 255)
                else:
                    designs[target_num] = self._crop(frame, tgt_box)
                    check_results.pop(target_num, None)
                    changed_box = None
                    if ref_cx is None:
                        ref_cx = [b[0]+b[2]//2 for b in sorted(boxes, key=lambda b: b[0])]
                    cv2.imwrite("/tmp/card{}_design.jpg".format(target_num),
                                designs[target_num])
                    result_text  = "Card {} design saved! ({} total)  N=next  C=check".format(
                        target_num, len(designs))
                    result_color = (0, 255, 0)
                    print("  [SAVED] Card {} design. Total saved: {}".format(
                        target_num, sorted(designs.keys())))

            # ── C: check current card against its saved design ────────────────
            elif key == ord('c'):
                tgt_box = smap.get(target_num)
                if target_num not in designs:
                    result_text  = "No design for Card {}. Press S first.".format(target_num)
                    result_color = (0, 0, 255)
                elif tgt_box is None:
                    result_text  = "Card {} not visible!".format(target_num)
                    result_color = (0, 0, 255)
                else:
                    saved   = designs[target_num]
                    current = self._crop(frame, tgt_box)
                    diff_n  = round(self._diff(saved, current), 2)
                    diff_f  = round(self._best_flip_diff(saved, current), 2)
                    gap     = round(diff_n - diff_f, 2)
                    flipped = gap > 0
                    check_results[target_num] = {"flipped": flipped, "gap": gap}
                    changed_box = tgt_box if flipped else None
                    cv2.imwrite("/tmp/card{}_current.jpg".format(target_num), current)
                    print("  Card {}: normal={:.1f}  flipped={:.1f}  gap={:+.1f}  → {}".format(
                        target_num, diff_n, diff_f, gap, "FLIPPED!" if flipped else "OK"))
                    if flipped:
                        result_text  = ">>> CARD {} FLIPPED!  gap={:+.1f}".format(target_num, gap)
                        result_color = (0, 0, 255)
                    else:
                        result_text  = "Card {} OK  gap={:+.1f}".format(target_num, gap)
                        result_color = (0, 200, 0)

            # ── R: reset ──────────────────────────────────────────────────────
            elif key == ord('r'):
                designs.clear()
                check_results.clear()
                changed_box   = None
                ref_cx        = None
                target_num    = 1
                result_text   = "Reset. Go to Card 1, press S to save design."
                result_color  = (0, 220, 255)
                print("  Reset.")

            # ── A: start/stop auto navigation to target card ─────────────────
            elif key == ord('a'):
                if self._nav_state != self._NAV_IDLE:
                    self._nav_state = self._NAV_IDLE
                    self.vel_pub.publish(Twist())
                    result_text  = "AUTO NAV stopped."
                    result_color = (0, 220, 255)
                    print("  [NAV] Stopped by user.")
                else:
                    # Visual drive: pass current card, stop when next card appears
                    self._nav_state     = self._NAV_TURN_RIGHT
                    self._nav_start_t   = time.time()
                    self._nav_lost_card = False
                    result_text  = "AUTO NAV → Card {}  (visual spacing)".format(target_num)
                    result_color = (0, 255, 255)
                    print("  [NAV] Moving to Card {} — will stop when card detected.".format(target_num))

            # ── N: next target card ───────────────────────────────────────────
            elif key == ord('n'):
                target_num  += 1
                result_text  = "Target → Card {}  (press A to go there)".format(target_num)
                result_color = (0, 255, 255)
                print("  [NAV] Target set to Card {}.".format(target_num))

            # ── L: line up — rotate until card is on centre line ─────────────
            elif key == ord('l'):
                if self._nav_state != self._NAV_IDLE:
                    self._nav_state = self._NAV_IDLE
                    self.vel_pub.publish(Twist())
                    result_text  = "Line-up stopped."
                    result_color = (0, 220, 255)
                elif smap:
                    self._nav_state   = self._NAV_LINE_UP
                    self._nav_start_t = time.time()
                    result_text  = "Lining up to Card {}...".format(target_num)
                    result_color = (0, 255, 255)
                    print("  [LINE UP] Rotating to align Card {} with centre line.".format(target_num))
                else:
                    result_text  = "No cards visible."
                    result_color = (0, 0, 255)

            # ── Q: quit ───────────────────────────────────────────────────────
            elif key == ord('q'):
                self.vel_pub.publish(Twist())       # stop robot before quitting
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
