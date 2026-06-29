#!/usr/bin/env python3


import rclpy
from rclpy.node import Node
from sensor_msgs.msg import Image
from cv_bridge import CvBridge
from geometry_msgs.msg import Twist
from nav_msgs.msg import Odometry
from rclpy.qos import QoSProfile, ReliabilityPolicy, HistoryPolicy
import math

import cv2
import numpy as np
import threading
import time


class CardDetector:
    """
    Detects cards in a camera frame and returns their bounding boxes.

    Pipeline:
      1. HSV white mask (low-saturation bright pixels)  OR  grayscale > 160 if saturation is unreliable
      2. Zero out the TOP 30% of the frame (walls / curtains / ceiling)
      3. Morphological close + open to clean the mask
      4. Contour filter: area, polygon approx, aspect, solidity, white-density
    """

    MIN_CARD_AREA       = 1500
    MAX_CARD_AREA_FRAC  = 0.08    # reject blobs > 8% of frame area
    CARD_ROI_TOP_FRAC   = 0.30    # ignore top 30% of the frame
    MIN_SOLIDITY        = 0.55    # contour fills its bbox
    MIN_WHITE_DENSITY   = 0.55    # bbox is mostly white pixels in the mask
    MIN_ASPECT          = 0.25    # short/long; angled cards can go this low
    MAX_ASPECT          = 0.90

    def _build_mask(self, frame):
        """
Returns a binary mask where the cards should be white blobs. 
 We want to be generous in this mask to avoid missing any cards, 
 but then rely on the contour filter to weed out false positives.  
 So we take anything that looks like a bright, low-saturation pixel in HSV,
   OR a bright pixel in grayscale (in case saturation is unreliable).  
   Then we clean up the mask with morphology and contour filters.  
   Finally, we zero out the top part of the frame which often has distracting 
   bright spots on walls / curtains / ceiling.
        """
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        hsv  = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
        H, _ = gray.shape

        lower_white = np.array([0, 0, 150])
        upper_white = np.array([180, 80, 255])
        mask_white  = cv2.inRange(hsv, lower_white, upper_white)
        _, mask_gray = cv2.threshold(gray, 160, 255, cv2.THRESH_BINARY)
        mask = cv2.bitwise_or(mask_white, mask_gray)

        roi_top = int(H * self.CARD_ROI_TOP_FRAC)
        if roi_top > 0:
            mask[:roi_top, :] = 0

        kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (5, 5))
        mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel, iterations=2)
        mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN,  kernel, iterations=1)
        return mask

    def get_scan_roi(self, frame):
        """Returns (roi_y, roi_h) — the active detection region (below cutoff).
            This is used to draw the yellow guide rectangle in the debug view, and
            also to compute the ROI for density checks in find_cards().
        """
        H     = frame.shape[0]
        roi_y = int(H * self.CARD_ROI_TOP_FRAC)
        roi_h = H - roi_y
        return roi_y, roi_h

    def find_cards(self, frame):
        """Returns list of (x, y, w, h) in full-frame coords, sorted left→right.
            Each box is a detected card candidate that passed all contour filters.
        """
        H, W = frame.shape[:2]
        mask = self._build_mask(frame)

        contours, _ = cv2.findContours(
            mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        boxes    = []
        max_area = self.MAX_CARD_AREA_FRAC * H * W

        for cnt in contours:
            area = cv2.contourArea(cnt)
            if area < self.MIN_CARD_AREA or area > max_area:
                continue

            peri   = cv2.arcLength(cnt, True)
            approx = cv2.approxPolyDP(cnt, 0.02 * peri, True)
            if len(approx) < 4 or len(approx) > 8:
                continue

            x, y, w, h = cv2.boundingRect(cnt)
            if w < 20 or h < 30:
                continue

            short  = min(w, h)
            longe  = max(w, h)
            aspect = short / float(longe)
            if aspect < self.MIN_ASPECT or aspect > self.MAX_ASPECT:
                continue

            bbox_area = float(w * h)
            if bbox_area <= 0:
                continue
            solidity = area / bbox_area
            if solidity < self.MIN_SOLIDITY:
                continue

            roi_mask = mask[y:y+h, x:x+w]
            if roi_mask.size == 0:
                continue
            white_density = float(np.count_nonzero(roi_mask)) / roi_mask.size
            if white_density < self.MIN_WHITE_DENSITY:
                continue

            boxes.append((x, y, w, h))

        boxes.sort(key=lambda b: b[0])
        return boxes

    def get_debug_mask(self, frame):
        """Full-frame mask with the ROI cutoff line drawn in yellow.
            This is useful for tuning the HSV / grayscale thresholds and the ROI cutoff.

        """
        mask  = self._build_mask(frame)
        debug = cv2.cvtColor(mask, cv2.COLOR_GRAY2BGR)
        _, W  = frame.shape[:2]
        roi_y, roi_h = self.get_scan_roi(frame)
        cv2.rectangle(debug, (0, roi_y), (W - 1, roi_y + roi_h - 1), (0, 255, 255), 2)
        return debug


class CardScannerNode(Node):
    """
    Node for scanning cards using computer vision.
    
    """
    WINDOW = "Card Scanner"

    # ── Navigation constants (tune if robot over/under-shoots) ────────────────
    _CARD_STEP        = 0.30   # m — distance between card scan positions
    _FWD_SPEED        = 0.05   # m/s — slow, steady forward motion
    _HEADING_KP       = 1.1    # heading hold proportional gain during forward
    _HEADING_MAX      = 0.12   # rad/s — max heading correction

    # Navigation states
    _NAV_IDLE        = 0
    _NAV_TURN_RIGHT  = 1   # step 1: rotate CW 90°
    _NAV_DRIVE_FWD   = 2   # step 2: drive forward calculated distance
    _NAV_TURN_LEFT   = 3   # step 3: rotate CCW 90°
    _NAV_ALIGN       = 4   # step 4: visual servo fine-alignment
    _NAV_LINE_UP     = 5   # L key: rotate only until card is on centre line
    _NAV_TURN_AROUND = 6   # 180° CW away from cards (after card 5 saved)
    _NAV_WAIT_SWAP   = 7   # wait WAIT_SECS for someone to flip a card
    _NAV_TURN_BACK   = 8   # 180° CW back to face cards, then start return

    WAIT_SECS        = 10.0   # seconds robot looks away for card swap

    def __init__(self):
        """
            Initializes the CardScannerNode, sets up subscriptions and publishers, and initializes state variables.
                - Subscribes to the camera topic to receive image frames.
            """
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
        self._nav_lost_card  = False
        self._nav_going_left = False          # True during return journey
        self._nav_waiting    = False          # True when paused at a changed card

        self._yaw              = 0.0    # current heading from odometry (radians)
        self._turn_start_yaw   = None  # yaw when the current turn began
        self._nav_first_approach = False  # True for initial straight-line approach to card 1
        self._x                = 0.0    # current odom X (m)
        self._y                = 0.0    # current odom Y (m)
        self._drive_start_x    = None
        self._drive_start_y    = None
        self._drive_heading    = None

        self._nav_wait_deadline   = 0.0    # timestamp when swap wait ends
        self._nav_turnaround_done = False  # signals display_loop to kick off return

        # depth=1 + BEST_EFFORT: always deliver the newest frame, never buffer old ones
        cam_qos = QoSProfile(
            depth=1,
            reliability=ReliabilityPolicy.BEST_EFFORT,
            history=HistoryPolicy.KEEP_LAST)
        self.img_sub  = self.create_subscription(Image, cam, self._img_cb, cam_qos)
        self.odom_sub = self.create_subscription(Odometry, '/odom', self._odom_cb, 10)
        self.vel_pub  = self.create_publisher(Twist, '/cmd_vel', 10)
        self.get_logger().info("Waiting for camera on {} ...".format(cam))

    def _img_cb(self, msg):
        '''Image callback: convert ROS Image to OpenCV format and store the latest frame.'''
        try:
            f = self.bridge.imgmsg_to_cv2(msg, "bgr8")
            with self.lock:
                self.frame = f
        except Exception as e:
            self.get_logger().error(str(e))

    def _odom_cb(self, msg):
        ## Odometry callback: extract current position and yaw from odometry message.
        q = msg.pose.pose.orientation
        self._x = msg.pose.pose.position.x
        self._y = msg.pose.pose.position.y
        # Yaw from quaternion (no external library needed)
        siny = 2.0 * (q.w * q.z + q.x * q.y)
        cosy = 1.0 - 2.0 * (q.y * q.y + q.z * q.z)
        self._yaw = math.atan2(siny, cosy)

    @staticmethod
    def _angle_diff(a, b):## useful for computing how much to turn from heading b to heading a   heading a is the target, heading b is the current
        """Signed shortest angle from b to a, in [-pi, pi].
        means if you want to rotate from heading b to heading a, this is how much and in which direction to turn."""
        d = a - b
        while d >  math.pi: d -= 2 * math.pi
        while d < -math.pi: d += 2 * math.pi
        return d

    @staticmethod
    def _wrap(angle):## useful for normalizing angles to a standard range, e.g. when computing ideal post-turn headings
        """Wrap angle to [-pi, pi]."""
        while angle >  math.pi: angle -= 2 * math.pi
        while angle < -math.pi: angle += 2 * math.pi
        return angle

    # ── Stable card numbering ────────────────────────────────────────────────
    @staticmethod
    def _stable_map(boxes, ref_cx):## useful for maintaining consistent card numbering across frames, even if detection order changes due to noise noise means that the detected boxes might not always come in the same left-to-right order, so we want to assign them to card numbers based on their proximity to reference x-centres (ref_cx) from a previous stable frame.
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
    _TURN_TARGET_RAD  = math.pi / 2           # 90° target for every turn
    _TURN_ANGLE_TOL   = math.radians(2.0)     # stop threshold
    _TURN_SPEED       = 0.22                 # rad/s — normal turn speed
    _TURN_CREEP_SPEED = 0.06                 # rad/s — creep near target
    _TURN_CREEP_RAD   = 0.18                 # rad — enter creep region
    _TURN_MAX_SECS    = 7.0                  # safety timeout if odometry stalls
    _DRIVE_MAX_SECS   = 10.0                 # safety timeout for forward drive
    _DIST_TOL         = 0.006                # m — close enough on distance

    def _nav_step(self, smap, target_num, cx_win, threshold):## useful for executing the navigation steps to approach and align with a target card, based on the current navigation state.  This is called repeatedly in the display loop to update the robot's motion commands until the target card is aligned.
        """
        Direction-aware turn → drive → turn → align state machine.

        Physical convention (independent of journey direction):
          NAV_TURN_RIGHT → CW rotation (negative angular.z)
          NAV_TURN_LEFT  → CCW rotation (positive angular.z)

        Forward journey (left=False): TURN_RIGHT → DRIVE → TURN_LEFT  → ALIGN
        Return  journey (left=True):  TURN_LEFT  → DRIVE → TURN_RIGHT → ALIGN

        Turns are yaw-based (odometry): robot stops exactly when it has
        rotated 90°, so the return journey is a true mirror of the scan.
        Forward motion is distance-based with heading hold; timeouts act
        as a safety cutoff if odom stalls.

        Returns (Twist cmd, status string, done flag).
        """
        cmd  = Twist()
        now  = time.time()
        done = False
        left = self._nav_going_left   # shorthand

        if self._nav_state == self._NAV_TURN_RIGHT:
            # Always CW.
            elapsed = now - self._nav_start_t
            if self._turn_start_yaw is None:
                self._turn_start_yaw = self._yaw
            angle_done = abs(self._angle_diff(self._yaw, self._turn_start_yaw))
            remaining  = self._TURN_TARGET_RAD - angle_done
            if remaining > self._TURN_ANGLE_TOL and elapsed < self._TURN_MAX_SECS:
                speed = (self._TURN_CREEP_SPEED
                         if remaining < self._TURN_CREEP_RAD
                         else self._TURN_SPEED)
                cmd.angular.z = -speed
                status = "Turning RIGHT... ({:.0f}° left)".format(
                    max(0.0, math.degrees(remaining)))
            else:
                cmd.angular.z = 0.0
                if left:
                    self._turn_start_yaw = None
                    self._nav_state = self._NAV_ALIGN
                    status = "Turn RIGHT done — aligning Card {}...".format(target_num)
                    print("  [NAV] Turn RIGHT done — aligning Card {}.".format(target_num))
                else:
                    # FIX: lock the IDEAL post-turn heading computed from where the
                    # turn started, not from the measured yaw (which may still be
                    # settling).  CW 90° → subtract π/2 from start heading.
                    self._drive_heading = self._wrap(self._turn_start_yaw - math.pi / 2)
                    self._turn_start_yaw = None
                    self._nav_state   = self._NAV_DRIVE_FWD
                    self._nav_start_t = now
                    self._drive_start_x = None
                    self._drive_start_y = None
                    status = "Turn RIGHT done — driving..."
                    print("  [NAV] Turn RIGHT done — driving.")

        elif self._nav_state == self._NAV_DRIVE_FWD:
            if self._drive_start_x is None or self._drive_start_y is None:
                self._drive_start_x = self._x
                self._drive_start_y = self._y
                # FIX: only fall back to current yaw when no ideal heading was
                # pre-computed by the preceding turn (e.g. initial approach).
                if self._drive_heading is None:
                    self._drive_heading = self._yaw
                self._nav_start_t   = now

            traveled = math.hypot(self._x - self._drive_start_x,
                                  self._y - self._drive_start_y)
            remaining = self._CARD_STEP - traveled
            elapsed = now - self._nav_start_t

            if remaining <= self._DIST_TOL or elapsed > self._DRIVE_MAX_SECS:
                cmd.linear.x  = 0.0
                cmd.angular.z = 0.0
                self._drive_start_x = None
                self._drive_start_y = None
                self._drive_heading = None
                if left:
                    self._nav_state      = self._NAV_TURN_RIGHT
                    self._nav_start_t    = now
                    self._turn_start_yaw = self._yaw
                    status = "Drive done — turning RIGHT 90°..."
                    print("  [NAV] Return drive done — turning RIGHT.")
                else:
                    self._nav_state      = self._NAV_TURN_LEFT
                    self._nav_start_t    = now
                    self._turn_start_yaw = self._yaw   # reset for the upcoming left turn
                    status = "Drive done — turning LEFT 90°..."
                    print("  [NAV] Drive done — turning LEFT.")
            else:
                herr = self._angle_diff(self._drive_heading, self._yaw)
                corr = max(min(self._HEADING_KP * herr, self._HEADING_MAX),
                           -self._HEADING_MAX)
                cmd.linear.x  = self._FWD_SPEED
                cmd.angular.z = corr
                status = "Driving... ({:.2f}m left)".format(max(0.0, remaining))

        elif self._nav_state == self._NAV_TURN_LEFT:
            # Always CCW.
            elapsed = now - self._nav_start_t
            if self._turn_start_yaw is None:
                self._turn_start_yaw = self._yaw
            angle_done = abs(self._angle_diff(self._yaw, self._turn_start_yaw))
            remaining  = self._TURN_TARGET_RAD - angle_done
            if remaining > self._TURN_ANGLE_TOL and elapsed < self._TURN_MAX_SECS:
                speed = (self._TURN_CREEP_SPEED
                         if remaining < self._TURN_CREEP_RAD
                         else self._TURN_SPEED)
                cmd.angular.z = speed
                status = "Turning LEFT... ({:.0f}° left)".format(
                    max(0.0, math.degrees(remaining)))
            else:
                cmd.angular.z = 0.0
                if left:
                    # FIX: lock ideal heading. CCW 90° → add π/2 to start heading.
                    self._drive_heading = self._wrap(self._turn_start_yaw + math.pi / 2)
                    self._turn_start_yaw = None
                    self._nav_state   = self._NAV_DRIVE_FWD
                    self._nav_start_t = now
                    self._drive_start_x = None
                    self._drive_start_y = None
                    status = "Turn LEFT done — driving..."
                    print("  [NAV] Turn LEFT done — driving.")
                else:
                    self._turn_start_yaw = None
                    self._nav_state = self._NAV_ALIGN
                    self._nav_first_approach = False
                    status = "Turn LEFT done — aligning Card {}...".format(target_num)
                    print("  [NAV] Turn LEFT done — aligning Card {}.".format(target_num))

        elif self._nav_state == self._NAV_ALIGN:
            # smap is already built with area-weighted selection during ALIGN
            # (largest card near centre wins) — so just read target_num directly.
            tight   = max(cx_win // 20, 8)   # ±5% of half-width
            tgt_box = smap.get(target_num)
            if tgt_box:
                offset = (tgt_box[0] + tgt_box[2]//2) - cx_win
                if abs(offset) <= tight:
                    self._nav_state = self._NAV_IDLE
                    done   = True
                    status = "Card {} aligned! Press N=next  A=go again.".format(target_num)
                    print("  [NAV] Card {} aligned — done.".format(target_num))
                else:
                    raw = -0.35 * (offset / float(cx_win))
                    cmd.angular.z = math.copysign(max(abs(raw), 0.12), raw)
                    status = "Aligning Card {}... ({} px off)".format(target_num, int(offset))
            else:
                status = "Waiting for Card {} in view...".format(target_num)

        elif self._nav_state == self._NAV_LINE_UP:
            # Tight threshold: W/16 (half of normal) for precise straight alignment
            tight = max(threshold // 2, 8)
            tgt_box = smap.get(target_num) or (
                min(smap.values(),
                    key=lambda b: abs((b[0]+b[2]//2) - cx_win))
                if smap else None)
            if tgt_box:
                offset = (tgt_box[0] + tgt_box[2]//2) - cx_win
                if abs(offset) <= tight:
                    self._nav_state = self._NAV_IDLE
                    done   = True
                    status = "STRAIGHT — press S to save Card {}".format(target_num)
                    print("  [STRAIGHT] Card {} aligned — press S to save.".format(target_num))
                else:
                    # Slow proportional turn for precise alignment
                    cmd.angular.z = -0.25 * (offset / float(cx_win))
                    status = "Aligning Card {}... ({} px off)".format(target_num, abs(offset))
            else:
                status = "No card visible — waiting..."

        elif self._nav_state == self._NAV_TURN_AROUND:
            # 180° CW rotation to face away from the cards
            elapsed = now - self._nav_start_t
            if self._turn_start_yaw is None:
                self._turn_start_yaw = self._yaw
            angle_done = abs(self._angle_diff(self._yaw, self._turn_start_yaw))
            remaining  = math.pi - angle_done
            if remaining > self._TURN_ANGLE_TOL and elapsed < self._TURN_MAX_SECS * 2:
                speed = (self._TURN_CREEP_SPEED
                         if remaining < self._TURN_CREEP_RAD
                         else self._TURN_SPEED)
                cmd.angular.z = -speed          # CW
                status = "Turning 180° away... ({:.0f}° left)".format(
                    max(0.0, math.degrees(remaining)))
            else:
                cmd.angular.z = 0.0
                self._turn_start_yaw      = None
                self._nav_state           = self._NAV_WAIT_SWAP
                self._nav_wait_deadline   = now + self.WAIT_SECS
                status = "SWAP A CARD NOW! Waiting {:.0f}s...".format(self.WAIT_SECS)
                print("  [TURN] 180° away done — {:.0f}s swap window open.".format(
                    self.WAIT_SECS))

        elif self._nav_state == self._NAV_WAIT_SWAP:
            remaining = max(0.0, self._nav_wait_deadline - now)
            if remaining <= 0.0:
                self._nav_state      = self._NAV_TURN_BACK
                self._nav_start_t    = now
                self._turn_start_yaw = self._yaw
                status = "Swap window closed — turning back..."
                print("  [WAIT] Swap window closed — turning back 180°.")
            else:
                status = "SWAP A CARD! {:.0f}s remaining...".format(remaining)

        elif self._nav_state == self._NAV_TURN_BACK:
            # 180° CW rotation back to face the cards
            elapsed = now - self._nav_start_t
            if self._turn_start_yaw is None:
                self._turn_start_yaw = self._yaw
            angle_done = abs(self._angle_diff(self._yaw, self._turn_start_yaw))
            remaining  = math.pi - angle_done
            if remaining > self._TURN_ANGLE_TOL and elapsed < self._TURN_MAX_SECS * 2:
                speed = (self._TURN_CREEP_SPEED
                         if remaining < self._TURN_CREEP_RAD
                         else self._TURN_SPEED)
                cmd.angular.z = -speed          # CW
                status = "Turning 180° back... ({:.0f}° left)".format(
                    max(0.0, math.degrees(remaining)))
            else:
                cmd.angular.z = 0.0
                self._turn_start_yaw      = None
                self._nav_turnaround_done = True
                self._nav_state           = self._NAV_IDLE
                done   = True
                status = "Turnaround complete — now check Card 5!"
                print("  [TURN] 180° back done — facing Card 5 for comparison.")

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
    ## current_crop is the central region of the detected card at full resolution, extracted by _crop() below.  By keeping only the inner portion of the card and stripping away the edges and borders, we can provide a cleaner input to the NCC comparator that focuses on the printed design, which is what gives a strong orientation signal when flipped.  The _best_corr() function then computes the normalized cross-correlation between the saved template and the current crop, allowing for some search padding to tolerate alignment drift.  The _identify_card() function uses these correlation scores to determine which saved design best matches the current

    SIZE         = (200, 300)   # width × height — internal compare size (larger = more detail)
    _CENTER_FRAC = 0.75         # keep inner 75% of box — strips symmetric border only,
                                # keeps most of the card design visible

    def _crop(self, frame, box):## useful for extracting the central region of the detected card at full resolution, which is used for template comparison.  By keeping only the inner portion of the card and stripping away the edges and borders, we can provide a cleaner input to the NCC comparator that focuses on the printed design, which is what gives a strong orientation signal when flipped.
        """Return the centre region of the detected card at full resolution.
        Edges and borders look identical before and after a 180° flip, so
        stripping them and keeping only the printed design in the middle gives
        the NCC comparator a clean orientation signal."""
        x, y, w, h = box
        H, W = frame.shape[:2]
        pad_x = int(w * (1 - self._CENTER_FRAC) / 2)
        pad_y = int(h * (1 - self._CENTER_FRAC) / 2)
        x0 = max(0, x + pad_x)
        y0 = max(0, y + pad_y)
        x1 = min(W, x + w - pad_x)
        y1 = min(H, y + h - pad_y)
        crop = frame[y0:y1, x0:x1]
        if crop.size == 0:          # box was too small — fall back to full box
            crop = frame[y:y+h, x:x+w]
        return crop.copy()

    # Correlation below this → card is considered changed
    _CHANGE_THRESHOLD = 0.7
    _FLIP_DELTA       = 0.05   # flipped orientation must beat normal by this margin
    _SEARCH_PAD       = 20     # px — search margin to tolerate alignment drift

    def _best_corr(self, tmpl, img):## useful for computing the normalized cross-correlation between the saved template and the current crop, allowing for some search padding to tolerate alignment drift.  By resizing both the template and the current crop to a standard size, we can ensure that the NCC comparison is consistent and robust even if the detected card boxes vary in size or position across frames.
        """Find best NCC of tmpl inside a padded version of img.
        Both inputs are resized to SIZE here, so callers can pass full-res
        crops of different dimensions without alignment problems."""
        g_t = cv2.cvtColor(cv2.resize(tmpl, self.SIZE), cv2.COLOR_BGR2GRAY).astype(np.float32)
        g_i = cv2.cvtColor(cv2.resize(img,  self.SIZE), cv2.COLOR_BGR2GRAY).astype(np.float32)
        pad = self._SEARCH_PAD
        # BORDER_CONSTANT (black) so padded border never gives a false match
        padded = cv2.copyMakeBorder(g_i, pad, pad, pad, pad, cv2.BORDER_CONSTANT, value=0)
        result = cv2.matchTemplate(padded, g_t, cv2.TM_CCOEFF_NORMED)
        _, max_val, _, _ = cv2.minMaxLoc(result)
        return float(max_val)

    def _identify_card(self, current, designs):### useful for finding which saved card design best matches the current crop, and whether the card has been flipped.  By comparing the current crop against both the normal and flipped orientations of each saved design, we can determine not only which card is present but also whether it has been flipped, based on the correlation scores and defined thresholds.
        """Find which saved card design best matches the current crop, and
        whether the card has been flipped.

        Flip is detected by EITHER of:
          (1) Back-face shown: best score against any saved design is below
              _CHANGE_THRESHOLD — current crop doesn't look like ANY saved
              card, so the user turned it over (back face is visible).
          (2) In-plane rotation: flipped orientation of a saved design
              matches current better than the normal orientation by
              _FLIP_DELTA — the card was spun 180° on the table.

        Position-independent: moving a card to a different slot still matches
        its own design in the normal orientation → not flipped.

        Returns dict: {matched_id, flipped, reason, corr_n, corr_f, score}.
        """
        best  = None
        rows  = []
        for sid, saved in designs.items():
            corr_n = self._best_corr(saved, current)
            corr_f = max(
                self._best_corr(cv2.rotate(saved, cv2.ROTATE_180), current),
                self._best_corr(cv2.flip(saved, 1), current),
                self._best_corr(cv2.flip(saved, 0), current),
            )
            rotated_better = corr_f > corr_n + self._FLIP_DELTA
            score   = corr_f if rotated_better else corr_n
            rows.append((sid, corr_n, corr_f, rotated_better, score))
            if best is None or score > best["score"]:
                best = {
                    "matched_id":     sid,
                    "rotated_better": rotated_better,
                    "corr_n":         round(corr_n, 3),
                    "corr_f":         round(corr_f, 3),
                    "score":          round(score, 3),
                }

        if best is None:
            return {"matched_id": None, "flipped": False, "reason": "no_designs",
                    "corr_n": 0.0, "corr_f": 0.0, "score": 0.0}

        # Decide flipped vs not, with reason for the log
        if best["score"] < self._CHANGE_THRESHOLD:
            best["flipped"] = True
            best["reason"]  = "low_score_back_face"
        elif best["rotated_better"]:
            best["flipped"] = True
            best["reason"]  = "rotated_orientation_matches_better"
        else:
            best["flipped"] = False
            best["reason"]  = "matches_saved_design"

        # Diagnostic table so we can see why a particular match was picked
        print("  [MATCH TABLE]  (threshold={:.2f}, flip_delta={:.2f})".format(
            self._CHANGE_THRESHOLD, self._FLIP_DELTA))
        for sid, cn, cf, rb, sc in sorted(rows, key=lambda r: -r[4]):
            mark = " ← picked" if sid == best["matched_id"] else ""
            print("    Card {}: n={:.3f}  f={:.3f}  rotated_better={}  score={:.3f}{}".format(
                sid, cn, cf, "Y" if rb else "n", sc, mark))
        print("    → matched_id={}  flipped={}  reason={}".format(
            best["matched_id"], best["flipped"], best["reason"]))
        return best

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
                continue

            boxes = self.detector.find_cards(frame)
            # Only one card visible at a time — assign it to target_num.
            # If multiple blobs slip through, take the one nearest to centre.
            H_f, W_f = frame.shape[:2]
            if boxes:
                if self._nav_state == self._NAV_ALIGN:
                    # During alignment pick the largest card near centre.
                    # The target card is directly in front → bigger than neighbours.
                    def _align_score(b):
                        cx_off   = abs((b[0] + b[2]//2) - W_f//2) / float(W_f//2 + 1)
                        area_n   = (b[2] * b[3]) / float(W_f * H_f + 1)
                        return cx_off - 2.5 * area_n   # prefer centred AND large
                    best_box = min(boxes, key=_align_score)
                else:
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
            if is_navigating and not self._nav_waiting:
                prev_state = self._nav_state   # capture BEFORE _nav_step changes it
                nav_cmd, nav_status, nav_done = self._nav_step(
                    smap, target_num, cx_win, threshold)
                self.vel_pub.publish(nav_cmd)
                # Debug — confirm commands are being sent
                if nav_cmd.angular.z != 0.0 or nav_cmd.linear.x != 0.0:
                    print("  [CMD] lin={:.2f}  ang={:.2f}  state={}".format(
                        nav_cmd.linear.x, nav_cmd.angular.z, self._nav_state))

                if nav_done:
                    self.vel_pub.publish(Twist())

                    if self._nav_turnaround_done:
                        # Turnaround complete — robot is now facing Card 5.
                        # Enter the return-journey wait so user can press C then SPACE.
                        self._nav_turnaround_done = False
                        self._nav_going_left      = True
                        self._nav_waiting         = True
                        # target_num is still 5 — correct position
                        result_text  = "Facing Card 5 — press C to compare, SPACE to continue"
                        result_color = (0, 255, 255)
                        print("  [AUTO] Facing Card 5 — C=compare  SPACE=continue to Card 4.")
                    elif self._nav_going_left:
                        # Return journey: robot arrived at the card. STOP here
                        # and wait for the user to press C (compare). After
                        # comparison, SPACE will continue to the next card.
                        self._nav_waiting = True
                        result_text  = "At pos {} — press C to compare, SPACE to continue".format(
                            target_num)
                        result_color = (0, 255, 255)
                        print("  [WAIT] At pos {} — C=compare  SPACE=next.".format(target_num))
                    else:
                        # ALIGN or LINE_UP finished — robot is at card, ready to save
                        result_text  = "READY — press S to save Card {}  (L=fine-align)".format(
                            target_num)
                        result_color = (0, 255, 0)
                        print("  [NAV] Card {} ready — press S to save.".format(target_num))
                else:
                    result_text  = nav_status
                    result_color = (0, 255, 255)

                cv2.putText(mask_debug, nav_status[:40],
                            (4, roi_y_g + roi_h_g - 8),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 255, 255), 1)

            # Navigation state indicator
            if is_navigating:
                label = "RETURN <" if self._nav_going_left else "NAV >"
                cv2.putText(mask_debug, label,
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

                # Show only status (no card number) centred inside the box
                if res and res["flipped"]:
                    label = "FLIPPED!"
                elif res:
                    label = "OK"
                else:
                    label = ""   # scanning — no text

                if label:
                    (tw, th), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.65, 2)
                    tx = x + (w - tw) // 2
                    ty = y + (h // 2) + th // 2
                    cv2.putText(disp, label, (tx, ty),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.65, (255, 255, 255), 2)

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

            # ── SPACE: resume after pausing at a changed card ─────────────────
            if key == ord(' ') and self._nav_waiting:
                self._nav_waiting = False
                if target_num > 1:
                    target_num -= 1
                    self._nav_state      = self._NAV_TURN_LEFT   # return — first turn is LEFT
                    self._nav_start_t    = time.time()
                    self._turn_start_yaw = self._yaw              # capture heading for yaw-based turn
                    self._drive_start_x  = None
                    self._drive_start_y  = None
                    self._drive_heading  = None
                    result_text  = "Resuming — moving to Card {}...".format(target_num)
                    result_color = (0, 255, 255)
                    print("  [RESUME] Continuing to Card {}.".format(target_num))
                else:
                    self._nav_going_left = False
                    flipped_cards = [k for k, v in check_results.items() if v["flipped"]]
                    result_text  = "All checked! Changed: {}".format(
                        flipped_cards if flipped_cards else "none")
                    result_color = (0, 0, 255) if flipped_cards else (0, 255, 0)
                    print("  [DONE] Return complete. Changed: {}".format(flipped_cards))

            # ── S: save design for the current target card ────────────────────
            elif key == ord('s'):
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

                    # After card 5 is saved: turn 180° away → wait 10s (swap window)
                    # → turn 180° back → check card 5 → then check cards 4 → 1.
                    if target_num == 5:
                        self._nav_state      = self._NAV_TURN_AROUND
                        self._nav_start_t    = time.time()
                        self._turn_start_yaw = self._yaw
                        result_text  = "Card 5 saved! Turning away — SWAP A CARD NOW!"
                        result_color = (0, 255, 255)
                        print("  [AUTO] Card 5 saved — turning 180° away for card swap.")

            # ── C: check current card — identify it and detect flip ──────────
            elif key == ord('c'):
                tgt_box = smap.get(target_num)
                if not designs:
                    result_text  = "No designs saved yet. Press S first.".format(target_num)
                    result_color = (0, 0, 255)
                elif tgt_box is None:
                    result_text  = "No card visible at position {}!".format(target_num)
                    result_color = (0, 0, 255)
                else:
                    current = self._crop(frame, tgt_box)
                    info    = self._identify_card(current, designs)
                    flipped = info["flipped"]
                    check_results[target_num] = {
                        "flipped":    flipped,
                        "matched_id": info["matched_id"],
                        "gap":        info["score"],
                    }
                    changed_box = tgt_box if flipped else None
                    cv2.imwrite("/tmp/card{}_current.jpg".format(target_num), current)
                    print("  Position {}: matched Card {} ({}). corr_n={:.3f} corr_f={:.3f}".format(
                        target_num, info["matched_id"],
                        "FLIPPED!" if flipped else "normal",
                        info["corr_n"], info["corr_f"]))
                    if flipped:
                        result_text  = ">>> Card {} FLIPPED at position {}!".format(
                            info["matched_id"], target_num)
                        result_color = (0, 0, 255)
                    else:
                        result_text  = "Position {}: Card {} (normal)  score={:.2f}".format(
                            target_num, info["matched_id"], info["score"])
                        result_color = (0, 200, 0)

            # ── R: reset ──────────────────────────────────────────────────────
            elif key == ord('r'):
                designs.clear()
                check_results.clear()
                self._nav_going_left      = False
                self._nav_waiting         = False
                self._nav_first_approach  = False
                self._nav_state           = self._NAV_IDLE
                self._drive_start_x       = None
                self._drive_start_y       = None
                self._drive_heading       = None
                self._turn_start_yaw      = None
                self._nav_wait_deadline   = 0.0
                self._nav_turnaround_done = False
                self.vel_pub.publish(Twist())
                changed_box   = None
                ref_cx        = None
                target_num    = 1
                result_text   = "Reset. Go to Card 1, press S to save design."
                result_color  = (0, 220, 255)
                print("  Reset.")

            # ── A: start/stop auto navigation to target card ─────────────────
            elif key == ord('a'):
                if self._nav_state != self._NAV_IDLE:
                    self._nav_state        = self._NAV_IDLE
                    self._nav_first_approach = False
                    self._turn_start_yaw   = None
                    self._drive_start_x    = None
                    self._drive_start_y    = None
                    self._drive_heading    = None
                    self.vel_pub.publish(Twist())
                    result_text  = "AUTO NAV stopped."
                    result_color = (0, 220, 255)
                    print("  [NAV] Stopped by user.")
                elif target_num == 1 and not designs:
                    # ── Initial approach ──────────────────────────────────────
                    # Robot starts facing along the row. Move forward one step,
                    # then turn left 90° to face card 1 for scanning.
                    self._nav_first_approach = True
                    self._nav_state      = self._NAV_DRIVE_FWD
                    self._nav_start_t    = time.time()
                    self._nav_lost_card  = False
                    self._drive_start_x  = None
                    self._drive_start_y  = None
                    self._drive_heading  = None   # will fall back to current yaw (no prior turn)
                    self._turn_start_yaw = None
                    result_text  = "INITIAL APPROACH → Card 1  (drive forward + turn left)"
                    result_color = (0, 255, 255)
                    print("  [NAV] Initial approach to Card 1 — driving forward then turning left.")
                else:
                    # ── Normal navigation (cards 2, 3, …) ────────────────────
                    # Turn right 90° → drive along row → turn left 90° → align
                    self._nav_first_approach = False
                    self._nav_state      = self._NAV_TURN_RIGHT
                    self._nav_start_t    = time.time()
                    self._nav_lost_card  = False
                    self._turn_start_yaw = self._yaw
                    result_text  = "AUTO NAV → Card {}  (step drive)".format(target_num)
                    result_color = (0, 255, 255)
                    print("  [NAV] Moving to Card {} — step drive, then align.".format(target_num))

            # ── N: next target card ───────────────────────────────────────────
            elif key == ord('n'):
                target_num  += 1
                result_text  = "Target → Card {}  (press A to go there)".format(target_num)
                result_color = (0, 255, 255)
                print("  [NAV] Target set to Card {}.".format(target_num))

            # ── B: begin return journey (go left, auto-check each card) ─────
            elif key == ord('b'):
                if self._nav_state != self._NAV_IDLE:
                    self._nav_state      = self._NAV_IDLE
                    self._nav_going_left = False
                    self._turn_start_yaw = None
                    self._drive_start_x  = None
                    self._drive_start_y  = None
                    self._drive_heading  = None
                    self.vel_pub.publish(Twist())
                    result_text  = "Return journey stopped."
                    result_color = (0, 220, 255)
                elif not designs:
                    result_text  = "No designs saved yet — scan cards first!"
                    result_color = (0, 0, 255)
                else:
                    # Start from the LAST saved card and go left
                    last_card = max(designs.keys())
                    target_num = last_card - 1   # first destination on return
                    self._nav_going_left  = True
                    self._nav_state       = self._NAV_TURN_LEFT   # FIRST turn on return is LEFT
                    self._nav_start_t     = time.time()
                    self._turn_start_yaw  = self._yaw
                    self._drive_start_x   = None
                    self._drive_start_y   = None
                    self._drive_heading   = None
                    result_text  = "RETURN: checking Cards {} → 1...".format(last_card - 1)
                    result_color = (0, 255, 255)
                    print("  [RETURN] Starting return from Card {}.".format(last_card))

            # ── L: line up — rotate until card is on centre line ─────────────
            elif key == ord('l'):
                if self._nav_state != self._NAV_IDLE:
                    self._nav_state = self._NAV_IDLE
                    self._turn_start_yaw = None
                    self._drive_start_x  = None
                    self._drive_start_y  = None
                    self._drive_heading  = None
                    self.vel_pub.publish(Twist())
                    result_text  = "Line-up stopped."
                    result_color = (0, 220, 255)
                elif smap:
                    self._nav_state   = self._NAV_LINE_UP
                    self._nav_start_t = time.time()
                    self._nav_waiting = False   # clear pause so state machine runs
                    result_text  = "Lining up to Card {}...".format(target_num)
                    result_color = (0, 255, 255)
                    print("  [LINE UP] Rotating to align Card {} with centre line.".format(target_num))
                else:
                    result_text  = "No cards visible."
                    result_color = (0, 0, 255)

            # ── T: test turn — bypass state machine, just spin for 3s ──────────
            elif key == ord('t'):
                print("  [TEST] Sending turn right for 3 seconds...")
                test_cmd = Twist()
                test_cmd.angular.z = -0.5
                deadline = time.time() + 3.0
                while time.time() < deadline and self._running:
                    self.vel_pub.publish(test_cmd)
                    cv2.waitKey(50)
                self.vel_pub.publish(Twist())
                print("  [TEST] Done.")

            # ── Q: quit ───────────────────────────────────────────────────────
            elif key == ord('q'):
                self.vel_pub.publish(Twist())       # stop robot before quitting
                self._running = False
                break



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
    