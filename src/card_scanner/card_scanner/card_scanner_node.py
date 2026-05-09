#!/usr/bin/env python3
"""
TurtleBot3 Burger - Playing Card Change Detector
SMART COMPARISON:
  - Stores the actual IMAGE of each card from Scan 1
  - Compares pixel-level appearance in Scan 2
  - Detects: card swapped, card flipped, card replaced
  - Works with ANY number of cards
FLOW:
  Scan 1 (store images) -> rotate 180 -> change card
  -> rotate back 180 -> Scan 2 -> compare images -> report
"""

import rclpy
from rclpy.node import Node
from geometry_msgs.msg import Twist
from sensor_msgs.msg import Image
from cv_bridge import CvBridge

import cv2
import numpy as np
import time
import math
import threading


# ═══════════════════════════════════════════════════════════════════════════
#  CARD DETECTOR
# ═══════════════════════════════════════════════════════════════════════════

class CardDetector:
    CARD_W = 200
    CARD_H = 300

    def detect(self, frame):
        h, w  = frame.shape[:2]
        total = h * w
        self.MIN_AREA = int(total * 0.015)
        self.MAX_AREA = int(total * 0.35)

        best = []
        for fn in [self._method_white,
                   self._method_hsv,
                   self._method_edges]:
            result = fn(frame)
            if len(result) > len(best):
                best = result
        return best

    def _method_white(self, frame):
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        blur = cv2.GaussianBlur(gray, (5, 5), 0)
        best = []
        for tval in [120, 130, 140, 150]:
            _, th = cv2.threshold(blur, tval, 255, cv2.THRESH_BINARY)
            cards = self._extract(th, frame)
            if len(cards) > len(best):
                best = cards
        return best

    def _method_hsv(self, frame):
        hsv  = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
        mask = cv2.inRange(hsv, (0, 0, 150), (180, 70, 255))
        k    = np.ones((5, 5), np.uint8)
        mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, k, iterations=3)
        mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN,  k, iterations=1)
        return self._extract(mask, frame)

    def _method_edges(self, frame):
        gray  = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        blur  = cv2.GaussianBlur(gray, (5, 5), 0)
        edges = cv2.Canny(blur, 20, 60)
        k     = np.ones((9, 9), np.uint8)
        edges = cv2.dilate(edges, k, iterations=3)
        filled = edges.copy()
        msk    = np.zeros((filled.shape[0]+2, filled.shape[1]+2), np.uint8)
        cv2.floodFill(filled, msk, (0, 0), 255)
        filled = cv2.bitwise_not(filled)
        filled = cv2.bitwise_or(filled, edges)
        return self._extract(filled, frame)

    def _extract(self, mask, frame):
        k    = np.ones((3, 3), np.uint8)
        mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, k, iterations=2)
        cnts, _ = cv2.findContours(
            mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        fh, fw   = frame.shape[:2]
        gray_frm = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        frm_mean = np.mean(gray_frm)

        cards = []
        for cnt in cnts:
            area = cv2.contourArea(cnt)
            if not (self.MIN_AREA < area < self.MAX_AREA):
                continue
            x, y, bw, bh = cv2.boundingRect(cnt)
            mg = 2
            if x <= mg or y <= mg or x+bw >= fw-mg or y+bh >= fh-mg:
                continue
            if bh == 0:
                continue
            asp = bw / float(bh)
            if not (0.45 < asp < 2.0):
                continue
            hull_area = cv2.contourArea(cv2.convexHull(cnt))
            if hull_area == 0 or area / hull_area < 0.70:
                continue
            roi_mean = np.mean(gray_frm[y:y+bh, x:x+bw])
            if roi_mean < frm_mean * 0.85:
                continue

            # Warp card to standard size for comparison
            pts    = np.array([[x,y],[x+bw,y],[x+bw,y+bh],[x,y+bh]],
                               dtype="float32")
            warped = self._warp(frame, pts)

            # Also crop raw region for pixel comparison
            crop = frame[y:y+bh, x:x+bw].copy()

            rank = self._rank(warped)
            suit = self._suit(warped)

            cards.append({
                "rank":           rank,
                "suit":           suit,
                "label":          "{} of {}".format(rank, suit),
                "bbox":           (x, y, bw, bh),
                "area":           area,
                "warped":         warped,   # standard-size card image
                "crop":           crop,     # raw crop for pixel diff
                "position_index": 0,
            })

        cards = self._dedup(cards)
        cards.sort(key=lambda c: c["bbox"][0])
        for i, c in enumerate(cards):
            c["position_index"] = i + 1
        return cards

    def _dedup(self, cards, thr=0.35):
        cards.sort(key=lambda c: c["area"], reverse=True)
        keep = []
        for card in cards:
            x1,y1,w1,h1 = card["bbox"]
            skip = False
            for k in keep:
                x2,y2,w2,h2 = k["bbox"]
                ix = max(0, min(x1+w1,x2+w2) - max(x1,x2))
                iy = max(0, min(y1+h1,y2+h2) - max(y1,y2))
                inter = ix*iy
                union = w1*h1 + w2*h2 - inter
                if union > 0 and inter/union > thr:
                    skip = True; break
            if not skip:
                keep.append(card)
        return keep

    def _warp(self, img, pts):
        dst = np.array([[0,0],[self.CARD_W-1,0],
                        [self.CARD_W-1,self.CARD_H-1],[0,self.CARD_H-1]],
                       dtype="float32")
        return cv2.warpPerspective(
            img, cv2.getPerspectiveTransform(pts, dst),
            (self.CARD_W, self.CARD_H))

    def _suit(self, warped):
        h, w  = warped.shape[:2]
        roi   = warped[h//4:3*h//4, w//4:3*w//4]
        hsv   = cv2.cvtColor(roi, cv2.COLOR_BGR2HSV)
        red1  = cv2.inRange(hsv, (0,50,50),(10,255,255))
        red2  = cv2.inRange(hsv, (160,50,50),(180,255,255))
        nr    = cv2.countNonZero(cv2.bitwise_or(red1,red2))
        nd    = cv2.countNonZero(cv2.inRange(hsv,(0,0,0),(180,60,90)))
        tot   = roi.shape[0]*roi.shape[1]
        if nr > tot*0.015: return self._red_suit(warped)
        if nd > tot*0.015: return self._black_suit(warped)
        return "Unknown"

    def _red_suit(self, w):
        c = cv2.cvtColor(w[5:50,2:28], cv2.COLOR_BGR2GRAY)
        _,bw = cv2.threshold(c,127,255,cv2.THRESH_BINARY_INV)
        cnts,_ = cv2.findContours(bw,cv2.RETR_EXTERNAL,cv2.CHAIN_APPROX_SIMPLE)
        if not cnts: return "Hearts"
        cnt = max(cnts,key=cv2.contourArea)
        ha  = cv2.contourArea(cv2.convexHull(cnt))
        sol = cv2.contourArea(cnt)/ha if ha>0 else 0
        return "Diamonds" if sol>0.88 else "Hearts"

    def _black_suit(self, w):
        c = cv2.cvtColor(w[5:50,2:28], cv2.COLOR_BGR2GRAY)
        _,bw = cv2.threshold(c,127,255,cv2.THRESH_BINARY_INV)
        cnts,_ = cv2.findContours(bw,cv2.RETR_EXTERNAL,cv2.CHAIN_APPROX_SIMPLE)
        if not cnts: return "Spades"
        cnt  = max(cnts,key=cv2.contourArea)
        peri = cv2.arcLength(cnt,True)
        app  = cv2.approxPolyDP(cnt,0.04*peri,True)
        return "Clubs" if len(app)>7 else "Spades"

    def _rank(self, warped):
        roi  = warped[3:48,2:30]
        gray = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
        try:
            import pytesseract
            _,bw = cv2.threshold(gray,0,255,
                                 cv2.THRESH_BINARY_INV+cv2.THRESH_OTSU)
            bw   = cv2.resize(bw,None,fx=4,fy=4,
                              interpolation=cv2.INTER_CUBIC)
            bw   = cv2.morphologyEx(bw,cv2.MORPH_CLOSE,np.ones((2,2),np.uint8))
            cfg  = "--psm 10 --oem 3 -c tessedit_char_whitelist=A23456789JQK10"
            txt  = pytesseract.image_to_string(bw,config=cfg).strip().upper()
            txt  = txt.replace(" ","").replace("\n","")
            for v in ["10","A","2","3","4","5","6","7","8","9","J","Q","K"]:
                if v in txt: return v
        except ImportError:
            pass
        _,bw   = cv2.threshold(gray,0,255,
                               cv2.THRESH_BINARY_INV+cv2.THRESH_OTSU)
        cnts,_ = cv2.findContours(bw,cv2.RETR_EXTERNAL,
                                  cv2.CHAIN_APPROX_SIMPLE)
        sig = [c for c in cnts if cv2.contourArea(c)>6]
        if not sig:     return "?"
        if len(sig)==2: return "10"
        c = max(sig,key=cv2.contourArea)
        _,_,cw,ch = cv2.boundingRect(c)
        asp = cw/float(ch) if ch>0 else 0
        if asp>0.85: return "A"
        if asp<0.38: return "1"
        return "?"

    def annotate(self, frame, cards):
        out = frame.copy()
        for card in cards:
            x,y,w,h = card["bbox"]
            cv2.rectangle(out,(x,y),(x+w,y+h),(0,255,0),2)
            lbl = "Pos {}".format(card["position_index"])
            (tw,th),_ = cv2.getTextSize(lbl,cv2.FONT_HERSHEY_SIMPLEX,0.5,1)
            ty = max(y-4, th+4)
            cv2.rectangle(out,(x,ty-th-2),(x+tw+4,ty+2),(0,0,0),-1)
            cv2.putText(out,lbl,(x+2,ty),
                        cv2.FONT_HERSHEY_SIMPLEX,0.5,(0,255,0),1)
        return out


# ═══════════════════════════════════════════════════════════════════════════
#  CARD MEMORY  —  stores Scan 1 card images for comparison
# ═══════════════════════════════════════════════════════════════════════════

class CardMemory:
    """
    Stores warped card images from Scan 1.
    Compares each position in Scan 2 using:
      1. Pixel difference (catches flipped cards)
      2. Histogram difference (catches replaced cards)
      3. Label difference (catches suit/rank changes if OCR works)
    A card is marked CHANGED if ANY of these differ beyond threshold.
    """

    COMPARE_SIZE = (128, 128)   # normalize all cards to this size
    FLIP_THRESHOLD   = 20.0     # mean pixel diff to call it "changed"
    HIST_THRESHOLD   = 0.15     # histogram correlation drop threshold

    def __init__(self):
        self.cards = {}   # position_index -> card dict with image

    def store(self, cards):
        """Save all cards from Scan 1."""
        self.cards = {}
        for c in cards:
            pos = c["position_index"]
            self.cards[pos] = {
                "label":   c["label"],
                "rank":    c["rank"],
                "suit":    c["suit"],
                "warped":  c["warped"].copy(),
                "thumb":   cv2.resize(c["warped"], self.COMPARE_SIZE),
                "hist":    self._histogram(c["warped"]),
            }
            # Save card image to disk for debugging
            cv2.imwrite("/tmp/scan1_card{}.jpg".format(pos), c["warped"])
        print()
        print("  [Memory] Stored {} card(s) from Scan 1:".format(len(cards)))
        for pos, data in sorted(self.cards.items()):
            print("    Position {:>2}: {}".format(pos, data["label"]))

    def compare(self, scan2_cards):
        """
        Compare scan2 cards against stored scan1 cards.
        Returns list of change dicts.
        """
        changes = []
        all_pos = sorted(
            set(self.cards.keys()) |
            {c["position_index"] for c in scan2_cards})

        map2 = {c["position_index"]: c for c in scan2_cards}

        for pos in all_pos:
            mem   = self.cards.get(pos)    # scan1 card at this position
            card2 = map2.get(pos)          # scan2 card at this position

            # Position existed before but is now empty
            if mem is not None and card2 is None:
                changes.append({
                    "position":    pos,
                    "type":        "removed",
                    "before":      mem["label"],
                    "after":       "NO CARD",
                    "pixel_diff":  999,
                    "hist_diff":   999,
                })
                continue

            # New card appeared at position that was empty before
            if mem is None and card2 is not None:
                changes.append({
                    "position":    pos,
                    "type":        "added",
                    "before":      "NO CARD",
                    "after":       card2["label"],
                    "pixel_diff":  999,
                    "hist_diff":   999,
                })
                continue

            if mem is None or card2 is None:
                continue

            # Save scan2 card image for debugging
            cv2.imwrite("/tmp/scan2_card{}.jpg".format(pos), card2["warped"])

            # ── Pixel difference ──────────────────────────────────────────
            thumb1 = mem["thumb"]
            thumb2 = cv2.resize(card2["warped"], self.COMPARE_SIZE)

            diff_normal  = self._pixel_diff(thumb1, thumb2)
            rotated2     = cv2.rotate(thumb2, cv2.ROTATE_180)
            diff_flipped = self._pixel_diff(thumb1, rotated2)

            # A flipped card: diff_normal is HIGH, diff_flipped is LOW.
            # Use diff_normal to decide if anything changed; use diff_flipped
            # only to classify the change type.
            pixel_diff  = diff_normal
            was_flipped = (diff_flipped < diff_normal * 0.7 and
                           diff_flipped < self.FLIP_THRESHOLD)

            # ── Histogram difference ──────────────────────────────────────
            hist2      = self._histogram(card2["warped"])
            hist_corr  = cv2.compareHist(mem["hist"], hist2,
                                          cv2.HISTCMP_CORREL)
            hist_diff  = 1.0 - hist_corr   # 0=same, 1=totally different

            # ── Label difference ──────────────────────────────────────────
            label_changed = (mem["label"] != card2["label"] and
                             "?" not in mem["label"] and
                             "?" not in card2["label"])

            # ── Decision ─────────────────────────────────────────────────
            pixel_changed = pixel_diff >= self.FLIP_THRESHOLD
            hist_changed  = hist_diff  >= self.HIST_THRESHOLD

            changed = was_flipped or pixel_changed or hist_changed or label_changed

            if changed:
                if was_flipped:
                    change_type = "flipped"
                elif label_changed:
                    change_type = "replaced"
                else:
                    change_type = "changed"

                changes.append({
                    "position":    pos,
                    "type":        change_type,
                    "before":      mem["label"],
                    "after":       card2["label"],
                    "pixel_diff":  round(pixel_diff, 1),
                    "hist_diff":   round(hist_diff, 3),
                    "was_flipped": was_flipped,
                })

        return changes

    def _pixel_diff(self, img1, img2):
        """Mean absolute pixel difference between two same-size images."""
        g1   = cv2.cvtColor(img1, cv2.COLOR_BGR2GRAY).astype(np.float32)
        g2   = cv2.cvtColor(img2, cv2.COLOR_BGR2GRAY).astype(np.float32)
        diff = np.abs(g1 - g2)
        return float(np.mean(diff))

    def _histogram(self, img):
        """Compute HSV histogram for color-based comparison."""
        hsv  = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
        hist = cv2.calcHist([hsv], [0, 1], None,
                             [18, 16], [0, 180, 0, 256])
        cv2.normalize(hist, hist)
        return hist


# ═══════════════════════════════════════════════════════════════════════════
#  ROS2 NODE
# ═══════════════════════════════════════════════════════════════════════════

class CardScannerNode(Node):

    def __init__(self):
        super().__init__("card_scanner_node")

        self.declare_parameter("camera_topic",  "/camera/image_raw")
        self.declare_parameter("cmd_vel_topic", "/cmd_vel")
        self.declare_parameter("rotate_speed",  1.2)
        self.declare_parameter("scan_frames",   20)

        cam       = self.get_parameter("camera_topic").value
        vel       = self.get_parameter("cmd_vel_topic").value
        self.spd  = self.get_parameter("rotate_speed").value
        self.nf   = self.get_parameter("scan_frames").value
        self.dur  = math.pi / self.spd

        self.bridge    = CvBridge()
        self.detector  = CardDetector()
        self.memory    = CardMemory()
        self.lock      = threading.Lock()
        self.latest_frame = None

        self.vel_pub = self.create_publisher(Twist, vel, 10)
        self.img_sub = self.create_subscription(
            Image, cam, self._img_cb, 10)

        self.get_logger().info("Card Scanner Node ready!")
        threading.Thread(target=self._game_loop, daemon=True).start()

    def _img_cb(self, msg):
        try:
            frame = self.bridge.imgmsg_to_cv2(msg, "bgr8")
            with self.lock:
                self.latest_frame = frame
        except Exception as e:
            self.get_logger().error("Image error: {}".format(e))

    # ── Game loop ─────────────────────────────────────────────────────────────

    def _game_loop(self):
        self._banner()
        self._wait_camera()

        # ── SCAN 1 ────────────────────────────────────────────────────────────
        input("  >> Press ENTER to do SCAN 1 (initial scan)...")
        print()
        print("  Scanning cards... hold still.")
        scan1 = self._capture(label="scan1")
        self._print_cards("SCAN 1 — Initial Position", scan1)

        if not scan1:
            print("  WARNING: No cards detected!")
            print("  Check /tmp/scan1_result.jpg")
            if input("  Retry? (y/n): ").strip().lower() == "y":
                self._game_loop()
                return

        # Store card images in memory for later comparison
        self.memory.store(scan1)

        # ── ROTATE FORWARD 180 ────────────────────────────────────────────────
        print()
        print("  ─────────────────────────────────────────────────")
        input("  >> Press ENTER — robot rotates 180 degrees...")
        print()
        print("  Rotating 180 degrees...")
        self._rotate(+1)
        print("  Done!")
        print()
        print("  ┌─────────────────────────────────────────────────┐")
        print("  │  CHANGE ONE CARD NOW:                           │")
        print("  │  - Swap it with a different card, OR            │")
        print("  │  - Flip it upside down (180 degrees)            │")
        print("  └─────────────────────────────────────────────────┘")
        print()

        # ── ROTATE BACK 180 ───────────────────────────────────────────────────
        input("  >> Press ENTER when done — robot rotates BACK 180...")
        print()
        print("  Rotating back 180 degrees...")
        self._rotate(-1)
        print("  Robot back to original position!")
        print()

        # ── SCAN 2 ────────────────────────────────────────────────────────────
        print("  Scanning cards again... hold still.")
        scan2 = self._capture(label="scan2")
        self._print_cards("SCAN 2 — After Change", scan2)

        # ── COMPARE WITH MEMORY ───────────────────────────────────────────────
        changes = self.memory.compare(scan2)
        self._report(scan1, scan2, changes)

        # ── Replay ────────────────────────────────────────────────────────────
        if input("  >> Play again? (y/n): ").strip().lower() == "y":
            self._game_loop()
        else:
            print("  Goodbye!")
            rclpy.shutdown()

    # ── Camera ────────────────────────────────────────────────────────────────

    def _wait_camera(self, timeout=30.0):
        print("  Waiting for camera...")
        cv2.namedWindow("TurtleBot3 Camera", cv2.WINDOW_NORMAL)
        cv2.resizeWindow("TurtleBot3 Camera", 640, 480)
        t0 = time.time()
        while time.time() - t0 < timeout:
            with self.lock:
                frame = self.latest_frame.copy() if self.latest_frame is not None else None
            if frame is not None:
                cv2.imshow("TurtleBot3 Camera", frame)
                cv2.waitKey(1)
                print("  Camera ready!")
                print()
                return
            time.sleep(0.05)
        print("  WARNING: Camera timeout — is ROS_DOMAIN_ID=15 set?")
        print()

    def _capture(self, label="scan"):
        """Capture frames, detect cards, return most consistent result."""
        results    = []
        last_frame = None
        n          = 0

        while n < self.nf:
            with self.lock:
                frame = self.latest_frame.copy() \
                    if self.latest_frame is not None else None
            if frame is None:
                time.sleep(0.1)
                continue
            cards = self.detector.detect(frame)
            results.append(cards)
            last_frame = frame
            n += 1
            # Show live camera window with detections
            ann = self.detector.annotate(frame, cards)
            cv2.putText(ann, "Scanning {}/{}".format(n, self.nf),
                        (5, 15), cv2.FONT_HERSHEY_SIMPLEX,
                        0.45, (0, 255, 255), 1)
            cv2.imshow("TurtleBot3 Camera", ann)
            cv2.waitKey(1)
            time.sleep(0.12)

        # Save annotated debug image
        if last_frame is not None:
            best = max(results, key=len)
            ann  = self.detector.annotate(last_frame, best)
            cv2.imwrite("/tmp/{}_result.jpg".format(label), ann)
            print("  Debug: /tmp/{}_result.jpg".format(label))

        counts     = [len(r) for r in results]
        best_n     = max(set(counts), key=counts.count)
        candidates = [r for r in results if len(r) == best_n]
        return candidates[len(candidates) // 2]

    # ── Robot ─────────────────────────────────────────────────────────────────

    def _rotate(self, direction):
        """
        Rotate exactly 180 degrees.
        Sends velocity for calculated duration, then stops cleanly.
        direction = +1 : forward 180
        direction = -1 : back 180 (same duration, opposite sign)
        """
        twist = Twist()
        twist.angular.z = float(self.spd * direction)

        # Publish at 20Hz for precise timing
        dt      = 0.05
        steps   = int(self.dur / dt)
        for _ in range(steps):
            self.vel_pub.publish(twist)
            time.sleep(dt)

        # Send zero velocity multiple times to ensure full stop
        stop = Twist()
        for _ in range(10):
            self.vel_pub.publish(stop)
            time.sleep(0.05)

        # Extra settle time for robot to physically stop
        time.sleep(1.0)

    # ── Result reporter ───────────────────────────────────────────────────────

    def _report(self, scan1, scan2, changes):
        print()
        print("  ================================")
        print("       RESULT")
        print("  ================================")
        print()
        print("  Total cards seen: {}".format(len(scan1)))
        print()

        if not changes:
            print("  No change detected.")
        else:
            for ch in changes:
                ctype = ch.get("type", "changed").upper()
                print("  >>> POSITION {} — {} <<<".format(ch["position"], ctype))
                print("      Before: {}   After: {}".format(
                    ch["before"], ch["after"]))
                if "pixel_diff" in ch and ch["pixel_diff"] != 999:
                    print("      pixel_diff={:.1f}  hist_diff={:.3f}".format(
                        ch["pixel_diff"], ch["hist_diff"]))
            print()
            if len(changes) == 1:
                ch = changes[0]
                print("  ANSWER: Position {} was {}!".format(
                    ch["position"], ch.get("type", "changed").upper()))
            else:
                parts = ["{} (pos {})".format(
                    c.get("type", "changed").upper(), c["position"])
                    for c in changes]
                print("  ANSWER: {}".format(", ".join(parts)))

        print()
        print("  ================================")
        print()


    def _banner(self):
        print()
        print("  ════════════════════════════════════════════════")
        print("    TurtleBot3 Card Change Detector")
        print("    Detects: swapped cards AND flipped cards!")
        print("  ════════════════════════════════════════════════")
        print("  1. Place cards face-up in front of robot")
        print("  2. ENTER -> Scan 1 (card images stored in memory)")
        print("  3. ENTER -> Robot rotates 180 degrees")
        print("  4. Change ONE card (swap it OR flip it 180 deg)")
        print("  5. ENTER -> Robot rotates back 180 degrees")
        print("  6. Scan 2 runs automatically")
        print("  7. Robot compares pixel-by-pixel with memory")
        print("  8. Announces: which position changed + how!")
        print("  ════════════════════════════════════════════════")
        print()

    def _print_cards(self, title, cards):
        print()
        print("  --- {} ---".format(title))
        if cards:
            positions = [str(c["position_index"]) for c in cards]
            print("  Cards found at positions: {}".format(
                ", ".join(positions)))
            print("  Total: {}".format(len(cards)))
        else:
            print("  No cards detected!")
        print()


def main(args=None):
    rclpy.init(args=args)
    node = CardScannerNode()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()
        cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
