import cv2


class VideoProcessor:
    """
    Watches a road video and only ACTS (crop -> ANPR -> OCR) on frames where a
    number plate appears CLEARLY. The caller plays the video back in its
    original form (no overlays); this class just decides when a plate is clear
    enough to process and keeps one best reading per tracked vehicle.

    "Clear" means the plate is:
      - detected with high confidence (>= clarity_conf),
      - large enough (>= min_plate_area), and
      - sharp / not motion-blurred (variance-of-Laplacian >= min_sharpness).

    De-duplication is by persistent track_id (ByteTrack). Among the clear
    frames for a vehicle we keep the highest  plate_conf * ocr_conf  reading.
    Only vehicles that produced at least one clear capture appear in results.
    """

    def __init__(self, anpr_detector, vehicle_classifier,
                 min_plate_area=600, clarity_conf=0.6, min_sharpness=80.0):
        self.anpr = anpr_detector
        self.vehicles = vehicle_classifier
        self.min_plate_area = min_plate_area
        self.clarity_conf = clarity_conf
        self.min_sharpness = min_sharpness
        self.tracks = {}  # track_id -> best record

    def _empty_record(self, track_id):
        return {
            'track_id': track_id,
            'vehicle_bbox': None,
            'vehicle_type': 'UNKNOWN',
            'vehicle_class': '',
            'vehicle_confidence': 0.0,
            'plate_bbox': None,
            'plate_crop': None,
            'raw_ocr': 'UNREADABLE',
            'final_plate': 'UNREADABLE',
            'plate_confidence': 0.0,
            'ocr_confidence': 0.0,
            'sharpness': 0.0,
            'score': -1.0,   # best plate_conf * ocr_conf so far
            'frames': 0,     # how many scanned frames this vehicle was seen in
        }

    @staticmethod
    def _sharpness(bgr_crop):
        """Variance of Laplacian — higher means sharper / less blurred."""
        gray = cv2.cvtColor(bgr_crop, cv2.COLOR_BGR2GRAY)
        return float(cv2.Laplacian(gray, cv2.CV_64F).var())

    def _is_clear(self, plate_conf, area, sharpness):
        return (
            plate_conf >= self.clarity_conf
            and area >= self.min_plate_area
            and sharpness >= self.min_sharpness
        )

    def scan_frame(self, frame):
        """
        Run tracking + clear-plate detection on a single BGR frame.

        The caller is responsible for displaying the original frame; this only
        ACTS when a plate is clearly captured.

        Returns:
            List of records that were captured/improved on this frame
            (empty if nothing clear was found).
        """
        captures = []
        vehicles = self.vehicles.track_vehicles(frame)

        for v in vehicles:
            tid = v['track_id']
            rec = self.tracks.get(tid)
            if rec is None:
                rec = self._empty_record(tid)
                self.tracks[tid] = rec

            rec['frames'] += 1
            rec['vehicle_bbox'] = v['bbox']
            if v['confidence'] > rec['vehicle_confidence']:
                rec['vehicle_type'] = v['vehicle_type']
                rec['vehicle_class'] = v['class_name']
                rec['vehicle_confidence'] = v['confidence']

            x1, y1, x2, y2 = v['bbox']
            vehicle_crop = frame[y1:y2, x1:x2]
            if vehicle_crop.size == 0:
                continue

            plates = self.anpr.detect_plates(vehicle_crop)
            if not plates:
                continue

            plate = max(plates, key=lambda p: p['confidence'])
            px1, py1, px2, py2 = plate['bbox']
            area = max(0, px2 - px1) * max(0, py2 - py1)

            fx1, fy1 = x1 + px1, y1 + py1
            fx2, fy2 = x1 + px2, y1 + py2
            plate_crop = frame[fy1:fy2, fx1:fx2]
            if plate_crop.size == 0:
                continue

            sharpness = self._sharpness(plate_crop)

            # CLARITY GATE — only act when the plate is clearly captured
            if not self._is_clear(plate['confidence'], area, sharpness):
                continue

            raw_ocr, final_plate, ocr_conf = self.anpr.perform_ocr_with_conf(plate_crop)
            score = plate['confidence'] * ocr_conf

            # Keep the highest-confidence clear reading for this vehicle
            if score > rec['score']:
                rec['score'] = score
                rec['plate_bbox'] = (fx1, fy1, fx2, fy2)
                rec['plate_crop'] = plate_crop.copy()
                rec['raw_ocr'] = raw_ocr
                rec['final_plate'] = final_plate
                rec['plate_confidence'] = plate['confidence']
                rec['ocr_confidence'] = ocr_conf
                rec['sharpness'] = sharpness
                captures.append(rec)

        return captures

    def get_results(self):
        """One record per vehicle that produced at least one clear capture."""
        results = []
        for tid in sorted(self.tracks.keys()):
            rec = self.tracks[tid]
            if rec['plate_crop'] is not None:
                results.append(rec)
        return results
