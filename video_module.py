import cv2


class VideoProcessor:
    """
    Processes a road video frame-by-frame and produces ONE result per physical
    vehicle.

    How de-duplication works:
      - Each vehicle is tracked with a persistent track_id (ByteTrack), so the
        same car across many frames maps to a single record.
      - "Balanced" speed: a vehicle's plate is only OCR'd when its plate crop is
        larger (closer) than any we have already OCR'd for that vehicle, so each
        car is OCR'd just a handful of times near its closest approach.
      - "Highest confidence" selection: among the frames we do OCR, we keep the
        reading with the best  plate_detection_conf * ocr_conf  score.

    Reuses the existing ANPRDetector and VehicleClassifier unchanged.
    """

    def __init__(self, anpr_detector, vehicle_classifier,
                 min_frames=2, min_plate_area=600):
        self.anpr = anpr_detector
        self.vehicles = vehicle_classifier
        self.min_frames = min_frames          # ignore tracks seen fewer times
        self.min_plate_area = min_plate_area  # skip tiny/far-away plates
        self.tracks = {}                      # track_id -> best record
        self.last_vehicles = []               # last tracked boxes (for in-between frames)

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
            'score': -1.0,      # best plate_conf * ocr_conf so far
            'ocr_max_area': 0,  # largest plate area we've OCR'd (balanced gate)
            'frames': 0,        # how many frames this vehicle was seen in
        }

    def process_frame(self, frame):
        """
        Run tracking + conditional OCR on a single BGR frame.
        Returns the annotated frame (BGR) for live preview.
        """
        vehicles = self.vehicles.track_vehicles(frame)
        self.last_vehicles = vehicles

        for v in vehicles:
            tid = v['track_id']
            rec = self.tracks.get(tid)
            if rec is None:
                rec = self._empty_record(tid)
                self.tracks[tid] = rec

            rec['frames'] += 1
            rec['vehicle_bbox'] = v['bbox']

            # Keep the strongest vehicle classification seen for this track
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

            # Strongest plate detection in this vehicle
            plate = max(plates, key=lambda p: p['confidence'])
            px1, py1, px2, py2 = plate['bbox']
            area = max(0, px2 - px1) * max(0, py2 - py1)
            if area < self.min_plate_area:
                continue

            # BALANCED gate: only OCR when this plate is closer (bigger) than any
            # we've already OCR'd for this vehicle.
            if area <= rec['ocr_max_area']:
                continue
            rec['ocr_max_area'] = area

            # Plate crop in full-frame coordinates
            fx1, fy1 = x1 + px1, y1 + py1
            fx2, fy2 = x1 + px2, y1 + py2
            plate_crop = frame[fy1:fy2, fx1:fx2]
            if plate_crop.size == 0:
                continue

            raw_ocr, final_plate, ocr_conf = self.anpr.perform_ocr_with_conf(plate_crop)
            score = plate['confidence'] * ocr_conf

            # HIGHEST-CONFIDENCE selection
            if score > rec['score']:
                rec['score'] = score
                rec['plate_bbox'] = (fx1, fy1, fx2, fy2)
                rec['plate_crop'] = plate_crop.copy()
                rec['raw_ocr'] = raw_ocr
                rec['final_plate'] = final_plate
                rec['plate_confidence'] = plate['confidence']
                rec['ocr_confidence'] = ocr_conf

        return self._draw_frame(frame, vehicles)

    def draw(self, frame):
        """
        Lightweight overlay for frames where ANPR did NOT run: re-draws the most
        recent tracked boxes/labels. Lets the output video stay at full FPS while
        ANPR runs at a lower rate.
        """
        return self._draw_frame(frame, self.last_vehicles)

    def _draw_frame(self, frame, vehicles):
        """Draw current-frame boxes with each vehicle's running-best plate."""
        annotated = frame.copy()
        for v in vehicles:
            tid = v['track_id']
            x1, y1, x2, y2 = v['bbox']
            rec = self.tracks.get(tid)
            plate = rec['final_plate'] if rec else ''

            if plate and plate not in ('INVALID', 'UNREADABLE'):
                color = (0, 255, 0)        # green = valid plate
            elif plate == 'INVALID':
                color = (0, 165, 255)      # orange
            else:
                color = (200, 200, 200)    # grey = no plate yet

            cv2.rectangle(annotated, (x1, y1), (x2, y2), color, 2)

            label = f"ID{tid} {v['vehicle_type']}"
            if plate and plate not in ('UNREADABLE',):
                label += f" {plate}"
            cv2.putText(
                annotated, label, (x1, max(20, y1 - 8)),
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2
            )
        return annotated

    def get_results(self):
        """Final de-duplicated list: one record per vehicle (track)."""
        results = []
        for tid in sorted(self.tracks.keys()):
            rec = self.tracks[tid]
            if rec['frames'] >= self.min_frames:
                results.append(rec)
        return results
