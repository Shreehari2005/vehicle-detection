
import math

class Tracker:
    def __init__(self):
        self.center_points = {}
        self.id_count = 0

    def update(self, objects_rect):
        objects_bbs_ids = []
        for rect in objects_rect:
            x1, y1, x2, y2 = rect
            cx, cy = (x1 + x2) // 2, (y1 + y2) // 2
            same_object_detected = False

            for obj_id, (px, py) in self.center_points.items():
                if math.hypot(cx - px, cy - py) < 50:
                    self.center_points[obj_id] = (cx, cy)
                    objects_bbs_ids.append([x1, y1, x2, y2, obj_id])
                    same_object_detected = True
                    break

            if not same_object_detected:
                self.center_points[self.id_count] = (cx, cy)
                objects_bbs_ids.append([x1, y1, x2, y2, self.id_count])
                self.id_count += 1

        self.center_points = {obj_id: pt for obj_id, pt in self.center_points.items() if obj_id in [obj[4] for obj in objects_bbs_ids]}
        return objects_bbs_ids
