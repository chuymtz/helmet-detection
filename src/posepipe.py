import mediapipe as mp
import cv2
class SimplePose(object):
    def __init__(self, frame) -> None:
        self.frame = frame
        self.mpPose = mp.solutions.pose
        self.mpDraw = mp.solutions.drawing_utils
        self.poses = self.mpPose.Pose(static_image_mode=True, 
                                      model_complexity=0,
                                      enable_segmentation=False,
                                      min_detection_confidence=0.5,
                                      min_tracking_confidence=0.5)
        
    def get_results(self):
        self.img = cv2.cvtColor(self.frame, cv2.COLOR_BGR2RGB)
        self.results = self.poses.process(self.img)
    
    def draw_on_frame(self):
        if self.results.pose_landmarks:
            self.mpDraw.draw_landmarks(self)
