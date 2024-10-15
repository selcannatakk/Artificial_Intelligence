import mediapipe as mp
import cv2


class PoseDetector():
    def __init__(self):
        self.mp_draw = mp.solutions.drawing_utils
        self.mp_pose = mp.solutions.pose
        self.pose = self.mp_pose.Pose(self.image_mode)

    def get_pose(self, image, draw=True):

        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        self.results = self.pose.process(image)

        if self.results.pose_landmarks:
            if draw:
                self.mp_draw.draw_landmarks(image, self.results.pose_landmarks, self.mp_pose.POSE_CONNECTIONS)

        return image

    def get_points(self, image, draw=True):

        points = []
        if self.results.pose_landmarks:
            for id, lm in enumerate(self.results.pose_landmarks.landmark):
                height, width, _ = image.shape
                x, y = int(lm.x * width), int(lm.y * height)
                points.append([id, x, y])
                if draw:
                    cv2.circle(image, (x, y), 4, (0, 0, 0), cv2.FILLED)

        return points
