import cv2 as cv
import mediapipe as mp
mp_drawing = mp.solutions.drawing_utils
mp_drawing_styles = mp.solutions.drawing_styles

class Visualizer:
    def __init__(self):
        self.image = None

    def set_image(self,image,header):

        image[0:header.shape[0],0:header.shape[1],:] = header
        self.image = image


    def draw_line(self,image,points,color,thickness):
        if points[0][0] != 0 and points[0][1] != 0 and points[1][0] != 0 and points[1][1] != 0:
            cv.line(image, (points[0][0], points[0][1]), (points[1][0], points[1][1]), color, thickness)

        return image

    def undo_lines(self,image, lines):
        for line in lines:
            points, color, thickness = line
            if points[0][0] != 0 and points[0][1] != 0 and points[1][0] != 0 and points[1][1] != 0:
                cv.line(image, (points[0][0], points[0][1]), (points[1][0], points[1][1]), color, thickness)

        return image

    def draw_bounding_rect(self,use_brect,brect):
        if use_brect:
            # Outer rectangle
            cv.rectangle(self.image, (brect[0], brect[1]), (brect[2], brect[3]),
                         (0, 0, 0), 1)


    def draw_info_text(self,brect,handedness,hand_sign_text):
        cv.rectangle(self.image, (brect[0], brect[1]), (brect[2], brect[1] - 22),
                     (0, 0, 0), -1)

        info_text = handedness.classification[0].label[0:]
        if hand_sign_text != "":
            info_text = info_text + ':' + hand_sign_text
        cv.putText(self.image, info_text, (brect[0] + 5, brect[1] - 4),
                   cv.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1, cv.LINE_AA)


    def apply_mask(self,mask):
        maskGray = cv.cvtColor(mask, cv.COLOR_BGR2GRAY)
        _, maskInverted = cv.threshold(maskGray, 1, 255, cv.THRESH_BINARY_INV)
        maskInverted = cv.cvtColor(maskInverted, cv.COLOR_GRAY2BGR)
        self.image = cv.multiply(self.image, maskInverted // 255)  # background roi
        self.image = cv.add(self.image, mask)

    def draw_performance(self,fps):
        cv.putText(self.image, "FPS:" + str(fps), (10, 30), cv.FONT_HERSHEY_SIMPLEX,
                   1.0, (0, 0, 0), 4, cv.LINE_AA)
        cv.putText(self.image, "FPS:" + str(fps), (10, 30), cv.FONT_HERSHEY_SIMPLEX,
                   1.0, (255, 255, 255), 2, cv.LINE_AA)

    def image_saved(self):
        cv.putText(self.image, "SAVED", (self.image.shape[1]-150, 30), cv.FONT_HERSHEY_SIMPLEX,
                   1.0, (0, 0, 0), 4, cv.LINE_AA)
        cv.putText(self.image, "SAVED", (self.image.shape[1]-150, 30), cv.FONT_HERSHEY_SIMPLEX,
                   1.0, (255, 255, 255), 2, cv.LINE_AA)




    def draw_landmarks(self,landmark_point):
        if len(landmark_point) > 0:
            # Thumb
            cv.line(self.image, tuple(landmark_point[2]), tuple(landmark_point[3]),
                    (0, 0, 0), 6)
            cv.line(self.image, tuple(landmark_point[2]), tuple(landmark_point[3]),
                    (255, 255, 255), 2)
            cv.line(self.image, tuple(landmark_point[3]), tuple(landmark_point[4]),
                    (0, 0, 0), 6)
            cv.line(self.image, tuple(landmark_point[3]), tuple(landmark_point[4]),
                    (255, 255, 255), 2)

            # Index finger
            cv.line(self.image, tuple(landmark_point[5]), tuple(landmark_point[6]),
                    (0, 0, 0), 6)
            cv.line(self.image, tuple(landmark_point[5]), tuple(landmark_point[6]),
                    (255, 255, 255), 2)
            cv.line(self.image, tuple(landmark_point[6]), tuple(landmark_point[7]),
                    (0, 0, 0), 6)
            cv.line(self.image, tuple(landmark_point[6]), tuple(landmark_point[7]),
                    (255, 255, 255), 2)
            cv.line(self.image, tuple(landmark_point[7]), tuple(landmark_point[8]),
                    (0, 0, 0), 6)
            cv.line(self.image, tuple(landmark_point[7]), tuple(landmark_point[8]),
                    (255, 255, 255), 2)

            # Middle finger
            cv.line(self.image, tuple(landmark_point[9]), tuple(landmark_point[10]),
                    (0, 0, 0), 6)
            cv.line(self.image, tuple(landmark_point[9]), tuple(landmark_point[10]),
                    (255, 255, 255), 2)
            cv.line(self.image, tuple(landmark_point[10]), tuple(landmark_point[11]),
                    (0, 0, 0), 6)
            cv.line(self.image, tuple(landmark_point[10]), tuple(landmark_point[11]),
                    (255, 255, 255), 2)
            cv.line(self.image, tuple(landmark_point[11]), tuple(landmark_point[12]),
                    (0, 0, 0), 6)
            cv.line(self.image, tuple(landmark_point[11]), tuple(landmark_point[12]),
                    (255, 255, 255), 2)

            # Ring finger
            cv.line(self.image, tuple(landmark_point[13]), tuple(landmark_point[14]),
                    (0, 0, 0), 6)
            cv.line(self.image, tuple(landmark_point[13]), tuple(landmark_point[14]),
                    (255, 255, 255), 2)
            cv.line(self.image, tuple(landmark_point[14]), tuple(landmark_point[15]),
                    (0, 0, 0), 6)
            cv.line(self.image, tuple(landmark_point[14]), tuple(landmark_point[15]),
                    (255, 255, 255), 2)
            cv.line(self.image, tuple(landmark_point[15]), tuple(landmark_point[16]),
                    (0, 0, 0), 6)
            cv.line(self.image, tuple(landmark_point[15]), tuple(landmark_point[16]),
                    (255, 255, 255), 2)

            # Little finger
            cv.line(self.image, tuple(landmark_point[17]), tuple(landmark_point[18]),
                    (0, 0, 0), 6)
            cv.line(self.image, tuple(landmark_point[17]), tuple(landmark_point[18]),
                    (255, 255, 255), 2)
            cv.line(self.image, tuple(landmark_point[18]), tuple(landmark_point[19]),
                    (0, 0, 0), 6)
            cv.line(self.image, tuple(landmark_point[18]), tuple(landmark_point[19]),
                    (255, 255, 255), 2)
            cv.line(self.image, tuple(landmark_point[19]), tuple(landmark_point[20]),
                    (0, 0, 0), 6)
            cv.line(self.image, tuple(landmark_point[19]), tuple(landmark_point[20]),
                    (255, 255, 255), 2)

            # Palm
            cv.line(self.image, tuple(landmark_point[0]), tuple(landmark_point[1]),
                    (0, 0, 0), 6)
            cv.line(self.image, tuple(landmark_point[0]), tuple(landmark_point[1]),
                    (255, 255, 255), 2)
            cv.line(self.image, tuple(landmark_point[1]), tuple(landmark_point[2]),
                    (0, 0, 0), 6)
            cv.line(self.image, tuple(landmark_point[1]), tuple(landmark_point[2]),
                    (255, 255, 255), 2)
            cv.line(self.image, tuple(landmark_point[2]), tuple(landmark_point[5]),
                    (0, 0, 0), 6)
            cv.line(self.image, tuple(landmark_point[2]), tuple(landmark_point[5]),
                    (255, 255, 255), 2)
            cv.line(self.image, tuple(landmark_point[5]), tuple(landmark_point[9]),
                    (0, 0, 0), 6)
            cv.line(self.image, tuple(landmark_point[5]), tuple(landmark_point[9]),
                    (255, 255, 255), 2)
            cv.line(self.image, tuple(landmark_point[9]), tuple(landmark_point[13]),
                    (0, 0, 0), 6)
            cv.line(self.image, tuple(landmark_point[9]), tuple(landmark_point[13]),
                    (255, 255, 255), 2)
            cv.line(self.image, tuple(landmark_point[13]), tuple(landmark_point[17]),
                    (0, 0, 0), 6)
            cv.line(self.image, tuple(landmark_point[13]), tuple(landmark_point[17]),
                    (255, 255, 255), 2)
            cv.line(self.image, tuple(landmark_point[17]), tuple(landmark_point[0]),
                    (0, 0, 0), 6)
            cv.line(self.image, tuple(landmark_point[17]), tuple(landmark_point[0]),
                    (255, 255, 255), 2)

        # Key Points
        for index, landmark in enumerate(landmark_point):
            if index == 0:  # 手首1
                cv.circle(self.image, (landmark[0], landmark[1]), 5, (255, 255, 255),
                          -1)
                cv.circle(self.image, (landmark[0], landmark[1]), 5, (0, 0, 0), 1)
            if index == 1:  # 手首2
                cv.circle(self.image, (landmark[0], landmark[1]), 5, (255, 255, 255),
                          -1)
                cv.circle(self.image, (landmark[0], landmark[1]), 5, (0, 0, 0), 1)
            if index == 2:  # 親指：付け根
                cv.circle(self.image, (landmark[0], landmark[1]), 5, (255, 255, 255),
                          -1)
                cv.circle(self.image, (landmark[0], landmark[1]), 5, (0, 0, 0), 1)
            if index == 3:  # 親指：第1関節
                cv.circle(self.image, (landmark[0], landmark[1]), 5, (255, 255, 255),
                          -1)
                cv.circle(self.image, (landmark[0], landmark[1]), 5, (0, 0, 0), 1)
            if index == 4:  # 親指：指先
                cv.circle(self.image, (landmark[0], landmark[1]), 8, (255, 255, 255),
                          -1)
                cv.circle(self.image, (landmark[0], landmark[1]), 8, (0, 0, 0), 1)
            if index == 5:  # 人差指：付け根
                cv.circle(self.image, (landmark[0], landmark[1]), 5, (255, 255, 255),
                          -1)
                cv.circle(self.image, (landmark[0], landmark[1]), 5, (0, 0, 0), 1)
            if index == 6:  # 人差指：第2関節
                cv.circle(self.image, (landmark[0], landmark[1]), 5, (255, 255, 255),
                          -1)
                cv.circle(self.image, (landmark[0], landmark[1]), 5, (0, 0, 0), 1)
            if index == 7:  # 人差指：第1関節
                cv.circle(self.image, (landmark[0], landmark[1]), 5, (255, 255, 255),
                          -1)
                cv.circle(self.image, (landmark[0], landmark[1]), 5, (0, 0, 0), 1)
            if index == 8:  # 人差指：指先
                cv.circle(self.image, (landmark[0], landmark[1]), 8, (255, 255, 255),
                          -1)
                cv.circle(self.image, (landmark[0], landmark[1]), 8, (0, 0, 0), 1)
            if index == 9:  # 中指：付け根
                cv.circle(self.image, (landmark[0], landmark[1]), 5, (255, 255, 255),
                          -1)
                cv.circle(self.image, (landmark[0], landmark[1]), 5, (0, 0, 0), 1)
            if index == 10:  # 中指：第2関節
                cv.circle(self.image, (landmark[0], landmark[1]), 5, (255, 255, 255),
                          -1)
                cv.circle(self.image, (landmark[0], landmark[1]), 5, (0, 0, 0), 1)
            if index == 11:  # 中指：第1関節
                cv.circle(self.image, (landmark[0], landmark[1]), 5, (255, 255, 255),
                          -1)
                cv.circle(self.image, (landmark[0], landmark[1]), 5, (0, 0, 0), 1)
            if index == 12:  # 中指：指先
                cv.circle(self.image, (landmark[0], landmark[1]), 8, (255, 255, 255),
                          -1)
                cv.circle(self.image, (landmark[0], landmark[1]), 8, (0, 0, 0), 1)
            if index == 13:  # 薬指：付け根
                cv.circle(self.image, (landmark[0], landmark[1]), 5, (255, 255, 255),
                          -1)
                cv.circle(self.image, (landmark[0], landmark[1]), 5, (0, 0, 0), 1)
            if index == 14:  # 薬指：第2関節
                cv.circle(self.image, (landmark[0], landmark[1]), 5, (255, 255, 255),
                          -1)
                cv.circle(self.image, (landmark[0], landmark[1]), 5, (0, 0, 0), 1)
            if index == 15:  # 薬指：第1関節
                cv.circle(self.image, (landmark[0], landmark[1]), 5, (255, 255, 255),
                          -1)
                cv.circle(self.image, (landmark[0], landmark[1]), 5, (0, 0, 0), 1)
            if index == 16:  # 薬指：指先
                cv.circle(self.image, (landmark[0], landmark[1]), 8, (255, 255, 255),
                          -1)
                cv.circle(self.image, (landmark[0], landmark[1]), 8, (0, 0, 0), 1)
            if index == 17:  # 小指：付け根
                cv.circle(self.image, (landmark[0], landmark[1]), 5, (255, 255, 255),
                          -1)
                cv.circle(self.image, (landmark[0], landmark[1]), 5, (0, 0, 0), 1)
            if index == 18:  # 小指：第2関節
                cv.circle(self.image, (landmark[0], landmark[1]), 5, (255, 255, 255),
                          -1)
                cv.circle(self.image, (landmark[0], landmark[1]), 5, (0, 0, 0), 1)
            if index == 19:  # 小指：第1関節
                cv.circle(self.image, (landmark[0], landmark[1]), 5, (255, 255, 255),
                          -1)
                cv.circle(self.image, (landmark[0], landmark[1]), 5, (0, 0, 0), 1)
            if index == 20:  # 小指：指先
                cv.circle(self.image, (landmark[0], landmark[1]), 8, (255, 255, 255),
                          -1)
                cv.circle(self.image, (landmark[0], landmark[1]), 8, (0, 0, 0), 1)