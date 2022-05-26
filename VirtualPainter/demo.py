from collections import deque
import cv2 as cv
import numpy as np
from omegaconf import DictConfig,OmegaConf
from utils import CvFpsCalc,calc_bounding_rect,calc_landmark_list, pre_process_landmark
from model import KeyPointClassifier
from visualizer import Visualizer
import mediapipe as mp
from copy import deepcopy
import csv
import datetime
#TODO: adauga si o functie de salvare poate un gest
#TODO: adauga si o functie de schimbare de header
class Demo:
    QUIT_KEYS = {27,ord('q')}

    def __init__(self,config: DictConfig):
        self.config = config
        self.hands_detection = self.load_hands_model()
        self.key_point_classifier = KeyPointClassifier(config)
        self.visualizer = Visualizer()
        self.cvFpsCalc = CvFpsCalc(buffer_len=10)
        self.key_point_classifier_labels = self.read_labels()
        self.header = self.load_header()


    def load_hands_model(self):
        mp_hands = mp.solutions.hands
        hands = mp_hands.Hands(
            static_image_mode= self.config.mediapipe.use_static_image_mode,
            max_num_hands= self.config.mediapipe.max_num_hands,
            min_detection_confidence= self.config.mediapipe.min_detection_confidence,
            min_tracking_confidence=self.config.mediapipe.min_tracking_confidence,
        )
        return hands

    def read_labels(self):
        # TODO: de scos la final citirea din csv, si puse direct in init labels
        # TODO: adauga o clasa unknown cand nu reuseste clasificarea de gesturi
        with open(self.config.gesture_classification.labels_path,
                  encoding='utf-8-sig') as f:
            keypoint_classifier_labels = csv.reader(f)
            keypoint_classifier_labels = [
                row[0] for row in keypoint_classifier_labels
            ]
            return keypoint_classifier_labels

    def load_header(self):

        header = cv.imread('header.png')
        header = cv.resize(header,(self.config.demo.cap_width // 2 ,self.config.demo.cap_height // 12))
        return header

    def save_drawing(self,img):
        dt = datetime.datetime.now()
        time = dt.strftime('%Y%m%d_%H%M%S')
        cv.imwrite(f'saved/drawing_{time}.png',img)




    def run(self):
        # Camera preparation ###############################################################
        cap = cv.VideoCapture(0)
        cap.set(cv.CAP_PROP_FRAME_WIDTH, self.config.demo.cap_width)
        cap.set(cv.CAP_PROP_FRAME_HEIGHT, self.config.demo.cap_height)

        # Coordinate history for UNDO function
        points = deque(maxlen=2)
        lines = []
        color = (255,255,0)
        thickness = 10
        mask = np.zeros((self.config.demo.cap_height, self.config.demo.cap_width, 3), dtype=np.uint8)
        visualizer = self.visualizer
        frame_counter = 0
        while True:
            fps = self.cvFpsCalc.get()

            # Press ESC or q to end
            key = cv.waitKey(10) & 0xff
            if key in self.QUIT_KEYS:
                self.save_drawing(visualizer.image)
                break


            # Camera capture ###############################################################
            ret, image = cap.read()
            if not ret:
                break
            image = cv.flip(image, 1)  # Mirror display
            debug_image = image.copy()
            visualizer.set_image(debug_image,self.header)

            ## If s is pressed save drawing
            if key == ord('s') & 0xff:
                self.save_drawing(visualizer.image)
                visualizer.image_saved()
                frame_counter = 10
            ## Keep the SAVE text on the image for 10 frames
            if frame_counter > 0:
                visualizer.image_saved()
                frame_counter -= 1

            # Detection implementation #############################################################
            image = cv.cvtColor(image, cv.COLOR_BGR2RGB)
            if color == (0,0,0):
                cv.circle(visualizer.image,
                          (self.header.shape[1] + 100, self.header.shape[0] - 30),
                          thickness // 2,
                          color)
            else:
                cv.circle(visualizer.image,
                          (self.header.shape[1] + 100, self.header.shape[0] - 30),
                          thickness // 2,
                          color, -1)
            image.flags.writeable = False
            results = self.hands_detection.process(image)
            image.flags.writeable = True

            #  ####################################################################
            if results.multi_hand_landmarks is not None:
                for hand_landmarks, handedness in zip(results.multi_hand_landmarks,
                                                      results.multi_handedness):
                    # Bounding box calculation
                    brect = calc_bounding_rect(debug_image, hand_landmarks)
                    # Landmark calculation
                    landmark_list = calc_landmark_list(debug_image, hand_landmarks)

                    # Conversion to relative coordinates / normalized coordinates
                    pre_processed_landmark_list = pre_process_landmark(landmark_list)

                    # Hand sign classification ##################################
                    hand_sign_id = self.key_point_classifier(pre_processed_landmark_list)

                    if hand_sign_id == 2: # Point gesture
                        points.append(landmark_list[8])
                        if len(points) > 1:
                            mask = visualizer.draw_line(mask, [points[-2], points[-1]], color,thickness)
                            lines.append([deepcopy(points), color,thickness])


                    elif hand_sign_id == 6 and len(lines) > 0:  # UNDO gesture
                        mask = np.zeros((self.config.demo.cap_height, self.config.demo.cap_width, 3), dtype=np.uint8)
                        lines.pop()
                        mask = visualizer.undo_lines(mask, lines)

                    else: ## If drawing mode is inactive add [0,0] to not draw a line from the last point
                        if len(points) > 0 and points[-1] != [0, 0]:
                            points.append([0, 0])


                    if hand_sign_id == 3: # OK gesture
                        points.clear()
                        lines = []
                        mask = np.zeros((self.config.demo.cap_height, self.config.demo.cap_width, 3), dtype=np.uint8)


                    #TODO: daca esti in select mode si degetele sunt la o anumita pozitie, schimba culoarea
                    # poti ori sa hardcodezi culoarea sau pur si simplu sa selectezi pixelii la pozitia respectiva
                    # cand esti in modul de selectie deseneaza un dreptungi plin cu culoarea selectata in jurul degetelor
                    # Deseneaza pe imagine nu pe masca !!


                    #TODO: deseneaza undeva de preferat in header un cerc cu raza = grosimea curenta
                    if hand_sign_id == 0:
                        thickness +=1
                    if hand_sign_id == 1:
                        if thickness >1:
                            thickness -= 1

                    if hand_sign_id == 5: # selection
                        x = landmark_list[8][0]
                        y = landmark_list[8][1]
                        box_w = self.header.shape[1] // 15
                        if 0 < y < self.header.shape[0] and 0 < x < self.header.shape[1] - box_w :
                            color = tuple(map(int,self.header[y,x,:]))
                        if 0 < y < self.header.shape[0] and self.header.shape[1] - box_w < x < self.header.shape[1]:
                            color = (0,0,0)

                    # Drawing part
                    visualizer.draw_bounding_rect(self.config.demo.draw_bbox, brect)
                    visualizer.draw_landmarks(landmark_list)
                    visualizer.draw_info_text(
                        brect,
                        handedness,
                        self.key_point_classifier_labels[hand_sign_id]
                    )

            else:
                if len(points) > 0 and points[-1] != [0, 0]:
                    points.append([0, 0])

            visualizer.apply_mask(mask)
            # Draw FPS
            visualizer.draw_performance(fps)

            cv.imshow("AI Painter",visualizer.image)

        cap.release()
        cv.destroyAllWindows()

if __name__ == '__main__':

    config_path = 'demo_config.yaml'
    config = OmegaConf.load(config_path)
    demo = Demo(config)
    demo.run()



