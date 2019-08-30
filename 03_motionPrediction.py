#!/usr/bin/env python
# -*- coding: utf-8 -*-

import os
import datetime
import imutils
from imutils.video import FPS
import cv2
from keras.applications.vgg16 import VGG16
from keras.models import Sequential, Model
from keras.layers import Input, Activation, Dropout, Flatten, Dense
from keras.preprocessing import image
import numpy as np
from numpy import mean
import pandas as pd
from PIL import ImageFont, ImageDraw, Image
from collections import Counter, defaultdict

# Class for cv2 putText function to support Chinese Font.
class CvPutChnText:
    def __init__(self):
        pass

    @classmethod
    def puttext(cls, cv_image, text, point, font_size=20, color=(0, 0, 0)):
        font = ImageFont.truetype('simhei.ttf', font_size, encoding='utf-8')
        cv_rgb_image = cv2.cvtColor(cv_image, cv2.COLOR_BGR2RGB)
        pil_image = Image.fromarray(cv_rgb_image)
        draw = ImageDraw.Draw(pil_image)
        draw.text(point, text, fill=color, font=font)
        cv_rgb_result_image = np.asarray(pil_image)
        cv_bgr_result_image = cv2.cvtColor(cv_rgb_result_image, cv2.COLOR_RGB2BGR)

        return cv_bgr_result_image

# Detect hand and return a hand mask as well as the hand areas
def extract_hand(frame):
    HSV_MIN = np.array([0, 20, 50])
    HSV_MAX = np.array([10, 100, 225])
    converted = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
    handMask = cv2.inRange(converted, HSV_MIN, HSV_MAX)
    # Apply a series of erosions and dilations to the mask using an elliptical kernel
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (11, 11))
    handMask = cv2.erode(handMask, kernel, iterations=2)
    handMask = cv2.dilate(handMask, kernel, iterations=2)
    contours, _ = cv2.findContours(handMask, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    extracted = []
    for cnt in contours:
        approx = cv2.convexHull(cnt)
        rect = cv2.boundingRect(approx)
        extracted.append(np.array(rect))
    return extracted, handMask

if __name__ == '__main__':
    # Define the model, output and logging dir
    input_dir = 'data/input_video/'
    model_dir = 'results/'
    sub_dir = model_dir + 'output_video_sub/'
    records_dir = 'Records/'
    cropped_dir = 'Cropped/'

    if not os.path.exists(sub_dir):
        os.mkdir(sub_dir)

    if not os.path.exists(records_dir):
        os.mkdir(records_dir)

    if not os.path.exists(cropped_dir):
        os.mkdir(cropped_dir)

    # Define the initial variables
    img_size = 150  # has to be the same as the model trained and generated
    classes = ['medicine00', 'medicine01', 'medicine02', 'medicine03', 'medicine04',
               'medicine05', 'medicine06', 'medicine07', 'medicine08', 'medicine09']
    nb_classes = len(classes)

    label_name = {'0': u'头孢克洛胶囊', '1': u'氯雷他定片', '2': u'润肺膏', '3': u'氯化钠注射液', '4': u'普乐安片', '5': u'正柴胡饮颗粒',
                  '6': u'阿莫西林克拉维酸钾', '7': u'维生素b2片', '8': u'维生素C片', '9': u'辛伐他汀片'}

    # import model
    input_tensor = Input(shape=(img_size, img_size, 3))
    vgg16 = VGG16(include_top=False, weights='imagenet', input_tensor=input_tensor)
    fc = Sequential()
    fc.add(Flatten(input_shape=vgg16.output_shape[1:]))
    fc.add(Dense(256, activation='relu'))
    fc.add(Dropout(0.5))
    fc.add(Dense(nb_classes, activation='softmax'))
    model = Model(input=vgg16.input, output=fc(vgg16.output))
    model.load_weights(model_dir + 'finetuning.h5')
    model.compile(loss='categorical_crossentropy',
                  optimizer='adam',
                  metrics=['accuracy'])

    # Start capturing the test motion videos
    for vf in os.listdir('./data/input_video'):
        # Cropped file to be reset
        if not os.path.exists(cropped_dir + vf):
            os.mkdir(cropped_dir + vf)

        for file in os.scandir(cropped_dir + vf):
            if file.name.endswith('.png'):
                os.unlink(file.path)

        # Start capturing
        print('[INFO] Capturing test videos...' + vf)
        vs = cv2.VideoCapture(input_dir + vf)

        # Resize the output video to save spaces
        W = 500
        H = 375
        fps = vs.get(cv2.CAP_PROP_FPS)
        fourcc = cv2.VideoWriter_fourcc(*'XVID')
        out = cv2.VideoWriter(sub_dir + vf.split('.')[0] + '.mp4', fourcc, fps, (W*2, H))
        # Start the timer and record the processing time.
        fps = FPS().start()

        # Initialize the first frame in the video stream
        firstFrame = None

        # Initialize the cropped images index and logging data
        idx = 0
        obj_array = []
        col_names = ['label', 'type', 'count', 'curr', 'mean', 'max']
        records = pd.DataFrame(columns=col_names)

        # Loop over the frames of the video
        while vs.isOpened():
            # Grab the current frame and initialize the detected/undetected text
            ret, frame = vs.read()
            text = 'Undetected'
            # If the frame could not be grabbed, go to the end video
            if frame is None:
                break

            # Resize the frame to fit with output streams
            frame = imutils.resize(frame, width=W)

            handcnts, handMask = extract_hand(frame)
            # blur the mask to help remove noise, then apply the
            # mask to the frame
            handMask = cv2.GaussianBlur(handMask, (3, 3), 0)
            hand = cv2.bitwise_and(frame, frame, mask=handMask)

            after = frame - hand
            after[np.where((after == [0, 0, 0]).all(axis=2))] = [255, 255, 255]

            # Remove the shadow of the object to improve the accuracy of detection.
            # IT'S JUST A BEST EFFORT.
            blur = cv2.blur(after, (51, 51))
            ratio = after / blur
            index = np.where(ratio >= 1.00)
            ratio[index] = 1
            ratio_int = np.array(ratio * 255, np.uint8)
            ration_HSV = cv2.cvtColor(ratio_int, cv2.COLOR_BGR2HSV)
            ret, thresh = cv2.threshold(ration_HSV[:, :, 2], 0, 255, cv2.THRESH_OTSU)
            ration_HSV[:, :, 2] = thresh
            ration_ret = cv2.cvtColor(ration_HSV, cv2.COLOR_HSV2BGR)

            # Convert to gray and blur it
            gray = cv2.cvtColor(ration_ret, cv2.COLOR_BGR2GRAY)
            gray = cv2.GaussianBlur(gray, (21, 21), 0)

            # If the first frame is None, initialize it.
            if firstFrame is None:
                firstFrame = gray
                continue

            # Calculate the difference between the current frame and the first frame.
            frameDelta = cv2.absdiff(firstFrame, gray)
            thresh = cv2.threshold(frameDelta, 25, 255, cv2.THRESH_BINARY)[1]

            # Dilate the thresholded frame to fill in.
            thresh = cv2.dilate(thresh, None, iterations=2)

            # Find contours of the thresholded frame.
            cnts = cv2.findContours(thresh.copy(), cv2.RETR_EXTERNAL,
                                    cv2.CHAIN_APPROX_SIMPLE)
            cnts = imutils.grab_contours(cnts)

            # Initialize the labels to be marked.
            labels = []

            # Loop over the contours.
            for c in cnts:
                # If the contour's size is too small, ignore it.
                if cv2.contourArea(c) < 2000:
                    continue

                # Calculate the bounding box for the contour and draw it on the frame.
                (x, y, w, h) = cv2.boundingRect(c)

                # Detect if the contour is a hand, if yes, skip it
                handcnts, handMask = extract_hand(frame[y:y + h, x:x + w])
                if len(handcnts) > 0:
                    continue

                cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)
                # Update the text
                text = 'Undetected'

                # Logging and saving the cropped contours to disk.
                new_cnt = frame[y:y + h, x:x + w]
                cv2.imwrite(os.path.join('Cropped', vf, str(idx) + '.png'), new_cnt)
                new_cnt = image.load_img(os.path.join('Cropped', vf, str(idx) + '.png'),
                                         target_size=(img_size, img_size))
                # img = cv2.resize(frame[y:y + h, x:x + w], dsize=(img_size, img_size))
                idx += 1

                # Format the contours to be classified by model.
                input = image.img_to_array(new_cnt)
                input = np.expand_dims(input, axis=0)
                # Same with modeling process that used rescale of ImageDataGenerator, normaiization is necessary.
                input = input / 255.0

                # Predict and mark the label of the object detected.
                img_pred = model.predict(input)[0]
                (max_p_img, max_label_img) = max((v, i) for i, v in enumerate(img_pred))
                labels.append(str(max_label_img))

                # The following part is ONLY for LOGGING.
                obj_array.append((str(max_label_img), max_p_img))
                dict = Counter(label for i, j in obj_array for label in i)
                count_list = [(label, count) for label, count in dict.most_common() if count >= 1]
                d = defaultdict(list)
                for key, value in obj_array:
                    d[key].append(value)
                records.loc[len(records)] = [str(max_label_img), 'img', dict[str(max_label_img)], max_p_img,
                                             mean(d[str(max_label_img)]), max(d[str(max_label_img)])]

            # Label the frame with the object detected and time stamp to record the processing time.
            frame = CvPutChnText.puttext(frame, 'Detected {}'.format(
                ','.join([':'.join(['NO.' + str(x), label_name[str(x)]]) for x in list(set(labels))])),
                                         (10, 20), 20, (0, 220, 0))
            frame = CvPutChnText.puttext(frame, datetime.datetime.now().strftime('%A %d %B %Y %I:%M:%S%p'),
                                         (10, frame.shape[0] - 20), 20, (255, 0, 0))

            # Show the frame for debug and outstream to the results.
            #cv2.imshow('Security Feed', np.hstack([frame, ration_ret]))
            out.write(np.hstack([frame, ration_ret]))
            # Until the `Esc` key is pressed, that will break from the loop.
            k = cv2.waitKey(100)
            if k == 27: break
            fps.update()

        # Save the logging data.
        records.to_csv(records_dir + vf.split('.')[0] + '.csv')
        # Stop the timer and display FPS information
        fps.stop()
        print('[INFO] Elapsed time: {:.2f}'.format(fps.elapsed()))
        print('[INFO] Approx. FPS: {:.2f}'.format(fps.fps()))

        # Clean up the opencv instances and close any open windows.
        vs.release()
        out.release()
    cv2.destroyAllWindows()