import numpy as np
import cv2
import pygame.mixer


### Set up sounds
pygame.init()
pygame.mixer.init()

# Melody
red_sound = pygame.mixer.Sound('loops/red_melody.wav')
red_sound.set_volume(0.0)
red_sound.play()

# Chords
green_sound = pygame.mixer.Sound('loops/green_chords.wav')
green_sound.set_volume(0.0)
green_sound.play()

# Birds
blue_sound = pygame.mixer.Sound('loops/blue_bird.wav')
blue_sound.set_volume(0.0)
blue_sound.play()

# Melody
orange_sound = pygame.mixer.Sound('loops/orange_harmony.wav')
orange_sound.set_volume(0.0)
orange_sound.play()

# Beat
grey_sound = pygame.mixer.Sound('loops/rock_beat.wav')
grey_sound.set_volume(0.0)
grey_sound.play()

### Capturing video through webcam
webcam = cv2.VideoCapture(0)

### Main application loop
while(1):
    has_red = False
    has_green = False
    has_blue = False
    has_orange = False
    has_grey = False

    # Reading the video from the
    # webcam in image frames
    _, imageFrame = webcam.read()

    # Convert the imageFrame in
    # BGR(RGB color space) to
    # HSV(hue-saturation-value)
    # color space
    hsvFrame = cv2.cvtColor(imageFrame, cv2.COLOR_BGR2HSV)

    # Set range for red color and
    # define mask
    red_lower = np.array([161, 155, 84], np.uint8)
    red_upper = np.array([180, 255, 255], np.uint8)
    red_mask = cv2.inRange(hsvFrame, red_lower, red_upper)

    # Set range for green color and
    # define mask
    green_lower = np.array([25, 52, 72], np.uint8)
    green_upper = np.array([102, 255, 255], np.uint8)
    green_mask = cv2.inRange(hsvFrame, green_lower, green_upper)

    # Set range for blue color and
    # define mask
    blue_lower = np.array([94, 80, 2], np.uint8)
    blue_upper = np.array([120, 255, 255], np.uint8)
    blue_mask = cv2.inRange(hsvFrame, blue_lower, blue_upper)

    # Set range for orange color and
    # define mask
    orange_lower = np.array([10, 100, 20], np.uint8)
    orange_upper = np.array([25, 255, 255], np.uint8)
    orange_mask = cv2.inRange(hsvFrame, orange_lower, orange_upper)

    # Set range for grey color and
    # define mask
    grey_lower = np.array([0, 0, 0], np.uint8)
    grey_upper = np.array([100, 100, 100], np.uint8)
    grey_mask = cv2.inRange(hsvFrame, grey_lower, grey_upper)

    # Morphological Transform, Dilation
    # for each color and bitwise_and operator
    # between imageFrame and mask determines
    # to detect only that particular color
    kernal = np.ones((5, 5), "uint8")

    # For red color
    red_mask = cv2.dilate(red_mask, kernal)
    res_red = cv2.bitwise_and(imageFrame, imageFrame,
            mask = red_mask)

    # For green color
    green_mask = cv2.dilate(green_mask, kernal)
    res_green = cv2.bitwise_and(imageFrame, imageFrame,
            mask = green_mask)

    # For blue color
    blue_mask = cv2.dilate(blue_mask, kernal)
    res_blue = cv2.bitwise_and(imageFrame, imageFrame,
            mask = blue_mask)

    # For orange color
    orange_mask = cv2.dilate(orange_mask, kernal)
    res_orange = cv2.bitwise_and(imageFrame, imageFrame,
            mask = orange_mask)

    # For grey color
    grey_mask = cv2.dilate(grey_mask, kernal)
    res_grey = cv2.bitwise_and(imageFrame, imageFrame,
            mask = grey_mask)

    # Creating contour to track red color
    contours, hierarchy = cv2.findContours(red_mask,
            cv2.RETR_TREE,
            cv2.CHAIN_APPROX_SIMPLE)

    for pic, contour in enumerate(contours):
        area = cv2.contourArea(contour)
        if(area > 300):
            x, y, w, h = cv2.boundingRect(contour)
            imageFrame = cv2.rectangle(imageFrame, (x, y),
                    (x + w, y + h),
                    (0, 0, 255), 2)

            cv2.putText(imageFrame, "Red Colour", (x, y),
                    cv2.FONT_HERSHEY_SIMPLEX, 1.0,
                    (0, 0, 255))
            has_red = True


    # Creating contour to track orange color
    contours, hierarchy = cv2.findContours(orange_mask,
            cv2.RETR_TREE,
            cv2.CHAIN_APPROX_SIMPLE)
    for pic, contour in enumerate(contours):
        area = cv2.contourArea(contour)
        if(area > 300):
            x, y, w, h = cv2.boundingRect(contour)
            imageFrame = cv2.rectangle(imageFrame, (x, y),
                    (x + w, y + h),
                    (0, 255, 255), 2)

            cv2.putText(imageFrame, "orange Colour", (x, y),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    1.0, (0, 255, 255))
            has_orange = True

    # Creating contour to track green color
    contours, hierarchy = cv2.findContours(green_mask,
            cv2.RETR_TREE,
            cv2.CHAIN_APPROX_SIMPLE)

    for pic, contour in enumerate(contours):
        area = cv2.contourArea(contour)
        if(area > 300):
            x, y, w, h = cv2.boundingRect(contour)
            imageFrame = cv2.rectangle(imageFrame, (x, y),
                    (x + w, y + h),
                    (0, 255, 0), 2)

            cv2.putText(imageFrame, "Green Colour", (x, y),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    1.0, (0, 255, 0))
            has_green = True


    # Creating contour to track blue color
    contours, hierarchy = cv2.findContours(blue_mask,
            cv2.RETR_TREE,
            cv2.CHAIN_APPROX_SIMPLE)
    for pic, contour in enumerate(contours):
        area = cv2.contourArea(contour)
        if(area > 300):
            x, y, w, h = cv2.boundingRect(contour)
            imageFrame = cv2.rectangle(imageFrame, (x, y),
                    (x + w, y + h),
                    (255, 0, 0), 2)

            cv2.putText(imageFrame, "Blue Colour", (x, y),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    1.0, (255, 0, 0))
            has_blue = True

    # Creating contour to track grey color
    contours, hierarchy = cv2.findContours(grey_mask,
            cv2.RETR_TREE,
            cv2.CHAIN_APPROX_SIMPLE)
    for pic, contour in enumerate(contours):
        area = cv2.contourArea(contour)
        if(area > 300):
            x, y, w, h = cv2.boundingRect(contour)
            imageFrame = cv2.rectangle(imageFrame, (x, y),
                    (x + w, y + h),
                    (128, 128, 128), 2)

            cv2.putText(imageFrame, "Grey Colour", (x, y),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    1.0, (128, 128, 128))
            has_grey = True

    # Play sounds based on what's in the shot
    if has_red:
        # Play red sounds
        red_sound.set_volume(1)
    else:
        # No red? Turn off sound
        red_sound.set_volume(0.0)

    if has_green:
        # Play green sounds
        green_sound.set_volume(1)
    else:
        # No green? Turn off sound
        green_sound.set_volume(0.0)

    if has_blue:
        # Play blue sounds
        blue_sound.set_volume(1)
    else:
        # No blue? Turn off sound
        blue_sound.set_volume(0.0)

    if has_orange:
        # Play orange sounds
        orange_sound.set_volume(1)
    else:
        # No orange? Turn off sound
        orange_sound.set_volume(0.0)

    if has_grey:
        # Play grey sounds
        grey_sound.set_volume(1)
    else:
        # No grey? Turn off sound
        grey_sound.set_volume(0.0)

    # Program Termination
    cv2.imshow("Multiple Color Detection in Real-TIme", imageFrame)
    if cv2.waitKey(10) & 0xFF == ord('q'):
        cap.release()
        cv2.destroyAllWindows()
        break
