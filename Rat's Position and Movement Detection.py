# 小鱼干编程
# 可可爱爱没有脑袋
# 开发时间：
import sensor, image, time, os, tf, math, uos, gc, pyb, lcd

p = pyb.Pin("P0", pyb.Pin.OUT_PP)  # Set the pin
sensor.reset()  # Reset and initialize the sensor.
sensor.set_pixformat(sensor.RGB565)  # Set pixel format to RGB565 (or GRAYSCALE)
sensor.set_framesize(sensor.QQVGA2)  # Set frame size to QVGA (320x240)
sensor.set_windowing((240, 240))  # Set 240x240 window.
sensor.skip_frames(time=2000)  # Let the camera adjust.

net = None
labels = None
min_confidence = 0.5
previous_detection = None

try:
    # load the model, alloc the model file on the heap if we have at least 64K free after loading
    net = tf.load("trained.tflite", load_to_fb=uos.stat('trained.tflite')[6] > (gc.mem_free() - (64 * 1024)))
except Exception as e:
    raise Exception('Failed to load "trained.tflite", did you copy the.tflite and labels.txt file onto the mass-storage device? (' + str(e) + ')')

try:
    labels = [line.rstrip('\n') for line in open("labels.txt")]
except Exception as e:
    raise Exception('Failed to load "labels.txt", did you copy the .tflite and labels.txt file onto the mass-storage device? (' + str(e) + ')')

colors = [ # Add more colors if you are detecting more than 7 types of classes at once.
    (  0,   0, 255),    #blue
    (  0, 255,   0),    #green
    (255, 255,   0),    #yellow
    (255,   0,   0),    # red
    (255,   0, 255),
    (  0, 255, 255),
    (255, 255, 255),
]

clock = time.clock()
while True:  # acquire image in while loop
    clock.tick()

    img = sensor.snapshot()

    # detect() returns all objects found in the image (splitted out per class already)
    # we skip class index 0, as that is the background, and then draw circles of the center
    # of our objects

    for i, detection_list in enumerate(net.detect(img, thresholds=[(math.ceil(min_confidence * 255), 255)])):
        if len(detection_list) == 0: continue  # no detections for this class?
        if i == 1:  # Rat class
            if previous_detection is not None:
                # Calculate motion between current and previous detection
                motion_x = abs(previous_detection[0] - detection_list[0][0])
                motion_y = abs(previous_detection[1] - detection_list[0][1])
                if motion_x > 10 or motion_y > 10:  # Adjust threshold as needed
                    p.high()  # Output high signal indicating motion
                    sensor.skip_frames(time=13000)
                else:
                    p.low()  # Output low signal indicating no motion
            else:
                p.low()  # Output low signal if there's no previous detection
            previous_detection = (detection_list[0][0], detection_list[0][1])  # Update previous detection position

        print("********** %s **********" % labels[i])
        for d in detection_list:
            [x, y, w, h] = d.rect()
            center_x = math.floor(x + (w / 2))
            center_y = math.floor(y + (h / 2))
            print('x %d\ty %d' % (center_x, center_y))
            img.draw_circle((center_x, center_y, 12),  color=colors[i], thickness=2)
            img.draw_string(center_x + 12, center_y + 12, labels[i], color=colors[i])
        #lcd.display(sensor.snapshot())  # 拍照并显示图像。

    print(clock.fps(), "fps", end="\n\n")