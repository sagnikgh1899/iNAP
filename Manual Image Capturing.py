# import the necessary packages
import cv2

# now let's initialize the list of reference point
ref_point = []
crop = False


def shape_selection(event, x, y, flags, param):
    # grab references to the global variables
    global ref_point, crop

    # if the left mouse button was clicked, record the starting
    # (x, y) coordinates and indicate that cropping is being performed
    if event == cv2.EVENT_LBUTTONDOWN:
        ref_point = [(x, y)]

        # check to see if the left mouse button was released
    elif event == cv2.EVENT_LBUTTONUP:
        # record the ending (x, y) coordinates and indicate that
        # the cropping operation is finished
        ref_point.append((x, y))

        # draw a rectangle around the region of interest
        cv2.rectangle(image, ref_point[0], ref_point[1], (0, 255, 0), 2)
        cv2.imshow("image", image)


# load the image, clone it, and setup the mouse callback function
path_img = r'E:\IIIT\Fingernail\FINGERNAIL IMAGE DATASET\48(R).jpg'
image = cv2.imread(path_img)  # download.jpg")
image = cv2.resize(image, (200, 200))  # Resize image
clone = image.copy()
cv2.namedWindow("image")
cv2.setMouseCallback("image", shape_selection)

# keep looping until the 'q' key is pressed
img_count = 1
while True:
    # display the image and wait for a keypress
    cv2.imshow("image", image)
    key = cv2.waitKey(1) & 0xFF

    if key == ord("q"):
        break

    # press 'r' to reset the window
    if key == ord("r"):
        image = clone.copy()

        # if the 'c' key is pressed, break from the loop
    elif key == ord("c"):
        if len(ref_point) == 2:
            crop_img = clone[ref_point[0][1]:ref_point[1][1], ref_point[0][0]:ref_point[1][0]]
            img_name = "Image_{}.png".format(img_count)
            cv2.imwrite(img_name, crop_img)
            print("Image_{}_Written!!".format(img_count))
            # cv2.imshow(img_name, crop_img)
            # cv2.waitKey(0)
        image = clone.copy()
        img_count += 1


# close all open windows
cv2.destroyAllWindows()
