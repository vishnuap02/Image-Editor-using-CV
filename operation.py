import cv2
import numpy as np
import random
import string
import datetime
from scipy.interpolate import UnivariateSpline

# Load the input image
# image = cv2.imread("lena_dip.jpg")
# image1 = cv2.imread("vishnu.jpg")
# image3 = cv2.imread("img3.jpg")
# image4 = cv2.imread("image4.jpg")  # SRK
# image5 = cv2.imread("image5.jpg")  # Salman bhai
# image6 = cv2.imread("image6.jpg")  # Himanshu bhai
# image = cv2.resize(image, (image3.shape[1], image3.shape[0]))
# image1 = cv2.resize(image1, (image3.shape[1], image3.shape[0]))
# image4 = cv2.resize(image4, (image3.shape[1], image3.shape[0]))
# image5 = cv2.resize(image5, (image3.shape[1], image3.shape[0]))
# image6 = cv2.resize(image6, (image3.shape[1], image3.shape[0]))


# img = image1
# normalized_img = np.zeros(img.shape)
# cv2.normalize(img, normalized_img, 0, 255, cv2.NORM_MINMAX)
#
# # Convert the image to uint8
# normalized_img = normalized_img.astype(np.uint8)


# Display the original and normalized images
# cv2.imshow('Original Image', img)
# cv2.imshow('Normalized Image', normalized_img.astype(np.uint8))
# cv2.waitKey(0)
# cv2.destroyAllWindows()
# gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
#
# # Apply histogram equalization to normalize the light distribution
# image = cv2.equalizeHist(gray)
# image = cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)

def initialize():
    image6 = cv2.imread("image6.jpg")  # Himanshu bhai


def histogram_equalization(img):
    # Calculate the histogram of the input image
    hist, bins = np.histogram(img.flatten(), 256, [0, 256])

    # Calculate the cumulative distribution function of the histogram
    cdf = hist.cumsum()

    # Normalize the cdf to have values between 0 and 1
    cdf_normalized = cdf / cdf.max()

    # Calculate the histogram equalization function
    cdf_equalized = np.round(cdf_normalized * 255)

    # Apply the histogram equalization function to the input image
    img_equalized = cdf_equalized[img]

    # Convert the data type of the image to uint8
    img_equalized = np.uint8(img_equalized)

    # Return the equalized image
    return img_equalized


def histogram_matching(source_img, reference_img):
    # Calculate the histogram of the source and reference images
    source_hist, _ = np.histogram(source_img.flatten(), 256, [0, 256])
    reference_hist, _ = np.histogram(reference_img.flatten(), 256, [0, 256])

    # Calculate the cumulative distribution functions of the source and reference histograms
    source_cdf = source_hist.cumsum()
    reference_cdf = reference_hist.cumsum()

    # Normalize the CDFs to have values between 0 and 1
    source_cdf_normalized = source_cdf / source_cdf.max()
    reference_cdf_normalized = reference_cdf / reference_cdf.max()

    # Calculate the mapping function from the source to the reference histogram
    mapping_function = np.interp(
        source_cdf_normalized, reference_cdf_normalized, range(256))

    # Apply the mapping function to the source image
    matched_img = np.round(
        np.interp(source_img.flatten(), range(256), mapping_function))
    matched_img = matched_img.reshape(source_img.shape)

    # Convert the data type of the matched image to uint8
    matched_img = np.uint8(matched_img)

    # Return the matched image
    return matched_img


def add_text_on_image(image, text):
    today = datetime.datetime.today().strftime('%Y-%m-%d')

    text = text+' '+str(today)
    # text = f"{text}\n{str(today)}"
    # Set the dimensions of the image
    image2 = cv2.imread("lena_dip.jpg")
    image = cv2.resize(image, (image2.shape[1], image2.shape[0]))
    image = add_salt_pep(image)
    width, height, _ = image.shape
    print(width, height)
    # Generate a random string to be written on the image
    string_length = 10
    random_string = text
    # Define the font and parameters for the text
    font = cv2.FONT_HERSHEY_COMPLEX
    font_scale = 1
    thickness = 3
    text_size, _ = cv2.getTextSize(random_string, font, font_scale, thickness)
    # Determine the position to write the text on the image
    x = int((width - text_size[0]) / 2)
    y = int((height + text_size[1]) / 1.33)
    # Write the text on the image
    cv2.putText(image, random_string, (x, y), font, font_scale,
                (255, 255, 255), thickness, cv2.LINE_AA)
    return image


def add_salt_pep(image):
    # Add salt and pepper noise to the image
    noise_image = np.copy(image)
    row, col, ch = noise_image.shape
    noise_density = 0.02  # Change this to adjust the noise density
    # Generate random values between 0 and 1 for each pixel in the image
    random_values = np.random.rand(row, col)
    # Add black noise where the random values are less than the noise density
    noise_image[random_values < noise_density / 2] = [0, 0, 0]
    # Add white noise where the random values are greater than 1 - the noise density
    noise_image[random_values > 1 - noise_density / 2] = [255, 255, 255]
    return noise_image


def bg_removal(img):
    # Convert the image to grayscale
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    # Threshold the image to create a mask
    # 250 is threshholding limit.
    ret, mask = cv2.threshold(gray, 250, 255, cv2.THRESH_BINARY)
    # Invert the mask
    mask = cv2.bitwise_not(mask)
    # Apply the mask to the image
    masked_img = cv2.bitwise_and(img, img, mask=mask)
    # cv2.imshow('Masked Image', masked_img)
    # cv2.waitKey(0)
    # cv2.destroyAllWindows()
    return masked_img


def remove_background(img):
    # Load the image
    # img = cv2.imread(image_path)

    # Convert to grayscale
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    # Apply thresholding
    ret, thresh = cv2.threshold(
        gray, 0, 255, cv2.THRESH_BINARY_INV+cv2.THRESH_OTSU)

    # Create a kernel for morphological operations
    kernel = np.ones((3, 3), np.uint8)

    # Perform morphological opening and closing
    opening = cv2.morphologyEx(thresh, cv2.MORPH_OPEN, kernel, iterations=2)
    closing = cv2.morphologyEx(opening, cv2.MORPH_CLOSE, kernel, iterations=2)

    # Find the contours in the image
    contours, hierarchy = cv2.findContours(
        closing, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    # Draw the contours on a blank mask
    mask = np.zeros(img.shape[:2], np.uint8)
    cv2.drawContours(mask, contours, -1, (255, 255, 255), -1)

    # Apply the mask to the original image
    result = cv2.bitwise_and(img, img, mask=mask)

    # Return the result
    return result


def LookupTable(x, y):
    spline = UnivariateSpline(x, y)
    return spline(range(256))


def apply_filter(img, filter_name='juno'):
    if filter_name == 'summer':
        increaseLookupTable = LookupTable([0, 64, 128, 256], [0, 80, 160, 256])
        decreaseLookupTable = LookupTable([0, 64, 128, 256], [0, 50, 100, 256])
        blue_channel, green_channel, red_channel = cv2.split(img)
        red_channel = cv2.LUT(
            red_channel, increaseLookupTable).astype(np.uint8)
        blue_channel = cv2.LUT(
            blue_channel, decreaseLookupTable).astype(np.uint8)
        img = cv2.merge((blue_channel, green_channel, red_channel))

    elif filter_name == 'winter':
        increaseLookupTable = LookupTable([0, 64, 128, 256], [0, 80, 160, 256])
        decreaseLookupTable = LookupTable([0, 64, 128, 256], [0, 50, 100, 256])
        blue_channel, green_channel, red_channel = cv2.split(img)
        red_channel = cv2.LUT(
            red_channel, decreaseLookupTable).astype(np.uint8)
        blue_channel = cv2.LUT(
            blue_channel, increaseLookupTable).astype(np.uint8)
        img = cv2.merge((blue_channel, green_channel, red_channel))

    else:
        # No filter applied
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

    return img


def add_counters(image1):
    img_gray1 = cv2.cvtColor(image1, cv2.COLOR_BGR2GRAY)

    ret, thresh1 = cv2.threshold(img_gray1, 150, 255, cv2.THRESH_BINARY)
    contours2, hierarchy2 = cv2.findContours(thresh1, cv2.RETR_TREE,
                                             cv2.CHAIN_APPROX_SIMPLE)
    image_copy2 = np.zeros(image1.shape, dtype=np.uint8)
    cv2.drawContours(image_copy2, contours2, -1, (255, 255, 0), 2, cv2.LINE_AA)
    # cv2.imshow('SIMPLE Approximation contours', image_copy2)
    # cv2.waitKey(0)
    # cv2.destroyAllWindows()
    return image_copy2


def foreground(img2, val):
    if val == 1:
        img = cv2.imread("image6.jpg")  # srk bhai
    else:
        img = cv2.imread("image7.jpg")  # srk bhai
    img2 = cv2.resize(img2, (img.shape[1], img.shape[0]))
    print(img.shape[-1], img2.shape[-1])
    hh, ww = img.shape[:2]

    # threshold on white
    # Define lower and uppper limits
    lower = np.array([200, 200, 200])
    upper = np.array([255, 255, 255])

    # Create mask to only select black
    thresh = cv2.inRange(img, lower, upper)

    # apply morphology
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (20, 20))
    morph = cv2.morphologyEx(thresh, cv2.MORPH_CLOSE, kernel)

    # invert morp image
    mask = 255 - morph

    # apply mask to image
    # result = cv2.bitwise_and(img, img, mask=mask)
    mask = cv2.resize(mask, (img2.shape[1], img2.shape[0]))
    # convert to 3 channel
    mask = cv2.cvtColor(mask, cv2.COLOR_GRAY2RGB)
    result = cv2.add(mask, img2)
    print(mask.shape, img2.shape)

    # convert to negative for addition
    result_neg = cv2.bitwise_not(result)
    img_neg = cv2.bitwise_not(img)
    result_neg = cv2.add(result_neg, img_neg)
    result = cv2.bitwise_not(result_neg)

    # cv2.imshow('thresh', thresh)
    # cv2.imshow('morph', morph)
    # cv2.imshow('mask', mask)
    # cv2.imshow('result', result)
    # cv2.waitKey(0)
    # cv2.destroyAllWindows()
    return result


def animate(img):
    img = bg_removal(img)
    height, width, c = img.shape

    i = 0

    while True:
        i += 1

        # divided the image into left and right part
        # like list concatenation we concatenated
        # right and left together
        l = img[:, :(i % width)]
        r = img[:, (i % width):]

        img1 = np.hstack((r, l))

        # this function will concatenate
        # the two matrices
        cv2.imshow('animation', img1)

        if cv2.waitKey(1) == ord('q'):
            # press q to terminate the loop
            cv2.destroyAllWindows()
            break

# histogram equalization
# image = histogram_equalization(image)
# image3 = histogram_equalization(image3)
# image6 = histogram_equalization(image6)

# SNAP
# Display the modified image
# img1 = add_salt_pep(image)
# today = datetime.datetime.today().strftime('%Y-%m-%d')
# text = "Thursday "+str(today)
# print(text)
# img1 = add_text_on_image(img1,text)
# cv2.imshow("Original  Image", image)
# cv2.imshow("Noise Image", img1)
# cv2.waitKey(0)
# cv2.destroyAllWindows()

# histogram matching
# img3 = histogram_matching(image3,image6)

# BG removal
# bg_img = bg_removal(img3)

# Draw the boundaries.
# bnd_img = add_counters(image3)

# Animate
# animate(bg_img)

# foreground
# foreground(image3,image4) # srk with fan
# foreground(image3,image5) # salman with fan
# img = foreground(image5) # salman with Himanshu


# img = add_text_on_image(image,'Topper')
# cv2.imshow("Noise Image", img)
# cv2.waitKey(0)
# cv2.destroyAllWindows()
