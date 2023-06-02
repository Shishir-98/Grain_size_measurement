import cv2
import numpy as np

def apply_kmeans_clustering(img, k):
    # Reshape the image to a 2D array of pixels
    pixels = img.reshape(-1, 3).astype(np.float32)
    # Define criteria for k-means clustering
    criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 5, 0.5)
    # Perform k-means clustering
    _, labels, centers = cv2.kmeans(pixels, k, None, criteria, 10, cv2.KMEANS_RANDOM_CENTERS)
    # Convert the centers to uint8 and reshape to image dimensions
    quantized_image = np.uint8(centers[labels.flatten()]).reshape(img.shape)
    return quantized_image

def apply_denoising(img):
    # Apply denoising using fastNlMeansDenoisingColored function
    denoised_image = cv2.fastNlMeansDenoisingColored(img, None, 10, 10, 7, 21)
    return denoised_image

def draw_circles(overlay_img, radii):
    # Initialize a copy of the original image for drawing
    output_image = overlay_img.copy()
    # Define the center of the image
    (height, width) = overlay_img.shape[:2]
    center = (width // 2, height // 2)
    for radius in radii:
        cv2.circle(output_image, center, radius, (255, 255, 255), 5)
    return output_image

def calculate_radii(img):
    (height, width) = img.shape[:2]
    minDim = min(width, height)
    radii = [int(minDim/8), int(2*minDim/8), int(3*minDim/8)]
    return radii

def draw_circles_and_count_intersections(Thres, original, radii):
    (height, width) = original.shape[:2]
    center = (width // 2, height // 2)
    intersections = []
    output_image = original.copy()
    for radius in radii:
        mask = np.zeros_like(Thres)
        cv2.circle(mask, center, radius, 255, 1)
        intersection = cv2.bitwise_and(Thres, mask)
        num_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(intersection, connectivity=8)
        intersections.append(num_labels - 1)
        cv2.circle(output_image, center, radius, (255, 255, 255), 5)
        for centroid in centroids[1:]:
            x, y = np.int0(centroid)
            cv2.circle(output_image, (x, y), 10, (0, 255, 0), -1)
    return intersections, output_image

class CropLayer(object):
    def __init__(self, params, blobs):
        # initialize our starting and ending (x, y)-coordinates of
        # the crop
        self.startX = 0
        self.startY = 0
        self.endX = 0
        self.endY = 0

    def getMemoryShapes(self, inputs):
        # the crop layer will receive two inputs -- we need to crop
        # the first input blob to match the shape of the second one,
        # keeping the batch size and number of channels
        (inputShape, targetShape) = (inputs[0], inputs[1])
        (batchSize, numChannels) = (inputShape[0], inputShape[1])
        (H, W) = (targetShape[2], targetShape[3])

        # compute the starting and ending crop coordinates
        self.startX = int((inputShape[3] - targetShape[3]) / 2)
        self.startY = int((inputShape[2] - targetShape[2]) / 2)
        self.endX = self.startX + W
        self.endY = self.startY + H

        # return the shape of the volume (we'll perform the actual
        # crop during the forward pass
        return [[batchSize, numChannels, H, W]]

    def forward(self, inputs):
        # use the derived (x, y)-coordinates to perform the crop
        return [inputs[0][:, :, self.startY:self.endY,
                self.startX:self.endX]]

cv2.dnn_registerLayer("Crop", CropLayer)


def callPlanimetric(imagepass):
    protoPath = "deploy.prototxt"
    modelPath = "hed_pretrained_bsds.caffemodel"
    net = cv2.dnn.readNetFromCaffe(protoPath, modelPath)
    img = imagepass
    (H, W) = img.shape[:2]
    mean_pixel_values= np.average(img, axis = (0,1))
    blob = cv2.dnn.blobFromImage(img, scalefactor=1, size=(W, H),
                                mean=(mean_pixel_values[0], mean_pixel_values[1], mean_pixel_values[2]),
                                #mean=(105, 117, 123),
                                swapRB= False, crop=False)
    blob_for_plot = np.moveaxis(blob[0,:,:,:], 0,2)
    net.setInput(blob)
    hed = net.forward()
    hed = hed[0,0,:,:]
    hed = (255 * hed).astype("uint8")
    blur = cv2.GaussianBlur(hed, (3,3), 0)
    del net
    thresh = cv2.threshold(blur, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)[1]
    n_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(thresh, connectivity=4)
    circle_diameter = int(2/3 * H)
    circle_radius = int(circle_diameter / 2)
    circle_center = (int(W / 2), int(H / 2))
    imCircl = img.copy()
    cv2.circle(imCircl, circle_center, circle_radius, (255, 255, 255), 5)


    mask = np.zeros_like(labels)
    cv2.circle(mask, circle_center, circle_radius, (255, 255, 255), -1)

            # Extract the region of interest within the circle
    roi = cv2.bitwise_and(labels, mask)

            # Calculate the number of labeled regions (grains)
    num_grains = 0
    marked_image = imCircl.copy()  # Create a copy of the quantized image to mark the grains

    for label in np.unique(labels):
        if label == 0:
            continue
        region = (labels == label).astype(np.uint8)
        intersection = cv2.bitwise_and(region,(roi>0).astype(np.uint8))
        if np.sum(intersection) > 0 and np.sum(region)>50:
            if np.sum(region) == np.sum(intersection):
                num_grains +=1
                contours, _ = cv2.findContours(region, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
                cv2.drawContours(marked_image, contours, -1, (0, 255, 0), 2)
                pass
            else:
                num_grains +=0.5
                contours, _ = cv2.findContours(region, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
                cv2.drawContours(marked_image, contours, -1, (0, 0, 255), 2)
    return img, imCircl, marked_image, num_grains
            # Display the marked image with labeled grains

def callIntercept(imagepass):
    
    protoPath = "deploy.prototxt"
    modelPath = "hed_pretrained_bsds.caffemodel"
    net = cv2.dnn.readNetFromCaffe(protoPath, modelPath)
    inputImg = imagepass
    img = apply_denoising(inputImg)
    (H, W) = img.shape[:2]
    mean_pixel_values= np.average(img, axis = (0,1))
    blob = cv2.dnn.blobFromImage(img, scalefactor=1, size=(W, H),
                                mean=(mean_pixel_values[0], mean_pixel_values[1], mean_pixel_values[2]),
                                #mean=(105, 117, 123),
                                swapRB= False, crop=False)
    blob_for_plot = np.moveaxis(blob[0,:,:,:], 0,2)
    net.setInput(blob)
    hed = net.forward()
    hed = hed[0,0,:,:]
    hed = (255 * hed).astype("uint8")
    del net
    blur = cv2.GaussianBlur(hed, (3,3), 0)
    thresh = cv2.threshold(blur, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)[1]
    radii = calculate_radii(img)
    image_withCircles = draw_circles(img,radii)
    # Display the image with circles
    intersections, output_image = draw_circles_and_count_intersections(thresh, img, radii)
    return img, intersections, image_withCircles, output_image
