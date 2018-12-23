import numpy as np
import cv2
import pdb

# Identify pixels above the threshold
# Threshold of RGB > 160 does a nice job of identifying ground pixels only
def color_thresh(img, rgb_thresh=(160, 160, 160)):
    # Create an array of zeros same xy size as img, but single channel
    color_select = np.zeros_like(img[:,:,0])
    # Require that each pixel be above all three threshold values in RGB
    # above_thresh will now contain a boolean array with "True"
    # where threshold was met
    above_thresh = (img[:,:,0] > rgb_thresh[0]) \
                & (img[:,:,1] > rgb_thresh[1]) \
                & (img[:,:,2] > rgb_thresh[2])
    # Index the array of zeros with the boolean array and set to 1
    color_select[above_thresh] = 1
    # Return the binary image
    return color_select

# Define a function to convert from image coords to rover coords
def rover_coords(binary_img):
    # Identify nonzero pixels
    ypos, xpos = binary_img.nonzero()
    # Calculate pixel positions with reference to the rover position being at the 
    # center bottom of the image.  
    x_pixel = -(ypos - binary_img.shape[0]).astype(np.float)
    y_pixel = -(xpos - binary_img.shape[1]/2 ).astype(np.float)
    return x_pixel, y_pixel


# Define a function to convert to radial coords in rover space
def to_polar_coords(x_pixel, y_pixel):
    # Convert (x_pixel, y_pixel) to (distance, angle) 
    # in polar coordinates in rover space
    # Calculate distance to each pixel
    dist = np.sqrt(x_pixel**2 + y_pixel**2)
    # Calculate angle away from vertical for each pixel
    angles = np.arctan2(y_pixel, x_pixel)
    return dist, angles


# Define a function to map rover space pixels to world space
def rotate_pix(xpix, ypix, yaw):
    # Convert yaw to radians
    yaw_rad = yaw * np.pi / 180
    xpix_rotated = (xpix * np.cos(yaw_rad)) - (ypix * np.sin(yaw_rad))
                            
    ypix_rotated = (xpix * np.sin(yaw_rad)) + (ypix * np.cos(yaw_rad))
    # Return the result  
    return xpix_rotated, ypix_rotated


def translate_pix(xpix_rot, ypix_rot, xpos, ypos, scale): 
    # Apply a scaling and a translation
    xpix_translated = (xpix_rot / scale) + xpos
    ypix_translated = (ypix_rot / scale) + ypos
    # Return the result  
    return xpix_translated, ypix_translated


# Define a function to apply rotation and translation (and clipping)
# Once you define the two functions above this function should work
def pix_to_world(xpix, ypix, xpos, ypos, yaw, world_size, scale):
    # Apply rotation
    xpix_rot, ypix_rot = rotate_pix(xpix, ypix, yaw)
    # Apply translation
    xpix_tran, ypix_tran = translate_pix(xpix_rot, ypix_rot, xpos, ypos, scale)
    # Perform rotation, translation and clipping all at once
    x_pix_world = np.clip(np.int_(xpix_tran), 0, world_size - 1)
    y_pix_world = np.clip(np.int_(ypix_tran), 0, world_size - 1)
    # Return the result
    return x_pix_world, y_pix_world


# Define a function to perform a perspective transform
def perspect_transform(img, src, dst):
           
    M = cv2.getPerspectiveTransform(src, dst)
    warped = cv2.warpPerspective(img, M, (img.shape[1], img.shape[0]))# keep same size as input image

    mask = cv2.warpPerspective(np.ones_like(img[:,:,0]), M, (img.shape[1], img.shape[0]))
    
    return warped, mask


def color_thresh_rock_samples(img):
    # Create an array of zeros same xy size as img, but single channel
    color_select = np.zeros_like(img[:, :, 0])
    # Require that each pixel be above all three threshold values in RGB
    # above_thresh will now contain a boolean array with "True"
    # where threshold was met
    # rock_thresh = ((img[:, :, 0] > 110) \
    #                & (img[:, :, 1] > 110) \
    #                & (img[:, :, 2] < 50))
    rock_thresh =  ((img[:, :, 2] > 0) & (img[:,:,2] < 110) \
                   & (img[:, :, 0] > 145) & (img[:,:,0] < 255)) \

    color_select[rock_thresh] = 1
    # Return the binary image
    return color_select


# Apply the above functions in succession and update the Rover state accordingly
def perception_step(Rover):
    # Perform perception steps to update Rover()

    # NOTE: camera image is coming to you in Rover.img
    # 1) Define source and destination points for perspective transform
    # Define calibration box in source (actual) and destination (desired) coordinates
    # These source and destination points are defined to warp the image
    # to a grid where each 10x10 pixel square represents 1 square meter
    # The destination box will be 2*dst_size on each side
    dst_size = 5
    # Set a bottom offset to account for the fact that the bottom of the image
    # is not the position of the rover but a bit in front of it
    # this is just a rough guess, feel free to change it!
    bottom_offset = 6
    source = np.float32([[14, 140], [301, 140], [200, 96], [118, 96]])
    destination = np.float32([[Rover.img.shape[1] / 2 - dst_size, Rover.img.shape[0] - bottom_offset],
                              [Rover.img.shape[1] / 2 + dst_size, Rover.img.shape[0] - bottom_offset],
                              [Rover.img.shape[1] / 2 + dst_size, Rover.img.shape[0] - 2 * dst_size - bottom_offset],
                              [Rover.img.shape[1] / 2 - dst_size, Rover.img.shape[0] - 2 * dst_size - bottom_offset],])
    
    # 2) Apply perspective transform
    warped, mask = perspect_transform(Rover.img, source, destination)
    
    # 3) Apply color threshold to identify navigable terrain/obstacles/rock samples
    # Identify pixels above the threshold
    # find drivable area
    threshed_navigable = color_thresh(warped)
    # find rocks
    threshed_rock = color_thresh_rock_samples(Rover.img)
    # find obstacles
    # This is basically the "inverse" of the threshed_navigable
    threshed_obstacles = (threshed_navigable - 1) * (-1)
    
    # 4) Update Rover.vision_image (this will be displayed on left side of screen)
    Rover.vision_image[:, :, 0] = threshed_obstacles * 255
    Rover.vision_image[:, :, 1] = threshed_rock * 255
    Rover.vision_image[:, :, 2] = threshed_navigable * 255
    
    # 5) Convert map image pixel values to rover-centric coords
    # Calculate pixel values in rover-centric coords and distance/angle to all pixels
    xpix, ypix = rover_coords(threshed_navigable)
    dist, angles = to_polar_coords(xpix, ypix)
    mean_dir = np.mean(angles)
    # calculate rock pixels centric coordinates
    xpix_rock, ypix_rock = rover_coords(threshed_rock)
    dist_rock, angles_rock = to_polar_coords(xpix_rock, ypix_rock)
    mean_dir_rock = np.mean(angles_rock)
    # centric coordinates obstacles
    xpix_obs, ypix_obs = rover_coords(threshed_obstacles)
    dist_obs, angles_obs = to_polar_coords(xpix_obs, ypix_obs)
    mean_dir_obs = np.mean(angles_obs)
    
    # 6) Convert rover-centric pixel values to world coordinates
    scale = 10
    # Get navigable pixel positions in world coords
    x_world, y_world = pix_to_world(xpix, ypix, Rover.pos[0],
                                    Rover.pos[1], Rover.yaw,
                                    Rover.worldmap.shape[0], scale)
    
    x_rock_world, y_rock_world = pix_to_world(xpix_rock, ypix_rock, Rover.pos[0],
                                              Rover.pos[1], Rover.yaw,
                                              Rover.worldmap.shape[0], scale)
    
    x_obs_world, y_obs_world = pix_to_world(xpix_obs, ypix_obs, Rover.pos[0],
                                              Rover.pos[1], Rover.yaw,
                                              Rover.worldmap.shape[0], scale)
    
    # 7) Update Rover worldmap (to be displayed on right side of screen)
    Rover.worldmap[y_obs_world, x_obs_world, 0] += 10
    Rover.worldmap[y_rock_world, x_rock_world, 1] = 255
    Rover.worldmap[y_world, x_world, 2] += 1
    
    
    # 8) Convert rover-centric pixel positions to polar coordinates
    # Update Rover pixel distances and angles
    Rover.nav_dists = dist
    Rover.nav_angles = angles

    # # Are we seeing some rocks? If yes, pick it up!
    # rock_world_pos = Rover.worldmap[:, :, 1].nonzero()
    # # If there are, we'll step through the known sample positions
    # # to confirm whether detections are real
    # if rock_world_pos[0].any():
    #     rock_size = 2
    #     for idx in range(len(Rover.samples_pos[0])):
    #         test_rock_x = Rover.samples_pos[0][idx]
    #         test_rock_y = Rover.samples_pos[1][idx]
    #         rock_sample_dists = np.sqrt((test_rock_x - rock_world_pos[1]) ** 2 + \
    #                                     (test_rock_y - rock_world_pos[0]) ** 2)
    #         # If rocks were detected within 3 meters of known sample positions
    #         # consider taking it
    #         if np.min(rock_sample_dists) < 0.1:
    #             Rover.near_sample = 1

    return Rover