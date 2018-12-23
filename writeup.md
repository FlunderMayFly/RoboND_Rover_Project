## Project: Search and Sample Return
### Writeup Template: You can use this file as a template for your writeup if you want to submit it as a markdown file, but feel free to use some other method and submit a pdf if you prefer.

---


**The goals / steps of this project are the following:**  

**Training / Calibration**  

* Download the simulator and take data in "Training Mode"
* Test out the functions in the Jupyter Notebook provided
* Add functions to detect obstacles and samples of interest (golden rocks)
* Fill in the `process_image()` function with the appropriate image processing steps (perspective transform, color threshold etc.) to get from raw images to a map.  The `output_image` you create in this step should demonstrate that your mapping pipeline works.
* Use `moviepy` to process the images in your saved dataset with the `process_image()` function.  Include the video you produce as part of your submission.

**Autonomous Navigation / Mapping**

* Fill in the `perception_step()` function within the `perception.py` script with the appropriate image processing functions to create a map and update `Rover()` data (similar to what you did with `process_image()` in the notebook). 
* Fill in the `decision_step()` function within the `decision.py` script with conditional statements that take into consideration the outputs of the `perception_step()` in deciding how to issue throttle, brake and steering commands. 
* Iterate on your perception and decision function until your rover does a reasonable (need to define metric) job of navigating and mapping.  

[//]: # (Image References)

[image1]: ./documentation/rock_docu.PNG
[image2]: ./documentation/threshed_obs_docu.PNG
[image3]: ./calibration_images/example_rock1.jpg 
[grid_image]: ./calibration_images/example_grid1.jpg
[warped_grid]: ./documentation/warped_grid.PNG
[rover_perspect]: ./documentation/rover_perspective.PNG
[rover_coords]: ./documentation/rover_coords.PNG
[simulator]: ./documentation/simulator.PNG
[video1]: ./output/test_mapping.mp4

## [Rubric](https://review.udacity.com/#!/rubrics/916/view) Points
### Here I will consider the rubric points individually and describe how I addressed each point in my implementation.  

---
### Writeup / README

### Notebook Analysis
#### 1. Run the functions provided in the notebook on test images (first with the test data provided, next on data you have recorded). Add/modify functions to allow for color selection of obstacles and rock samples.
To identify rock samples, I wrote the following function. The thresholds came by testing different color combinations as well as 
identifying the color in a picture with a gold sample.
````python
def color_thresh_rock_samples(img):
    # Create an array of zeros same xy size as img, but single channel
    color_select = np.zeros_like(img[:,:,0])
    # Require that each pixel be above all three threshold values in RGB
    # above_thresh will now contain a boolean array with "True"
    # where threshold was met
    rock_thresh = ((img[:, :, 0] > 145) & (img[:,:,0] < 255)) \
                   & ((img[:, :, 2] > 0) & (img[:,:,2] < 110))
    color_select[rock_thresh] = 1
    # Return the binary image
    return color_select
````
![alt text][image1]

To identify obstacles I simply "inverted" the freespace pixels, saying; all that is not drivable is an obstacle. It did 
this by using the following code line
````python
threshed_obstacles = (threshed_navigable - 1) * (-1)
````
![alt text][image2]


#### 1. Populate the `process_image()` function with the appropriate analysis steps to map pixels identifying navigable terrain, obstacles and rock samples into a worldmap.  Run `process_image()` on your test data using the `moviepy` functions provided to create video output of your result. 

````python
def process_image(img):
    # Example of how to use the Databucket() object defined above
    # to print the current x, y and yaw values 
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
    source = np.float32([[14, 140], [301 ,140],[200, 96], [118, 96]])
    destination = np.float32([[img.shape[1]/2 - dst_size, img.shape[0] - bottom_offset],
                      [img.shape[1]/2 + dst_size, img.shape[0] - bottom_offset],
                      [img.shape[1]/2 + dst_size, img.shape[0] - 2*dst_size - bottom_offset], 
                      [img.shape[1]/2 - dst_size, img.shape[0] - 2*dst_size - bottom_offset],
                      ])
    
    # 2) Apply perspective transform
    warped, mask = perspect_transform(img, source, destination)
    
    # 3) Apply color threshold to identify navigable terrain/obstacles/rock samples
    # Identify pixels above the threshold
    # find drivable area
    threshed_navigable = color_thresh(warped)
    # find rocks
    threshed_rock = color_thresh_rock_samples(Rover.img)
    # plt.imshow(threshed, cmap='gray')
    # find obstacles
    # This is basically the "inverse" of the threshed_navigable
    threshed_obstacles = (threshed_navigable - 1)
    
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

    # 7) Make a mosaic image, below is some example code
        # First create a blank image (can be whatever shape you like)
    output_image = np.zeros((img.shape[0] + data.worldmap.shape[0], img.shape[1]*2, 3))
        # Next you can populate regions of the image with various output
        # Here I'm putting the original image in the upper left hand corner
    output_image[0:img.shape[0], 0:img.shape[1]] = img

        # Let's create more images to add to the mosaic, first a warped image
    warped, mask = perspect_transform(img, source, destination)
        # Add the warped image in the upper right hand corner
    output_image[0:img.shape[0], img.shape[1]:] = warped

        # Overlay worldmap with ground truth map
    map_add = cv2.addWeighted(data.worldmap, 1, data.ground_truth, 0.5, 0)
        # Flip map overlay so y-axis points upward and add to output_image 
    output_image[img.shape[0]:, 0:data.worldmap.shape[1]] = np.flipud(map_add)


        # Then putting some text over the image
    cv2.putText(output_image,"Populate this image with your analyses to make a video!", (20, 20), 
                cv2.FONT_HERSHEY_COMPLEX, 0.4, (255, 255, 255), 1)
    data.count += 1 # Keep track of the index in the Databucket()
    
    return output_image
````
Video:<br>
![alt text][video1]


### Autonomous Navigation and Mapping

#### 1. Fill in the `perception_step()` (at the bottom of the `perception.py` script) and `decision_step()` (in `decision.py`) functions in the autonomous mapping scripts and an explanation is provided in the writeup of how and why these functions were modified as they were.
The ``perception_steps`` are defined as the following main steps:
1. Defenition of source and destination points. These are defined by writing down pixel position from a sample
picture which has to contain the grid on the ground. The 4 pixel points define a square meter on the ground. 
The destination pixels are in scale 1/10 meter. 
![alt text][grid_image] 
2. perspective transform which transforms the predefined source and destination pixels. 
This guarantees a deformation free map image.
![alt_text][warped_grid]
3. Apply color thresholds to evaluate picture whether areas are 
    1. drivable (white / bright areas in pictures mark ground). <br>
        chosen colors: ``[160 160 160]``
    2. Obstacle ("inverse" of drivable area -> dark)
    3. Rock (Yellow colorscheme)
        chosen colors 
        ````python 
        rock_thresh1 =  ((img[:, :, 2] > 0) & (img[:,:,2] < 110) \
                      & (img[:, :, 0] > 145) & (img[:,:,0] < 255)) \
        ````
        These colors were found as described at point 1 in this markdown. 
        It seems that they are not good for identifying (see picture right below). However,
        If I choose more identical colors, the robot does not find the rocks. Just when He is
        right before the rock. 
        ````python
        rock_thresh2 = ((img[:, :, 0] > 110) \
                      & (img[:, :, 1] > 110) \
                      & (img[:, :, 2] < 50))
        ````
        The thresholds of ``rock_thresh2`` let the rocks shine solid and bright in the vision of the rover. However,
        they appear smaller. Now, the thresholds of I choose (``rock_thresh2``) do not cover the center of the rock, but
        the frame, so the rocks appear much bigger and more stable. This leds the rover see rocks, even
        if they're further away.
4. Update Rover vision image
    The rover vision image is on the bottom left in the screen. Here will be displayed what the
    rover sees as well as the color thresholds we applied on the incoming image (upper half).
    ![alt_text][rover_perspect]
5. Convert threshed images to Rover coordinates <br>
    In this step we create a binary image of the vision and transform it into rover coordinates.
    Which means basically bring the seen navigable threshed pixels into the ego perspective of the rover. <br>
    Assuming the camera sits on the ground at the center of the coordinate frame, the x axis defines
    longitudinal (driving direction) and y axis lateral by right-hand-side rule the following image
    results (right).
    ![alt_text][rover_coords]
    The red arrow on the right image describes the mean angle of the navigable terrain. This will be
    the steering angle which let the rover navigate through this terrain.
6. transform rover coordinates to world coordinates
    ````python
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
    ````
    This is basically a so-called 5 parameter Helmert transformation. This transformation is applicable here
    because the axis of both coordinate systems are assumed to be perpendicular to each other. If 
    this wouldn't be the case, one would have to choose a 7 parameter transformation. <br>
    The five parameters are:
    ``(xpix, ypix, xpos, ypos, scale)`` 
7. Update worldmap with threshed image data <br>
Updating the worldmap to see how much is covered by mapping and where the rocks are.

8. Converting rover-centric pixel positions to polar coordinates <br>
This last step updates the rover actuators ``nav_dists`` and ``nav_angles`` which will be used to
move and steer the robot. Why does this need polar coordinates? Well, the rover is centered on the origin
of his own coordinate system. So if he needs to move somewhere (steer, throttle), he basically needs to
know where to go from his position. Polar coordintes define points on a plane related to an origin.
This origin is the rover. The angle and distance in polar coordinates tell the rover exactly
what the moving direction is. What to do with this information is then done by the decision part.

#### 2. Launching in autonomous mode your rover can navigate and map autonomously.  Explain your results and how you might improve them in your writeup.  

**Note: running the simulator with different choices of resolution and graphics quality may produce different results, particularly on different machines!  Make a note of your simulator settings (resolution and graphics quality set on launch) and frames per second (FPS output to terminal by `drive_rover.py`) in your writeup when you submit the project so your reviewer can reproduce your results.**

Run configuration: 
1. Computer: Thinkpad, Windows, i5, 2048 mb RAM
2. Simulator: Roversim_x86
    ![alt text][simulator]

#### Results
The rover navigates autonomously through the environment and gets itself free when he is stuck. 
He identifies rocks reliable. If he passes one and has it on the "screen" the rock is always counted.
He also does not count any rocks if there are none. <br>
The mapping works reliable as well, however, the fidelty goes low once in a while. This is strongly dependend
on what the starting position is / was. If the rover starts doing a large circle to orientate the fidelty goes 
lower than if the rover starts moving straight forward.

#### Improvements
I try to implement the code to collect rocks and bring them back. However, I want to send in the project within
due time, so I will do this afterwards and update my GitHub project with it. <br>
What definetly can be improved is, when the rover sits on an edge where he has not enough power with the given
throttle to get further and rolls back, he tries again and again. I could build in a function which 
checks the groundspeed and if it turns below zero the rover could turn a little bit. However, this causes problems sometimes,
that is why i commented all these improvements out. <br>
Another improvement could be to check if the rover is on ground where he already mapped until the end of 
the "valley" (dead end). If yes, he could turn around immediately, if the quest is to map the whole area
in the shortest amount of time.

