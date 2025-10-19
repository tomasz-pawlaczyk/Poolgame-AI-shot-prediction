# Poolgame analysis and prediction system

An AI-powered system that analyzes pool (billiards) table images. It combines computer vision for ball and table recognition with mathematic formulas to suggest the best move possible.

<hr>



## Table of Contents

- [Overview](#overview)
- [Key Features](#key-features)
- [Math-backed Code](#Math-backed-code)
- [Methodology & Code Deep Dive](#methodology--code-deep-dive)

## Overview

This project aims to assist billiard players of all skill levels by providing AI-driven shot recommendations. By simply uploading a picture of the pool table, the system:

1. Corrects perspective of the photo
  
2. Normalises colors for consistent analysis
  
3. Detects and classifies balls
  
4. Generates hundreds of possible shots
  
5. Validates the shots
  
6. Recommends the best shots for the player and displays them
  

![zestwienie](img/zestawienie%20napisy.png)







<!--
![gora](img/gora.png)
**&emsp;&emsp;&emsp;&emsp;&emsp;&emsp;1. Input photo &emsp;&emsp;&emsp;&emsp;&emsp;&emsp;&emsp;&emsp;2. Transformation &emsp;&emsp;&emsp;&emsp;&emsp;&emsp;3. Color**
**&emsp;&emsp;&emsp;&emsp;&emsp;&emsp;&emsp;&emsp;&emsp;&emsp;&emsp;&emsp;&emsp;&emsp;&emsp;&emsp;&emsp;&emsp;&emsp;&emsp;&emsp;&emsp;&emsp;&emsp;&emsp;&emsp;&emsp;&emsp;&emsp;&emsp;&emsp;&emsp;&emsp;&emsp;&emsp;normalisation**

![gora](img/dol.png)
**&emsp;&emsp;4. Balls &emsp;&emsp;&emsp;&emsp;&emsp;&emsp;&emsp;&emsp; 5. Shots &emsp;&emsp;&emsp;&emsp;&emsp;&emsp;&emsp;&emsp; 6. Shots &emsp;&emsp;&emsp;&emsp;&emsp;&emsp; &emsp;&emsp;7. Final**
**&emsp;&emsp; &emsp;&emsp;detection &emsp;&emsp;&emsp;&emsp;&emsp;&emsp;&emsp;calculation &emsp;&emsp;&emsp;&emsp; &emsp;&emsp;&emsp;validation &emsp;&emsp;&emsp;&emsp;&emsp;&emsp; suggestion**
-->




## Key Features

|     | Feature | Description |
| --- | --- | --- |
| ✅   | **Smart Image Preprocessing** | Automatically corrects perspective and normalises colors for consistent analysis under any lighting conditions |
| ✅   | **AI Ball Detection** | Our YOLOv5 model detects and classifies all balls with over 92% accuracy, handling overlaps and difficult lighting |
| ✅   | **Shot Recommendation** | Generates and ranks hundreds of possible shots based on angle, distance, and complexity to suggest the 3 best options. |
| ✅   | **Physics-Based Simulation** | Implements a simplified physical model for cue ball trajectory prediction after collision, accounting for friction |

## Math-backed code


# 3.2 Physics of Billiard Ball Collision (Summary)

# 3.2 Physics of Billiard Ball Collision (Summary)

The motion of a billiard ball combines energy transfer from the cue, friction, rotational motion, and collision dynamics.  
The initial energy of the cue ball (Ep) is divided into kinetic energy (Ek) and frictional work (Wt):

Ep = Ek + Wt

For a rolling sphere:
Ek = (1/2) * m * v² + (1/2) * I * ω² = (7/10) * m * v²  
v = sqrt(10 * Ek / (7 * m))

During motion, the ball loses speed due to friction:
Ft = -μ * g * m  
a = -μ * g  
v = sqrt(v₀² - 2 * μ * g * d)

The collision is described by the coefficient of restitution (COR ≈ 0.8).  
Along the collision axis (y-axis):

v1yₖ = (1 - COR)/2 * v1yₚ  
v2yₖ = (1 + COR)/2 * v1yₚ

The cue ball keeps its horizontal velocity:  
v1xₖ = v1xₚ, and the target ball starts from rest: v2xₖ = 0.

The new trajectory angle of the cue ball:

θ = arctan( sin(α) / ((1 - COR) * cos(α)) )

The distance traveled after the collision:

d = (v² - v₀²) / (2 * (-μ) * g)

Final coordinates of the cue ball after it stops:

x' = x - d / sqrt( tan²(θ - α) + 1 )  
y' = y - (d * tan(θ - α)) / sqrt( tan²(θ - α) + 1 )

This simplified physical model estimates post-collision trajectories and velocities, accounting for friction, rotation, and inelastic impact effects.

This sentence uses $\` and \`$ delimiters to show math inline: $`\sqrt{3x-1}+(1+x)^2`$






# 3.2 Physics of Billiard Ball Collision (Summary)

The motion of a billiard ball combines energy transfer from the cue, friction, rotational motion, and collision dynamics.  
The initial energy of the cue ball ($E_p$) is divided into kinetic energy ($E_k$) and frictional work ($W_t$):

$$
E_p = E_k + W_t
$$

For a rolling sphere:

$$
E_k = \frac{1}{2} m v^2 + \frac{1}{2} I \omega^2 = \frac{7}{10} m v^2
$$

$$
v = \sqrt{\frac{10 E_k}{7 m}}
$$

During motion, the ball loses speed due to friction:

$$
F_t = -\mu g m
$$

$$
a = -\mu g
$$

$$
v = \sqrt{v_0^2 - 2 \mu g d}
$$

The collision is described by the coefficient of restitution (COR ≈ 0.8).  
Along the collision axis (y-axis):

$$
v_{1y_k} = \frac{1 - COR}{2} \, v_{1y_p}
$$

$$
v_{2y_k} = \frac{1 + COR}{2} \, v_{1y_p}
$$

The cue ball keeps its horizontal velocity:

$$
v_{1x_k} = v_{1x_p}
$$

and the target ball starts from rest:

$$
v_{2x_k} = 0
$$

The new trajectory angle of the cue ball:

$$
\theta = \arctan\left( \frac{\sin(\alpha)}{(1 - COR)\cos(\alpha)} \right)
$$

The distance traveled after the collision:

$$
d = \frac{v^2 - v_0^2}{2(-\mu) g}
$$

Final coordinates of the cue ball after it stops:

$$
x' = x - \frac{d}{\sqrt{\tan^2(\theta - \alpha) + 1}}
$$

$$
y' = y - \frac{d \tan(\theta - \alpha)}{\sqrt{\tan^2(\theta - \alpha) + 1}}
$$

This simplified physical model estimates post-collision trajectories and velocities, accounting for friction, rotation, and inelastic impact effects.

This sentence uses $\` and \`$ delimiters to show math inline: $`\sqrt{3x-1}+(1+x)^2`$







## Methodology & Code Deep Dive

1. **Perspective Correction & Color Normalization** 
    
    Before analysis, the image is standardized. We detect the table's corner pockets using Hough Circle Transform and warp the perspective to a consistent top-down view. Colors are then normalized based on the white ball to counteract different lighting conditions.
  
      ```python
      # Snippet: Color Normalization using the white ball
      from PIL import Image, ImageStat
      import numpy as np
      
      def normalize_colors(img, white_ball_center, white_ball_radius):
          """ Normalizes image colors using the white ball as a reference. """
          # Extract region of interest (ROI) around the white ball
          x, y = white_ball_center
          r = white_ball_radius
          roi = img.crop((x-r, y-r, x+r, y+r))
      
          # Convert ROI to numpy array and create a circular mask
          roi_np = np.array(roi)
          h, w = roi_np.shape[:2]
          Y, X = np.ogrid[:h, :w]
          dist_from_center = np.sqrt((X - w/2)**2 + (Y - h/2)**2)
          mask = dist_from_center <= r
      
          # Get pixels within the white ball using the mask
          pixels = roi_np[mask]
      
          # Use a high percentile to avoid shadows and get the "true" white
          ref_color = np.percentile(pixels, 95, axis=0)
      
          # Calculate scaling factors for each RGB channel
          scale_factors = 255.0 / ref_color
      
          # Apply scaling to the entire image
          img_normalized = (img * scale_factors).clip(0, 255).astype(np.uint8)
          return img_normalized
      ```
  

 2. **Ball Detection & Classification with YOLO**




4. **Trajectory visualization**

    Visualizes the cue ball trajectory after collision with direction indicated.
    ```python
    def draw_cue_trajectory(img, shot):
        """Draws the cue ball trajectory after collision"""
        
        start_point, end_point = shot.get_cue_trajectory_line(200)
    
        # Draw the trajectory line (blue color)
        cv.line(img, start_point, end_point, (255, 0, 0), 3)
    
        # Add an arrow indicating the direction
        angle_rad = math.radians(shot.get_cue_angle_after_collision())
        arrow_length = 15
        arrow_angle = math.radians(25)
    
        # Calculate the points of the arrowhead
        arrow_x1 = end_point[0] - arrow_length * math.cos(angle_rad - arrow_angle)
        arrow_y1 = end_point[1] - arrow_length * math.sin(angle_rad - arrow_angle)
    
        cv.line(img, end_point, (int(arrow_x1), int(arrow_y1)), (255, 0, 0), 3)
    ```

