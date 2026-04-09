TODO Hendrik: Add requests from professors on minor changes about the paper


# in the abstract: 
"define what "IK" is"
"either use the term "pipeline" or "system" throughout the entire paper instead of vaguely using both. For example only use pipeline (not IK pipeline and clearly define it, then stick to that.

replace "where hand occlusion by surrounding objects reduces
success to 9.3% (N = 75)" by "and find that success is reduced to 9.3% due to..."

# Related work
"Throughout the paper you use classes and annotations defined in the code (like input / outputs functions from table 1) which doesn't really makes sense in the paper,it is not necessary to reuse the class and functions names in the paper" (remove Each pipeline stage is implemented as a
Transformation[InT, OutT] subclass that wraps a
_transform method with automatic latency tracking.)

Merge table 1 and figure 1 together since they are kinda the same thing.


III. SYSTEM OVERVIEW
Create a diagram of the hardware were you define the angles used (camera angles, camera height relative the base for the translation etc), gripper angles

Remove figure 2 picture (robot.jpg) and replace with diagram as well as the hardware_description.png picture (shows annotated image of the robot)

IV. METHODS

again no "the pyrealsense2 SDK"

"you define (fx,fy)  and (cx, cy) but don't define (X,Y,Z) (Pcam?) define that too.


B. HAND POSE ESTIMATION
Equation 2 mistakes:
- we defined (ui,vi) but not (ui_raw, vi_raw) the output of mediapipe which is different
- define Pt_raw the array of all (ui_raw, vi_raw) at time t
- Pt+1 = alpha * Pt_raw + (1-alpha) * Pt 
- "unpack" Pt into (ui, vi)


C. DEPTH-BASED 3D RECONSTRUCTION
- Define D[vi, ui], use D[ui, vi] to keep the order consistent. Also use a depth in meters not some scaled depth over s
- also instead of using [X,Y,Z] use Pcam which should be defined
- remove [dmin, dmax] since not used elsewhere
- remove THUMB_MCP or INDEX_FINGER_MCP and use P1 and P2 instead which should be defined and used throughout the paper (maybe using the mediapipe hand landmarks picture)


D. CAMERA-TO-ROBOT COORDINATE TRANSFORM

- instead of using Rfinal and tfinal in prob = Rfinal pcam + tfinal and subsequently. Use simple R ant t throughout.
- define angle theta in the previous system diagram


E. TARGET POSE COMPUTATION
- define (ptarget, qtarget)
- The resulting rotation matrix R = [e1 e2 e3] -> R should be qtarget
(same as before remove scipy.spatial.transform)
- drafted_gripper_vectors.png
- why do we estimate dhat = (uthumb + uindex)/ ||uthumb + uindex|| with uthumb index the unit vectors in p space? and not simply use d ? is this because in 3D/general case d needs to be estimated or something else?


F. INVERSE KINEMATICS
- use "by solving the following regularized least square problem" before the equation
- remove "position ptarget" replace by "pose (ptarget, qtarget)"
- remove "arm using PyBullet’s calculateInverseKinematics"
- this is Pybullet for the solver, mention it: The solver runs up to
100 iterations with a residual threshold of 10−4


G. GRIPPER CONTROL
- gripper control angle should be called "target" ?
- don't understand this part (Use THUMB_TIP and INDEX_FINGER_TIP (pri-
mary).

2) Use THUMB_IP and INDEX_FINGER_DIP (knuckle
fallback).
3) Use the last valid gripper angle (temporal persistence).
4) Use the mid-open default (φmin + φmax)/2.) Basically explain is simple terms with "a depth fallback mechanism  handles the critical case where exactly one of the two landmarks locating thumb and index finger are invalid."



H. SIMULATION PREVIEW
- remove with gravity g = 9.81 m/s2
- Joint commands
are executed via position control with uniform force limits of
6.0 N, position gains of 0.2, and velocity gains of 1.0. -> explain that the PID was empirically tuned to reflect the motion speed and motion of the real robot arm


Others:
- FIGURE 5. instead use a simple square matrice with the positions tile numbers as well as robotmountedcamera_pov.png
- without a clear matrice we cannot understand "IK pipeline: tiles #6–#9"
- the matrice is like this from the robot perspective:

                    [1, 2, 3]
[box on the left]   [4, 5, 6]
                    [7, 8, 9] 
            [robot]


V. EXPERIMENTAL SETUP
- remove Software: Python 3.10, PyBullet ≥3.2.5, MediaPipe
≥0.10, LeRobot ≥0.4.1, OpenCV ≥4.8.

Remove this author:
YU-CHUN TUAN is a graduate student in the Department of Mechanical
Engineering at the University of California, Berkeley.

and add this author at the top and end of the paper
Professor Gabriel Gomes as a coauthor (department of mechanical engineering) 
(some info about him from uc berkeley TAKE WHAT YOU NEED ONLY for the paper: Gabriel Gomes
Lecturer, Mechanical Engineering at UC Berkeley

headshot of Gabriel Gomes
Gabriel Gomes is a lecturer with the Mechanical Engineering Department at U.C. Berkeley. He received a doctorate degree in automatic control theory in 2004 from U.C. Berkeley. Since then, he has focused his research on various problems in the modeling, simulation, and control of traffic networks. As a lecturer, he has taught courses in partial differential equations, control theory, statistics, data science, and Python programming. He also supervises capstone projects with the Master of Engineering program of the Fung Institute. These projects cover a wide range of topics, including robotics, solar energy, machine learning, natural language processing, traffic simulation, reinforcement learning, and smart exercise machines. He is the author of over 50 papers in various areas of engineering.)


