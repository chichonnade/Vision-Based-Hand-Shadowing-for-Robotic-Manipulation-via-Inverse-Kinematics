
You, Jul 30, 12:19 PM, Edited
observation for run 1:
simulation is error (orange line) initally decreases very slowly from home position to the real commanded postion (slower than the real arm) = this means it's almost overly smooth, this could be caused either by the sim being laggy or very low P,D gains
also notice on both plots the blue shoulder pan line (controls how much the arm moves from left to right), the real robot seems to track this well, also it doesn't move much in real life. In the sim the left to right pan is more extreme
PS:  gripper torque config is currently commented out in the script, but the gripper seems to really lag behind on the real robot
You, Jul 30, 12:32 PM
run 2 :
only slowed down the FPS to 5 (to give the robot more time to reach positions)

You, Jul 30, 12:37 PM
run 2 simulation graph looks the same at 5 fps

You, Jul 30, 12:41 PM
as Isaac suggested, the jitter in the error is probably coming from the commanded actions in the npy in the first place which are jerky
You, Jul 30, 12:46 PM
Run 3:
kept 5 fps to provide the robot plenty of time to reach the position
increased P,D to make it track aggressively (note: there is a max accel and max velocity that is set still)

Observation:
real robot tracks more aggressively the small movements since it has the time to reach them
there is still no large movements
next run: remove the accel and speed limit

You, Jul 30, 12:49 PM, Edited
Run 4:
set the acceleration and speed limits much higher (x3)

Observation:
all the joints are very responsive, but the gripper motion is always very sluggish, wondering if LeRobot intentionally has a parameter to slow down the gripper
This is trivial at the beginning of the brown line, regardless of the PID, accel and speed params it always moves very slowly, the commanded gripper pos obviously can be very jerky and explains why the error is oscillating

Antoine Jamme, Jul 30, 12:58 PM
Good runs, so plan is to keep those P, D and max accel/velocity limits?
You, Jul 30, 12:59 PM
Actually the plots are misleading, it is good to have a jerky plot because the commanded position is very very bad so not following it closely is good
Antoine Jamme, Jul 30, 1:02 PM
You mean the forward kinematics of all joint commands is off from the coordinates returned by mediapipe + depth + tranform (target_pos)?
What's " bad"?
Isaac Neal, Jul 30, 1:07 PM
Yes this is what I was trying to say on the meeting haha
Oscillating around 0 is good, it means the errors aren’t biased and are just smoothing noise
You, Jul 30, 1:07 PM
yeah I guess mediapipe+IK is jittery especially for the wrist flex (orientation of the wrist + closing of the fingers)

Isaac Neal, Jul 30, 1:08 PM
QuotedQuoted Sent by Antoine JammeWhat's " bad"?End Quote press L to link back to original quoteEnd Quote press L to link back to original quoteIt’s probably just hand detection gives some jitter
Sorry had no data my messages are only sending now
Antoine Jamme, Jul 30, 1:11 PM, Edited
So what's the smoothest .npy (better bag) that we can use for the demo, there's probably some work on the IK to do but for the purpose of the demo, what can be good enough to grab something and drop it into the bucket?
Actually having kinda jittery movements for the demo is a good sign where's not faking it right
Isaac Neal, Jul 30, 1:13 PM
It doesn’t matter if it’s kinda jittery, we just need a simple action that works irl
Antoine Jamme, Jul 30, 1:13 PM
Right
You, Jul 30, 1:13 PM, Edited
the poor tracking is mostly on the wrist flex (hand orientation) 

we can see in the command plot above that the "noise pattern" period is <=5 frames so having ~5 frames or lower oscillations in the error plot is good, it means we don't follow the jitter
Isaac Neal, Jul 30, 1:13 PM
Im working on a script that allows you to setup the robot, capture a demo, and replay in the sim right there, then if you approve replay on the robot
Should take ~1 min for a short  action
Hendrik can try that at home and if it works can run it in Safeway tonight to capture new simple actions
Plus can preprocess some promising traces as backup
Just got home from my Spanish lesson tho so gonna eat lunch then finish the script
Antoine Jamme, Jul 30, 1:16 PM
Yeah I mean the target_pos and wrist flex are really close to each other it might affect the IK solutions, idk this is some technical problem we can figure out with more time, at the moment we just need a video for the demo. So good job!
Isaac Neal, Jul 30, 1:19 PM
Yeah we will figure all that stuff out I’m sure
Isaac Neal, Jul 30, 2:22 PM
I’ve been cooking rice for legit like 45 minutes and it’s still hard
😮‍💨
You, Jul 30, 2:22 PM
what the rice cooker plugged in :)
You, Jul 30, 2:23 PM, Edited
or maybe the heat tracking of ur rice cooker needs a higher P value
Isaac Neal, Jul 30, 2:25 PM
Fuck didn’t vibetune with Claude Arroz
The PID is surely off
Claúd
You, Jul 30, 2:53 PM, Edited
I think my base calibration was slightly off as I didn't fully move the shoulder pan yesterday to prevent damage 

I recalibrated and the pan is larger which is in line with the video

also added send_standby() similar to send_home but that sets the arm in a pos that would make it fall down when the torque is disabled at the end
Antoine Jamme, Jul 30, 3:18 PM
Good idea cause home home was not a stable position without power
Antoine Jamme, Jul 30, 3:23 PM
I was thinking since you have some TPU and we have the macro that creates the eflesh structure, I could design some pads we could screw to the jaws to have a better griping
You, Jul 30, 3:33 PM, Edited
Ok I fixed the slow gripper, but I'm not sure why.

To compensate for the gripper being slower I simply applied different acceleration and velocity limits to it :
    acceleration_limit: int = 50 
    velocity_limit: int = 1500 
    gripper_acceleration_limit: int = 200  
    gripper_velocity_limit: int = 5000  

note: also added the gripper torque limit (0-1000) but when set to something low like 50 it "blocks" the desired speed up and acceleration effect

Isaac Neal, Jul 30, 3:34 PM
Yeah makes sense RE the torque
I think we need to figure out the backoff thing where it stops applying torque after encountering an object
But not rn, we should just grab soft things like candy bars or bags of chips
Instead of e.g. glass jars
My script is ready, wanna try it out?
You, Jul 30, 3:35 PM
https://meet.google.com/owb-nkgv-vvm

