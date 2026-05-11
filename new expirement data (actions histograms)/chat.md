still failing to make the arm work
Isaac Neal, Jul 29, 12:41 PM
I mean copy the code exactly from the teleoperation lerobot code right?
Then just modify it until you figure out what's failing
Since we know that works there must be something between that and the script we wrote that makes it fail
Personally I would probably hand write a really simple test that sets the robot to a known pose and confirms it can reach it
Should be like 20 lines of code, no agent, just copy paste from existing files
You, Jul 29, 12:47 PM
I tried a bunch of stuff already
Isaac Neal, Jul 29, 12:47 PM
And what were the results?
You, Jul 29, 12:48 PM
same as yesterday
Isaac Neal, Jul 29, 12:48 PM
So it can or cannot reach a fixed position?
E.g. [0.1, 0.1, 0.1, 0.1, 0.1]
If not, where does it go? Does that imply we're sending angles in the wrong format?
You, Jul 29, 12:49 PM, Edited
i haven't tried that, and what would this unit mean ? there is 4 different units we use

rad, degree, servo positions, and normalized m100_100
Isaac Neal, Jul 29, 12:50 PM
The idea is to figure out how to get the robot to assume a given pose, so I would experiment with sending data in a variety of formats till we figure out which is right
You, Jul 29, 12:50 PM
the config min max are in servo pos
the npy is rad
that's all we know
Isaac Neal, Jul 29, 12:50 PM
Right, that's what I would try to find out through this script
Alternatively you could open the lerobot teleoperation script and add print statements to log exactly what data is being sent to the robot during teleoperation
You, Jul 29, 12:51 PM
also there is 2 calls that affect the memory, make robot from config() and connect()
Isaac Neal, Jul 29, 12:51 PM
And compare that to what the data looks like that's being sent to the robot in our script
Isaac Neal, Jul 29, 12:54 PM
Q1: Are make_robot_from_config and connect called in both lerobot teleoperate and our script, and are they called in the same order as in our script?
Q2: Does the data format being sent to the robot in the lerobot teleop script match exactly the format being sent to the robot in our script? (add logging to check)
Q3: Does anything look different between the way they control the robot in the lerobot teleop script and the way we control it in our script? (can use Claude in ask mode with reference to both files)
These are the questions I would start w
You, Jul 29, 12:59 PM
I can dig into these question, would you have time to do that too if you clone lerobot ? I'm not sure I'll be able to get that robot working by myself
Isaac Neal, Jul 29, 12:59 PM
I can do it the problem is I can't run anything without the robot
https://open-vsx.org/extension/typefox/open-collaboration-tools
I wonder if this will work with cursor
Then I can control your IDE
Would definitely help me speed things up
You, Jul 29, 1:00 PM
I know but most of these questions above for instance don't need you to do that
Isaac Neal, Jul 29, 1:00 PM
Yeah I mean send me a copy of your repo and I will check it out
I'll do what I can without being able to run it
You, Jul 29, 1:01 PM
i can either push lerebot old or lerobot new
Isaac Neal, Jul 29, 1:01 PM
The new one is cleaner I think
Isaac Neal, Jul 29, 1:04 PM
Let me know when you've pushed it and I'll answer whatever questions I can and also try to write you some test scripts that will confirm or deny our hypotheses about what's going wrong
Teleop script working means it
1. isn't an issue with calibration
2. isn't a hardware issue
3. isn't a driver issue
It basically only leaves some weirdness with our script
I'm confident we will figure it out today
You, Jul 29, 1:07 PM
https://github.com/OMGrab/lerobotnew
Isaac Neal, Jul 29, 1:07 PM
Ty
Gonna make lunch then get to work
I'll make a branch so you can try my scripts out
Logging this here so we can try this later

Apparently this version works with cursor
I used it in my last job and it was kinda buggy but it was like multiplayer coding
Antoine Jamme, Jul 29, 1:14 PM
It's freaking a mess with units, the angles are using 2 bytes so the range is [0:8191] in servo pos unit and 0.087 degrees/unit is the resolution but that doesn't make sense because 8191x0.087 = ~712 degrees of range (like almost 2 turns)
Also I don't understand if it reads negative values the translation is confusing
Isaac Neal, Jul 29, 1:30 PM
I mean even if the physical range is limited since they're using 2 bytes you can send data in that range
[0, 8191]
Except that should be 13 bits lol wait a minute
Not 16
Who the fuck knows
Anyway lerobot should abstract all of this away from us, we just need to know the mode and how it works
And we give data to lerobot in whatever format it expects and it knows how to talk to the servos
We as programmers should talk to the servos directly as infrequently as possible
Therein lies so many errors
You, Jul 29, 1:34 PM, Edited
insights on teleoperate.py

1. it simply connects both arms
teleop = make_teleoperator_from_config(cfg.teleop)
robot = make_robot_from_config(cfg.robot)

 teleop.connect()
 robot.connect()

then reads actions from teleop and sends them to robot in a loop as below:

action = teleop.get_action()
robot.send_action(action)

that is it

here is what the "action" variable looks like:

{'shoulder_pan.pos': 20.088138082996693, 'shoulder_lift.pos': -99.16142557651992, 'elbow_flex.pos': 98.65168539325845, 'wrist_flex.pos': 40.90136054421768, 'wrist_roll.pos': -12.018730489073874, 'gripper.pos': 2.1209740769835035}

action is a dict of value, I tested them live and all of them go from -100 to +100 except the 'gripper.pos" that goes from 0 to 100.

Note: the keys have .pos in the name !
Isaac Neal, Jul 29, 1:34 PM
Interesting
So why would that work but our script doesn't?
I think there's something more complex at play. I'm going to start with that teleoperation script and simply read the npy actions, convert them to the right range, and play them back
Nothing else
You can run it and home and tell me what happens
You, Jul 29, 1:45 PM, Edited
from there I took playback_actions.py script and simply commented out robot.send_action(action_dict) to avoid breaking the arm and just printed out some action_dict (which would be incorrect if sent):

{'shoulder_pan.pos': np.float32(92.10536), 'shoulder_lift.pos': -100, 'elbow_flex.pos': np.float32(4.459717), 'wrist_flex.pos': np.float32(83.16397), 'wrist_roll.pos': np.float32(42.758972), 'gripper.pos': 0}

{'shoulder_pan.pos': 100, 'shoulder_lift.pos': -100, 'elbow_flex.pos': np.float32(-14.863625), 'wrist_flex.pos': 100, 'wrist_roll.pos': np.float32(46.20955), 'gripper.pos': np.float32(17.577066)}
You, Jul 29, 1:48 PM, Edited
Observations:
uses np.float32() for some reason
'wrist_flex.pos' goes from np.float32(83.16397) to np.float32(-14.863625) in a single time step which is a huge movement (range for each servo should is -100 to 100)
the key names is in line with the key names used in teleoperate.py
Isaac Neal, Jul 29, 1:50 PM
Weird. Maybe that bag had some bad data? Either way the PID should be tuned to be smooth enough to not be fucked up by the massive movements (like in the sim)
Antoine Jamme, Jul 29, 1:54 PM
It can be smooth between 2 frames but if it doesn't have time to get there because there's huge movement commands (suspicious) then it's gonna be lagging behind.
Huge movement should indicate that the IK solutions are too different from one another
Isaac Neal, Jul 29, 1:56 PM
Yeah but that isn’t the problem rn
I mean, it could be part of it, but if one huge movement fucks it up today and didnt three days ago, then something has changed
Antoine Jamme, Jul 29, 1:57 PM
Are we using the same bag?
Isaac Neal, Jul 29, 1:57 PM
Hendrik says this bag worked before
Antoine Jamme, Jul 29, 1:58 PM
Ok nvm
Isaac Neal, Jul 29, 1:58 PM
It does seem like a tough movement though, I would suggest we try another too
Even in the sim the arm jerks a lot
Isaac Neal, Jul 29, 2:37 PM
Okay back from lunch I'm starting to install everything now then will write the script
You, Jul 29, 2:49 PM
in playback_actions.py the relationship between the input (radian) from the sim and output (-100 +100 range) is non-linear and even not a bijection

first there is clamping that is applied to the input based on the max/min joint limit in the sim 

# Joint limits in simulation (radians) - same as teleoperate_sim.py
    'joint_limits': [
        (-1.92, 1.92),     # Joint 1: shoulder_pan
        (-1.75, 1.75),     # Joint 2: shoulder_lift  
        (-1.75, 1.57),     # Joint 3: elbow_flex
        (-1.66, 1.66),     # Joint 4: wrist_flex
        (-2.79, 2.79),     # Joint 5: wrist_roll
        (-0.17, 1.75)      # Joint 6: gripper (-10° to 100°)
    ]

so all the values that are greater that 1.92 for joint 1 will be clamped at +100. 
I guess we shouldn't have many impossible joint limits like that, but that is normal
I glanced over the values in and we don't have many +100 and -100 values so that means that the IK calculates correct sim-urdf compatible values mostly.

#### single sim_action #### : [ 2.0766485  -1.907291   -0.33673626  1.8656893   1.2892468   1.0674796 ]
 #### single real_action #### : [100.0, -100.0, -14.863624572753906, 100.0, 46.20954895019531, 17.57706642150879]
 #### single sim_action #### : [ 2.2871118 -1.8457497 -0.5132968  2.077784   1.3297888  1.1401486]
 #### single real_action #### : [100.0, -100.0, -25.49980926513672, 100.0, 47.66267395019531, 21.36191177368164]
Isaac Neal, Jul 29, 2:49 PM
Sounds correct to me
You, Jul 29, 2:50 PM
everytime I try a to modify the playback_actions it just never works
trying to duplicate teleoperation.py and go from there
Isaac Neal, Jul 29, 2:52 PM
I'm doing that rn if you want to try something else
Or we can both try but feels like maybe a waste of time
You, Jul 29, 2:55 PM
going to try too and we can compare it doesn't hurt
this thing is such a shit show
Isaac Neal, Jul 29, 2:55 PM
Sure
You, Jul 29, 2:55 PM
also note: the new lerobot now officially supports bi_so100_leader
Isaac Neal, Jul 29, 3:48 PM
We seem to have a decent number of negative values for the shoulder joint but I guess clipping them is harmless

Isaac Neal, Jul 29, 3:50 PM
Cursor is so useful for matplotlib it would have taken me like 30 mins to generate these otherwise
Isaac Neal, Jul 29, 3:53 PM
Okay so apparently we are only clipping the gripper angles in the main pipeline in the other repo 🤦

You, Jul 29, 3:55 PM
what should the logic be already if there is a list of 6 NaN actions in the middle of all actions ? should I just "clip" the lists containing NaN ?
Isaac Neal, Jul 29, 3:56 PM
NaN just indicates no hand was detected and thus no action inferred for that hand at that timestep
I would just skip it
if np.isnan(action).any():
  return None
For example
Isaac Neal, Jul 29, 4:17 PM
Okay I've finished my script
These are the raw action histograms (in radians)

Action histograms in motor space

You, Jul 29, 4:18 PM
ok I may have fixed it
Isaac Neal, Jul 29, 4:18 PM, Edited
The issue I see is that elbow_flex and shoulder_lift have many values above the max for that joint (returned by IK I guess?) that get mapped to 100% rotation on the motor
They look like outliers somewhat but idk
QuotedQuotedSent by Youok I may have fixed itEnd Quote press L to link back to original quoteEnd Quote press L to link back to original quoteWanna jump on a call?
Can try your fix then my script if that doesn't work
My script was 100% non-vibe coded besides the plotting
You, Jul 29, 4:25 PM
my head is going to blow up
Isaac Neal, Jul 29, 4:25 PM

Isaac Neal, Jul 29, 5:48 PM
We fixed it @Antoine Jamme
Both our scripts independently fixed it actually lol
Hendrik is gonna surf and I'm gonna have dinner then we'll get the script ready for recording in safeway
Antoine Jamme, Jul 29, 5:49 PM
Good job guys!
What was the problem ?
Isaac Neal, Jul 29, 5:57 PM
No idea haha
Vibe coding gone wrong
Hendrik spent like 4 hours trying to fix the vibe coded mess and it took me like 30 mins to just rewrite it without the LLM
We have to add back the connection to the simulator now but otherwise all good
Gonna do that later tonight
Isaac Neal, Jul 29, 6:08 PM
QuotedQuoted Sent by Isaac NealHendrik spent like 4 hours trying to fix the vibe coded mess and it took me like 30 mins to just rewrite it without the LLMEnd Quote press L to link back to original quoteEnd Quote press L to link back to original quoteI should say "us", we both rewrote it super fast
Antoine Jamme, Jul 29, 6:09 PM
Nice!!
Antoine Jamme, Jul 29, 6:14 PM
Can I help by preselecting some bags that are the best once you have pushed the code or you want to take care of that?
You, Jul 29, 6:19 PM
I extracted all the actions already for the files on the drive ut took a while with unzipping
no need to do that again
Antoine Jamme, Jul 29, 6:33 PM
Ok awesome we just need to go to Safeway and CVS and we have our demo
Antoine Jamme, Jul 29, 6:50 PM
Good job, I was starting to really believe we were gonna fake it by teleop lol
Issuing Helper, App, Jul 29, 8:32 PM
Welcome to Issuing Helper!
Issuing Helper simplifies your customers integration with DECTA. 
To get started, please contact your integration manager.
Isaac Neal, Jul 29, 8:33 PM
tf is this haha
You, Jul 29, 8:33 PM
quick shower and Im ready @Isaac Neal
not issue helper lol
Isaac Neal, Jul 29, 8:33 PM
Ready when you are
Let's try to install the colab thing so I can run on your PC
Isaac Neal, Jul 29, 8:37 PM
QuotedQuoted Sent by Isaac Neal    Logging this here so we can try this laterEnd Quote press L to link back to original quoteEnd Quote press L to link back to original quoteOkay so this doesn't work on cursor
But what we can do is just I screen share and write code and push it to a branch whenever we want to run it
