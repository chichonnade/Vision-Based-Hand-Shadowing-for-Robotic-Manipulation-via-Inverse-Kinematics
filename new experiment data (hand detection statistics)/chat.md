Hand swapping detection
Isaac Neal, Jul 17, 12:27 PM
The single hand case works pretty well. We detect pretty much every erroneous swap to left hand after the first few seconds of the right hand only clip.

It's basically impossible to avoid misclassifying some as left at the start since we literally don't know
One solution for now would be to wait 20 frames or so to figure out which hand we think is in the scene before we start applying actions
QuotedQuoted Sent by Isaac NealIt's basically impossible to avoid misclassifying some as left at the start since we literally don't knowEnd Quote press L to link back to original quoteEnd Quote press L to link back to original quoteAs in, if for the first 5 frames the detector thinks it's left, then the correct thing to do is apply the actions on the left hand
Isaac Neal, Jul 17, 12:33 PM, Edited
How it works is:
We track an exponential moving average of confidence for each hand. This is updated every time we see detect a given hand. 

I'll proceed below assuming the right is the actual hand in the demos, but same holds for left.

In the case that the right hand appears almost all the time, then the confidence for the right hand will be very close to 1, and the confidence for left very close to 0. 

When we encounter a left hand, we first ask: 
1. Is the confidence that we should see a right hand substantially higher than that for seeing left?
2. If so, is this newly detected "left" hand close to the last known position of the right hand?
3. If so, we label it as an error, and correct the left label to right
You, Jul 17, 12:36 PM
nice that is pretty sick
does this have to do with a likelihood or maximum likelihood ?

"1. Is the confidence that we should see a right hand substantially higher than that for seeing left?"
Isaac Neal, Jul 17, 12:40 PM
It isn't a statistical measure it's just a heuristic
So it is in [0, 1] but it is not a probability
We tune it with an alpha parameter that determines how quickly we change our confidence when a new hand shows up
Isaac Neal, Jul 17, 3:12 PM
I'm adding a minimum confidence to apply an action. Tweaking it to try to remove any false left hands at all, then I'll see if this still works when we have both hands in frame
Results with min_confidence 0.5

It's improving
Will try 0.75 and see if the results still look good on screen
You, Jul 17, 3:19 PM
I think for the demo we only want to have one arm turned on, I'm too afraid of the left arm wanting to quickly reach the falsely predicted left hand
It's already gonna be tricky to not get kicked out, so me may just as well keep things to the bare minimum idk, what do you think ?
But I think this safety feature is a great idea !
Isaac Neal, Jul 17, 3:21 PM
Yeah makes sense
I'll just put a conservative minimum and move on then
Making it be very confident (confidence of 0.75) before acting makes the robot go crazy :/
I guess we just filter out too many valid hand positions
Isaac Neal, Jul 17, 3:29 PM
Okay opening a PR with this stuff now before I move on
Isaac Neal, Jul 17, 3:33 PM
Gimme a minute before merging I want to test it
Isaac Neal, Jul 17, 3:38 PM
Okay I think separately the IK is bugged for left hands (as @Antoine Jamme pointed out might be the case), will fix that next but at least my change didn't cause it
You, Jul 17, 3:45 PM
the initial videos after you fixed the IK looked very good, I don't know how much we need to use filters since we'll preview the videos in simulation first before actually sending them over to the bot for now.
Isaac Neal, Jul 17, 3:46 PM
So far all my testing has been with the right hand
You, Jul 17, 3:46 PM
It's def nice to have though but "as long" as the demos looks good I think we should just be fine
Isaac Neal, Jul 17, 3:46 PM
I think there's just a simple bug with the left hand
If I don't find it quickly we can just do everything with the right hand and mirror things without text in them haha
You, Jul 17, 3:47 PM
yeah that sounds like a good idea if we have one arm working for now it's all we need
Isaac Neal, Jul 17, 3:47 PM
Agreed
Isaac Neal, Jul 17, 4:24 PM
Nvm so it does work when there's only a left hand - it seems the issue is more when there are both hands it can get really messed up. Something to work on later I guess.
Antoine Jamme, Jul 17, 4:32 PM
Yeah it's suprising the left arm has always been kinda the problem considering the fact that it's joints 1 to 7 (or 0 to 6) meaning it's the first arm that gets called when doing a loop and in the urdf.
Antoine Jamme, Jul 17, 4:34 PM

Isaac Neal, Jul 17, 4:36 PM
There is probably a bug somewhere between IK and performing the actions if I had to guess
I'll ask cursor in the background while I do something else
QuotedQuoted Sent by Isaac NealThere is probably a bug somewhere between IK and performing the actions if I had to guessEnd Quote press L to link back to original quoteEnd Quote press L to link back to original quoteBecause IK seems to solve correctly if there is a left hand and no right hand
Antoine Jamme, Jul 17, 5:00 PM
QuotedQuoted Sent by Isaac NealBecause IK seems to solve correctly if there is a left hand and no right handEnd Quote press L to link back to original quoteEnd Quote press L to link back to original quoteOh ok that's cool
Isaac Neal, Jul 17, 5:01 PM
Yeah more likely it's just something simple that's wrong 🙏
