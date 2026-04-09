# Claude Code Context

## Current Work: IEEE Paper Revision (paper-revision-occlusion-mitigation branch)

### What's Been Done
1. **paper.tex**: All professor requests addressed (27/27), ~75% of IEEE reviewer concerns addressed
2. **paper.tex**: "system" -> "pipeline" consistency fix (3 occurrences)  
3. **paper.tex**: Added paragraph reframing offline 5 FPS processing as a feature for generating imitation-learning action labels from pre-recorded RGB-D video
4. **scripts/compare_hand_detectors.py**: New script to compare MediaPipe vs WiLoR hand detection rate and valid IK target rate on a recorded .bag file

### What's Left (in order)
1. **Run the comparison script** on the Mac (where `vbhs` conda env is set up):
   ```bash
   conda activate vbhs
   python scripts/compare_hand_detectors.py
   ```
   The script processes `new experiment data (realsense bag file...)/human_demo.bag` and compares both detectors on detection rate and valid IK target rate.

2. **Add occlusion mitigation section to paper.tex** (Section VI.J) using the script results:
   - Root cause: MediaPipe's BlazePalm detector fails entirely during occlusion, so no landmarks at all
   - WiLoR's DarkNet localizer is more robust to partial occlusion
   - Report quantitative comparison: detection rate, valid IK target rate, recovery frames
   - Note trade-off: WiLoR requires GPU vs MediaPipe CPU-only

3. **Update Related Work** (Section II.A): Strengthen WiLoR citation to note it as the occlusion-robust alternative tested in Section VI.J

4. **Update Future Work** (Section VII): Revise occlusion bullet to note WiLoR integration is implemented and evaluated; temporal Kalman filtering and multi-camera remain future work

5. **Verify**: grep for "system" in paper.tex, ensure no remaining instances refer to the authors' pipeline

### Key Files
- `IEEE_publication/paper.tex` -- the paper being revised
- `scripts/compare_hand_detectors.py` -- WiLoR vs MediaPipe comparison script
- `IEEE request/email.md` -- IEEE reviewer comments (4 reviewers)
- `Professor request/REQUEST.MD` -- professor feedback
- `REVISION_PLAN.md` -- original revision plan (predates this session)
- `src/vbhs/pipeline/hands/wilor_hand_detector.py` -- WiLoR detector (already integrated)
- `src/vbhs/pipeline/hands/mediapipe_hand_detector.py` -- MediaPipe detector

### Bag File for Testing
`new experiment data (realsense bag file with associated computed actions data for pick and place/human_demo.bag` (1.4 GB)
