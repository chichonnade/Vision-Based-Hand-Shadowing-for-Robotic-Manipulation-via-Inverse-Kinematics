
IEEE Access <onbehalfof@manuscriptcentral.com>
Attachments
Sat, Apr 4, 10:37 AM (4 days ago)
to me, tre_mart, anandakumar.psgtech

04-Apr-2026

Dear Mr. CHICHE:

I am writing to you regarding manuscript # Access-2026-11582 entitled "Vision-Based Hand Shadowing for Robotic Manipulation via Inverse Kinematics" which you submitted to IEEE Access.

Your article was peer reviewed with interest but has not been recommended for publication in its current form.  We strongly encourage you to address the reviewers’ concerns, which can be found at the bottom of this letter, and resubmit your article to IEEE Access once you have updated it accordingly.
 
Please note that IEEE Access has a binary peer review process. Therefore, to uphold quality to IEEE standards, an article is rejected even if it requires minor edits.
 
When updating your manuscript, you should elaborate on your points and clarify with references, examples, data, etc. If you disagree with any technical points the reviewers have made, please include your counterarguments in your response to the reviewers (more information detailed below) and work this into the updated manuscript. 

Also, note that if a reviewer suggested references, you should only add those that are relevant to your work if you feel they strengthen your article. Recommending references to specific publications is not appropriate for reviewers and you should report excessive cases to ieeeaccessEIC@ieee.org.  Authors are not obligated to cite articles that are recommended by the reviewers, and the final decision on the article will not be influenced by whether or not authors cite these suggested references.
 
IEEE Access allows one opportunity to resubmit. If the updated manuscript is determined not to have addressed all of the previous reviewers’ concerns, or if the Associate Editor still has substantial technical concerns, the article will be rejected and no further resubmissions will be allowed.
 
When you are ready to resubmit your updated article, you can do so in the IEEE Author Portal.  When you log into the IEEE Author Portal you will see the title of the rejected article and the option to “Start Resubmission”.

https://ieee.atyponrex.com/submission/submissionBoard/REX-PROD-2-4A2B5170-9B15-4DCE-B75C-C53FC6FBEED3-38A442E1-FB86-4D1E-A724-C01E45CB00AC-10649/current?idtype=external
 
Upon resubmission you will be asked to upload the following 3 files:

1) A document containing your response to reviewers from the previous peer review.  The “response to reviewers” document (template attached) should have the following regarding each comment: a) Reviewer’s concern, b) your response to the concern, c) your action to remedy the concern. The document should be uploaded with your manuscript files under "Author's Response Files.”

2) Your updated manuscript with all your individual changes highlighted, including grammatical changes (e.g. preferably with the yellow highlight tool within the pdf file). This file should be uploaded with your manuscript files as “Highlighted PDF.”

3) A clean copy of the final manuscript (without highlighted changes) submitted as a Word or LaTeX file, and as a PDF, both submitted as the “Main Manuscript.”

**IMPORTANT: Please see the attached Resubmission Checklist that details all the items listed above.  Please utilize this checklist to ensure you have made the necessary edits to your manuscript, and to ensure you have all the necessary files prepared prior to resubmission.

*** AUTHOR LIST CHANGES: If your revised manuscript has an updated author list, you will need to submit a formal request to the Editor by completing the attachment labelled ‘Request for Byline Change,’ and uploading it as 'Request for byline change form.' This should include a DETAILED justification explaining each author’s contribution(s) to the work. You will also need to provide the justification for the author change during the submission process.  Change in the author list is considered rare and exceptional, and the decision to allow such changes rests with the Editor. Once the list and order of authors has been established, the list and order of authors should not be altered without permission of all living authors of that article.

We sincerely hope you will update your manuscript and resubmit soon. Please contact me if you have any questions.

Thank you for your interest in IEEE Access.

Sincerely,

Dr. Anandakumar Haldorai
Associate Editor, IEEE Access
anandakumar.psgtech@gmail.com

Reviewers' Comments to Author:

Reviewer: 1

Comments:
1.The manuscript presents a vision-based hand shadowing pipeline leveraging inverse kinematics; however, the novelty is not sufficiently distinguished from prior teleoperation and imitation learning frameworks. While the integration of RGB-D sensing and IK is well-executed, the authors fail to clearly articulate how their approach significantly advances beyond existing hybrid pipelines. The contributions appear incremental rather than fundamentally innovative. A stronger positioning against state-of-the-art methods is required.

2. The proposed system operates at approximately 5 FPS due to a total latency of 213 ms per frame, as shown in the results section (page 9). This severely limits its applicability in real-time robotic manipulation scenarios, which typically require at least 20–30 FPS. The reliance on offline processing undermines the practical usability of the system. The authors must address optimization strategies or justify this limitation more rigorously.

3. The system demonstrates a drastic performance drop from 90% success in structured settings to only 9.3% in real-world environments (page 10). This indicates poor robustness and generalization capability, which is critical for real-world deployment. The manuscript does not sufficiently explore mitigation strategies for occlusion and environmental variability. This significantly weakens the practical impact of the proposed method.

4. Hand occlusion is identified as the primary failure mode, yet the proposed solution lacks any robust occlusion-handling mechanism beyond simple fallback heuristics. Given that occlusion is a well-known challenge in vision-based systems, the absence of advanced techniques such as multi-view fusion or temporal tracking is a major limitation. The discussion acknowledges the issue but does not provide a concrete solution. This reduces the technical depth of the work.

5.The evaluation is restricted to a single pick-and-place task using a specific robot (SO-ARM101) and a constrained tile-based setup (pages 7–8). This narrow experimental scope limits the generalizability of the findings. The manuscript lacks testing across different tasks, objects, and robotic platforms. Broader experimentation is necessary to validate the robustness and scalability of the approach.

6. Although the paper compares the IK pipeline with four VLA models, the comparison is not entirely fair or comprehensive. The training configurations, dataset sizes, and optimization levels differ significantly, potentially biasing the results. Additionally, statistical significance and variance across trials are not reported. A more rigorous and standardized benchmarking protocol is required.

7. The system heavily depends on MediaPipe for hand landmark detection, which is known to struggle under occlusion and clutter. Furthermore, multiple heuristic-based fallback strategies are introduced for depth estimation and gripper control. These heuristics, while practical, lack theoretical justification and may not generalize well. This reliance reduces the robustness and scientific rigor of the approach.

8. The manuscript lacks a formal analysis of error propagation across the pipeline stages, particularly from 2D detection to 3D reconstruction and IK solving. Additionally, quantitative metrics such as positional error, orientation error, and trajectory smoothness are not reported. The evaluation focuses primarily on success rate, which is insufficient for a comprehensive technical assessment. Inclusion of detailed error metrics would strengthen the paper significantly.

Additional Questions:
Please confirm that you have reviewed all relevant files, including supplementary files and any author response files, which can be found in the "View Author's Response" link above (author responses will only appear for resubmissions): Yes, all files have been reviewed

1) Does the paper contribute to the body of knowledge?: yes

2) Is the paper technically sound?: yes

3) Is the subject matter presented in a comprehensive manner?: yes

4) Are the references provided applicable and sufficient?: yes

5) Are there references that are not appropriate for the topic being discussed?: No

5a) If yes, then please indicate which references should be removed.:


Reviewer: 2

Comments:
1. The core process of this paper is highly dependent on the simple stack of existing mature tools (such as extracting 2D key points using MediaPipe and solving IK using PyBullet's ready-made interface), lacking substantive theoretical breakthroughs in the underlying algorithms of 3D hand pose estimation or robot inverse kinematics.
2. As a teleoperation and motion capture oriented system, the single frame processing time of the algorithm reaches 213ms (about 5 FPS), and it completely relies on Offline processing, which cannot meet the basic requirements of low latency for robot control. The authors are advised to optimize the algorithm architecture and complement the experimental evaluation with online real-time control (at least 30 FPS).
3. In unstructured environments such as supermarkets, the success rate of the system plummeted to 9.3% due to hand occlusion; The paper only reports this phenomenon, but fails to propose any occlusion compensation or temporal deep fusion mechanism to solve this pain point. It is suggested to supplement the algorithm improvement and corresponding ablation experiments for severely occluded scenes.
4. The IK method of pure geometric analysis is compared with the end-to-end visual-language-action model (VLA, such as ACT, SmolVLA) on the extremely simple monotone task (only grasping the purple square), which is difficult to fully reflect the advantages and disadvantages of the system. It is recommended to introduce more diverse benchmark tasks containing complex physical contact and obstacle avoidance to supplement the direct comparison experiments with other advanced Marker-free teleoperation frameworks.
5. The control of the grip is only based on the simple geometric calculation of the Angle between the 3D coordinates of the thumb and the index finger. This hard-coded strategy cannot adapt to the grasping requirements of complex shapes and unknown objects. It is suggested to introduce a 6-DoF grasp pose estimation module, and to add quantitative test experiments for grasping objects of different shapes, sizes and materials.

Additional Questions:
Please confirm that you have reviewed all relevant files, including supplementary files and any author response files, which can be found in the "View Author's Response" link above (author responses will only appear for resubmissions): Yes, all files have been reviewed

1) Does the paper contribute to the body of knowledge?: yes

2) Is the paper technically sound?: yes

3) Is the subject matter presented in a comprehensive manner?: yes

4) Are the references provided applicable and sufficient?: yes

5) Are there references that are not appropriate for the topic being discussed?: No

5a) If yes, then please indicate which references should be removed.:


Reviewer: 3

Comments:
Major Points
1. The proposed hand-shadowing pipeline using RGB-D and IK is effective, but its distinction from existing teleoperation and imitation learning frameworks is not clearly emphasized.
2. The system operates at ~5 FPS offline, which contradicts practical teleoperation requirements; real-time feasibility and optimization strategies should be discussed.
3. The drastic drop in success rate (90% to 9.3%) highlights a major limitation, and no robust solution for occlusion handling is proposed.
4. Apart from success rate, other metrics such as positional error, trajectory smoothness, latency impact, and IK accuracy are not evaluated.
5. Comparison with state-of-the-art is incomplete. Although VLA models are compared, traditional teleoperation or advanced IK-based baselines are not included.
6. Evaluation is limited to a simple pick-and-place task and two environments; more complex manipulation tasks should be tested.
7. Key parameters (EMA smoothing, IK damping, thresholds) are fixed without systematic sensitivity or ablation study.
8. While PyBullet preview is used, discrepancies between simulation and physical execution are not quantitatively analyzed.
9. The multi-level fallback mechanism lacks theoretical justification and may affect consistency under varying conditions.
10. Limitations of the SO-ARM101 robot like workspace, precision, actuator limits are not deeply analyzed.
11. The method’s applicability to multi-arm systems, different robots, or higher DOF manipulators is not explored.



Additional Questions:
Please confirm that you have reviewed all relevant files, including supplementary files and any author response files, which can be found in the "View Author's Response" link above (author responses will only appear for resubmissions): Yes, all files have been reviewed

1) Does the paper contribute to the body of knowledge?: Partial

2) Is the paper technically sound?: Partial

3) Is the subject matter presented in a comprehensive manner?: Partial

4) Are the references provided applicable and sufficient?: Yes

5) Are there references that are not appropriate for the topic being discussed?: No

5a) If yes, then please indicate which references should be removed.:


Reviewer: 4

Comments:
Review Report:
Vision-Based Hand Shadowing for Robotic Manipulation via Inverse Kinematics
The paper presents a vision-based hand-shadowing pipeline for robotic manipulation using inverse kinematics, achieving a 90% success rate in structured environments.
1. It uses single egocentric RGB-D camera for hand tracking and retargeting and explain it.
2. It achieves 90% success rate in structured pick-and-place benchmark, elaborate it.
3. Compares with vision-language-action policies (ACT, SmolVLA, π0.5, GR00T N1.5).
4. Highlight specific research gaps in existing robotic teleoperation approaches.
5. Compare novelty of IK retargeting pipeline with other hand-tracking methods.
6. Provide more details on damped-least-squares IK solver implementation.
7. Discuss gripper controller fallback hierarchy.
8. Include more unstructured environment tests to validate robustness.
9. Compare results with additional baseline approaches.
10. Discuss deployment challenges in real-world scenarios.
11. Suggest future research directions (e.g., dynamic hand tracking).
12.  Abstract: Rephrase “complexity of mapping human hand articulations” for clarity.
13. Introduction: Cite recent literature on robotic teleoperation.
14. Address methodological clarity and add comparative analysis.


Additional Questions:
Please confirm that you have reviewed all relevant files, including supplementary files and any author response files, which can be found in the "View Author's Response" link above (author responses will only appear for resubmissions): Yes, all files have been reviewed

1) Does the paper contribute to the body of knowledge?: needs revision

2) Is the paper technically sound?: needs revision

3) Is the subject matter presented in a comprehensive manner?: needs revision

4) Are the references provided applicable and sufficient?: Yes

5) Are there references that are not appropriate for the topic being discussed?: No

5a) If yes, then please indicate which references should be removed.:

If you have any questions, please contact article administrator: Mr. Dharmendra Sharma dharmendra.sharma@ieee.org