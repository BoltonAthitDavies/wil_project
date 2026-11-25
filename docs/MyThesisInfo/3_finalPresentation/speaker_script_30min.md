# WiL final presentation — 30-minute speaker script

Deck: `WiL final presentation compressed.pdf` (version of 25 Nov 2026), main slides 1–53; slide 54 opens the appendix, which is for questions only (the hardware table is now appendix slide 76).
Pace: the spoken text is about 3,000 words, which is about 110 words a minute once you pause to point at the figures. Times in brackets are per slide; the **clock** line is where you should be at the end of each part.
Numbers match the final report as rebuilt on 25 Nov 2026.

| Part | Slides | Time | Clock |
|---|---|---|---|
| Problem and objectives | 1–8 | 3:40 | 3:40 |
| Requirements and scope | 9–10 | 1:45 | 5:25 |
| Literature review | 11–15 | 2:45 | 8:10 |
| Methodology | 16–20 | 2:25 | 10:35 |
| Experimental setup | 21–23 | 1:20 | 11:55 |
| Experiment 1 | 24–39 | 8:00 | 19:55 |
| Experiment 2 | 40–49 | 5:50 | 25:45 |
| Summary, limitations, plan | 50–53 | 1:50 | 27:35 |

Spare: about two and a half minutes, for slide changes and for slowing down on slides 28, 37, 47 and 48, which carry the findings. If you run late, shorten slides 5–7, 25, 30 and 42 first: they are context, not findings.

---

## Part 1 — Problem and objectives (3:40)

**Slide 1 — Title** [0:15]
Good morning. My project is a vision-based navigation pipeline for autonomous mobile robots in dynamic warehouses. I am Bolton Athit Davies, and my advisor is Dr. Rattanachai Ramaithitima.

**Slide 2 — Gensurv's problem** [0:45]
The project starts from Gensurv's warehouses. The robots they use today, latent mobile robots, tow trucks and QR-guided AGVs, all depend on the building: floor markers, QR codes or reflectors. Changing the layout means changing that infrastructure, which is expensive. 2-D LiDAR navigation removes the markers, but it struggles to localise when the scene keeps moving and when the environment changes. So the question is whether cameras can replace that infrastructure.

**Slide 3 — LiDAR versus stereo camera** [0:35]
This table is the trade-off. LiDAR measures depth directly and its scale is exact, but a good unit is expensive. A stereo camera gets depth indirectly from disparity, so it needs more computation, and it fails where there is no texture or no light. Its advantage is cost: tens to hundreds of dollars per camera instead of thousands.

**Slide 4 — What is visual SLAM** [0:15]
Visual SLAM is the process of estimating where the camera is while building a map of where it has been, at the same time.

**Slide 5 — Front end** [0:25]
It has three parts. The front end estimates motion from frame to frame. Feature-based methods, which both of my systems use, pick out a sparse set of corners and match them between frames. Direct methods work on the raw pixel intensities instead.

**Slide 6 — Back end** [0:25]
The back end takes those rough motion estimates and makes them globally consistent. A filter does this recursively from state to state; an optimisation-based back end builds a graph of poses and constraints and solves it. Both of my estimators are optimisation-based.

**Slide 7 — Mapping** [0:20]
The third part is the map. A sparse map is enough to localise against; a dense map, such as an occupancy grid, is what a navigation stack needs to plan a path.

**Slide 8 — Objectives** [0:40]
So the project has four objectives. First, build a localisation and mapping pipeline that uses only cameras and an IMU, with no floor markers. Second, make that localisation robust in a warehouse with moving traffic. Third, produce a 2-D cost map a navigation stack can use. And fourth, relocalise the robot in a prior map, so that the map can correct the drift.

**Clock: 3:40**

---

## Part 2 — Requirements and scope (1:45)

**Slide 9 — Requirements** [1:00]
These objectives become four requirements. The pipeline is two stereo-inertial front ends, ORB-SLAM3 and VINS-Fusion, with RTAB-Map as the back end that builds a prior map and relocalises in it, and with outputs in the form Nav2 consumes.

Then three evaluations. Localisation accuracy against ground truth, in static and moving-traffic warehouses, and against estimators that use no camera at all. The prior map, in two ways: its pose graph and its occupancy grid. And relocalisation in that map.

The table on the right is the targets, fixed before the results: 5 centimetres absolute trajectory error, 10 centimetres relative error per second, 90 percent map precision, 85 percent completeness, at most 1 percent false obstacles, at most 10 percent of the time without a fix, and no pose jump over one metre.

**Slide 10 — Scope** [0:45]
The scope. In simulation, a realistic Gazebo warehouse in two layouts, with four localisation recordings, two long mapping recordings, and six relocalisation runs. On the real robot, an Ackermann platform at Gensurv with fisheye stereo cameras and a 200 Hz IMU, and four recordings with tracker ground truth. Everything runs on one laptop; the robot's computer only records. Recorded bags are replayed at real-time rate, and the relocalisation runs are live in simulation, driven by hand.

**Clock: 5:25**

---

## Part 3 — Literature review (2:45)

**Slide 11 — VI-SLAM methods** [0:35]
Why visual-inertial? A camera alone loses scale and fails in fast motion; an IMU alone drifts within seconds. Combining them covers both weaknesses. This review lists the main visual-inertial systems of the last decade. I shortlisted the two that support stereo and fisheye cameras, are actively maintained, and run in ROS: ORB-SLAM3 and VINS-Fusion.

**Slide 12 — Systems challenges** [0:35]
This study measured exactly that question on real hardware. On the left, latency: on a laptop at 20 frames per second, ORB-SLAM3 takes 46 milliseconds per frame, and on a Jetson about 93. On the right, the trajectory error is a few centimetres on both easy and hard sequences. The lesson for this project is that accuracy is not the bottleneck on embedded hardware; computation is. That is why compute cost is part of my first experiment.

**Slide 13 — ORB-SLAM3** [0:35]
ORB-SLAM3 runs three threads. Tracking estimates the pose of every frame from ORB features and the preintegrated IMU. Local mapping refines the recent keyframes and initialises the IMU. Loop closing recognises places it has seen before and corrects the whole trajectory. Keep that loop closer in mind: it decides most of the results.

**Slide 14 — VINS-Fusion** [0:30]
VINS-Fusion is a tightly coupled, optimisation-based estimator. It tracks corners with optical flow and solves a sliding window of recent keyframes against the IMU; old states are marginalised out. In our version there is no loop closure, so it is pure odometry, and its error can only accumulate.

**Slide 15 — RTAB-Map** [0:30]
RTAB-Map is a graph-based SLAM library. The key property for this project is that it takes odometry from outside. So either front end can feed it, and RTAB-Map adds what they lack: a saved map, place recognition against that map, and an occupancy grid for Nav2.

**Clock: 8:10**

---

## Part 4 — Methodology (2:25)

**Slide 16 — System architecture** [0:40]
This is the whole system. The stereo pair and IMU feed the front end at 30 and 200 hertz. The front end's odometry goes to RTAB-Map, which localises on the prior map built once beforehand. Its corrections pass through a correction gate and a pose gate before they reach Nav2. Planning and control on the right are standard Nav2 with a Hybrid-A* planner and pure pursuit; planning is outside my scope. My work is the orange block on the left.

**Slide 17 — ORB-SLAM3 plus RTAB-Map** [0:30]
Inside RTAB-Map there are four steps. A new node is created from the front-end pose and the stereo pair. Bag-of-words proposes which stored place it might be. Geometric verification checks that proposal. And graph optimisation turns an accepted match into a correction. Then the map is assembled and the correction gated.

**Slide 18 — Bag-of-words** [0:20]
Why bag-of-words? Checking every node's geometry would be far too slow. So each image is reduced to a set of visual words, and the node with the most similar words becomes the one candidate.

**Slide 19 — Geometric verification** [0:25]
But words cannot tell two identical aisles apart. Geometry can: if it is really the same place, the matched points must line up in 3-D under one rigid motion. If too few points agree, the candidate is rejected.

**Slide 20 — Correction gate** [0:30]
And the gate is my addition. In localisation mode RTAB-Map's output is a correction to the robot's position. A wrong correction moves the robot to the wrong aisle instantly. The gate asks a physical question: could the robot really be that far from where its odometry says, given how far it has driven since the last fix? If not, the correction is refused.

**Clock: 10:35**

---

## Part 5 — Experimental setup (1:20)

**Slides 21–22 — U and I layouts** [0:35]
To test this I built two warehouse layouts from a forwarding-industry guideline. The U shape has receiving and dispatch on the same side, with storage around the back. The I shape is a straight flow, receiving at one end and dispatch at the other. Both are generated from the same script, with pallet racks, pallet jacks driving on jobs, and walking people.

**Slide 23 — The two experiments** [0:45]
That gives two experiments. Experiment 1 asks how well each estimator localises, in static scenes and with traffic, against non-visual baselines. Experiment 2 is the SLAM part: build a prior map, judge whether it reflects the warehouse, and judge whether the robot can relocalise in it well enough to hand Nav2 a usable pose.

Both run in two domains. Simulation has exact calibration and perfect ground truth. The real robot has fisheye cameras, tape-measured extrinsics, and an optical tracker whose marker offset is not yet measured; keep that in mind for the real-robot numbers. The hardware table is in the appendix.

**Clock: 11:55**

---

## Part 6 — Experiment 1 (8:00)

**Slide 24 — Process** [0:30]
Experiment 1 has three steps. Collect the datasets. Measure the sensor noise against ground truth, so every estimator gets honest noise values. Then reconstruct the route five ways: wheels only, IMU only, wheels and IMU fused in a Kalman filter, VINS-Fusion, and ORB-SLAM3. The main metric is aligned ATE, the RMS error after the best rigid fit.

**Slide 25 — Simulation datasets** [0:10]
Four recordings, 100 to 170 metres each: U static, I static, and the same two layouts with traffic.

**Slide 26 — Non-visual estimators** [0:25]
The three baselines. Wheel odometry integrates wheel speeds and steering. IMU dead reckoning integrates the gyro and accelerometer. The EKF predicts with the IMU and corrects the speed with the wheels.

**Slide 27 — Noise** [0:20]
Our simulated IMU is twice as noisy as the EuRoC benchmark in the gyroscope and nine times in the accelerometer. So this is a harder IMU than the one most published results are tuned on.

**Slide 28 — ATE** [0:50]
Here is the main result. Each panel is one recording. The IMU alone drifts 12 to 23 metres; wheels 1 to 5 metres; the filter 0.7 to 2.8. The visual estimators sit far below: VINS-Fusion 0.22 to 0.66 metres, and ORB-SLAM3 0.14 to 0.52 metres in 11 of its 16 runs.

The other five ORB-SLAM3 runs diverge, and you can see two of them here: ORB with loop closure on the I-static recording reaches almost 5 metres, and ORB without it on the I-traffic recording almost 2. I will come back to both.

**Slide 29 — RPE** [0:40]
Relative error over one second tells a different story. Every visual run is between 1.6 and 7 centimetres per second, inside the 10-centimetre target. Wheels pass too. The IMU alone and the filter on the visual systems' noise model do not. So locally, frame to frame, the cameras are good. What fails the 5-centimetre ATE target is accumulation over the route, and the occasional large event.

**Slide 30 — Real datasets** [0:10]
On the real robot, four recordings at Gensurv, 22 to 37 metres each.

**Slide 31 — Real ATE** [0:40]
ORB-SLAM3, in its four configurations, holds 0.19 to 0.48 metres, and on the three later recordings its configurations agree within about 2 centimetres. VINS-Fusion completes all of them at 0.37 to 0.79 metres. One caution: the robot was driven at 0.28 metres per second, not the design speed of 1.5, and the tracker's marker offset is not measured, so read these as the same accuracy class as simulation, not as a target test.

**Slide 32 — Real RPE** [0:15]
Per second, both stay under the 10-centimetre target: about 5 to 9 centimetres.

**Slide 33 — Simulation versus real** [0:35]
Is the real camera worse? No. The real images have 1.7 times the gradient and three times the corners per frame. What differs is motion. Per metre, the real robot drifts ten times more than simulation; per second, only twice. Because the real robot drives slowly, the IMU integrates far longer for every metre travelled. So we compare the domains per second, never per metre.

**Slide 34 — Front end paired** [0:30]
Does the GPU front end help? It keeps 97 to 99 percent of frames against 72 to 74 for the CPU, but the ATE does not follow. On I-static both loop-closing runs are metres off, CPU and GPU alike. What decides the error is the loop closer and the recording, not the front end.

**Slide 35 — Dynamic scene** [0:35]
Now traffic. Here a pallet jack drives just ahead of the camera, from 24 to 44 seconds, shown on the left. On the right, the error each run makes per second. Inside the shaded window all five runs rise two- to eightfold, then fall back. Over the four seconds that is less than 10 centimetres in total, and no run loses tracking. So traffic raises the local error, but at this density it does not drive the global error.

**Slide 36 — VINS-Fusion feature tracking** [0:30]
VINS-Fusion's weak spot is different. On the U-traffic recording, when the camera faces a blank wall, its tracked features fall below the 120 line, and its position error climbs to half a metre. Two runs of the same recording behave the same way, so this is the scene, not chance.

**Slide 37 — The false closure** [0:35]
And this is the biggest failure in the experiment. On the I-static recording, both loop-closing ORB-SLAM3 runs accept one loop closure at 73.2 seconds, and in that single step the error goes to 14 metres. The two aisles look alike, the closure is self-consistent, and nothing asks whether a 14-metre jump is plausible after half a metre of drift. That question is exactly what my correction gate asks in Experiment 2.

**Slide 38 — Computation** [0:25]
On cost, the two pipelines sit in opposite corners. ORB-SLAM3 needs about four times the memory but under half the processor time; VINS-Fusion about eight cores' worth. The two ORB-SLAM3 front ends sit together, so the feature extractor is not what separates the pipelines.

**Slide 39 — Summary of Experiment 1** [0:50]
Three messages. One: the visual estimators are about ten times better than any non-visual one, but none reaches the 5-centimetre target. Why not? On the I layout it is height: the robot drives on a flat floor, yet the estimate drifts up and down. In the floor plane alone, one ORB-SLAM3 run reaches 4.4 centimetres and three more about 6. On the U layout the error is in the plane itself, 10 to 33 centimetres.

Two: the big failures come from the loop closer on look-alike aisles, not from traffic.

Three: on the real robot both pipelines work, after two fixes to VINS-Fusion: masking the fisheye rim beyond 70 degrees, and holding the stereo extrinsics fixed.

**Clock: 19:55**

---

## Part 7 — Experiment 2 (5:50)

**Slide 40 — Objective** [0:25]
Experiment 2 asks two things. Does the prior map reflect the real warehouse, well enough for a planner? And does the robot relocalise in it accurately and continuously enough to hand Nav2 a usable pose? Metrics: precision, recall and false obstacles for the map; tracking error, availability and continuity for the pose.

**Slide 41 — Which front end, and why** [0:30]
For this experiment the front end is ORB-SLAM3. The reason is the robot's computer. On the laptop, VINS-Fusion used about eight cores' worth of processor time and ORB-SLAM3 about four. The robot's Carmel CPU has six cores, so ORB-SLAM3 is the one that fits; and streaming two cameras to a laptop over the network was not an option on the real robot.

**Slide 42 — Map datasets** [0:15]
Three mapping recordings: the tiny warehouse, and one long drive of about 375 metres through each realistic layout.

**Slide 43 — Map result** [0:40]
The tiny map passes every target: 96.6 percent precision, 89.5 percent completeness, 0.7 percent false obstacles. The realistic maps fail: precision 78 and 82 percent, and false obstacles up to 10 percent. But their pose graphs are accurate, within tens of centimetres over the whole route. So the failure is in building the grid, not in the poses. Most of the obstacles that block the route are laid in the first seconds, during the inertial start-up.

**Slide 44 — Relocalisation runs** [0:15]
Then six relocalisation runs, three route shapes per layout, straight, square and figure-eight, from 21 to 148 metres.

**Slide 45 — Tracking result** [0:35]
The localised pose holds a median of 13 to 21 centimetres in every run. Five of six meet the availability target; only the I straight is blind 12.5 percent of the time. Four have no pose jump. One run, the I figure-eight, went over a metre for 6.5 seconds, and recovered.

**Slide 46 — Planar error** [0:50]
Here is the error against distance. On the U layout it stays between about 10 and 25 centimetres for 150 metres. On the I layout it grows over the first 20 to 40 metres and then drops back; the straight run never drops back. How does this compare with the map? Along the route, the map's own nodes are off by 9 to 14 centimetres, and the localised pose is 1 to 1.7 times that. So the map bounds the error. But inside a run, places where the map is worse do not show larger error, so whether a better map would lower it is still to be tested.

**Slide 47 — Place recognition** [0:40]
Now the safety side. Over six runs, place recognition made 346 proposals, and 251 were accepted. On the right are the 95 rejected ones. The horizontal axis is how many visual words matched; the vertical axis is how many features agreed in 3-D. Some proposals share 150 to 190 words and have zero geometric agreement. That is the look-alike aisle: it looks the same and it is not the same place. Words alone would have accepted them.

**Slide 48 — The correction gate** [0:55]
And the gate catches what geometry lets through. It accepted 63 of 68 corrections and refused 5, all on the U square. On the left you can see they are separate: the accepted corrections are small and inside the box, and the refused ones are steps of four to six metres. Replaying the run without the gate, those five would have put the robot 2.7 to 4.7 metres off. That figure comes from a replay, because the ungated runs were lost. With the gate, all six runs hold 13 to 21 centimetres. Panel b compares the prior with ORB-SLAM3 alone; treat it as a trend only, because the two are on slightly different time bases.

**Slide 49 — Summary of Experiment 2** [0:45]
Three messages. One: the poses are right, the grid is not. The tiny map passes, the realistic maps fail on false obstacles, though their poses are accurate. Two: two filters stop a look-alike aisle from moving the robot. Geometry rejected 95 proposals, and the gate refused the 5 corrections that would have moved it metres. Three: the relocalised pose holds 13 to 21 centimetres, bounded by the map.

**Clock: 25:45**

---

## Part 8 — Summary, limitations, plan (1:50)

**Slide 50 — Requirements** [0:45]
Against the requirements. ATE: not met. The best run is 0.14 metres, about three times the target. RPE: met, in both domains. Map precision, completeness and false obstacles: met on the tiny map, not on the realistic maps. Availability: met in 5 of 6 runs. Continuity: met in 4 of 6. The real-robot map cells are empty because no prior map was built from real imagery.

**Slide 51 — Limitations** [0:35]
Three limitations. Building a map with a non-holonomic robot and two forward cameras is hard: it cannot turn on the spot to look around. The real site changed between recording days, so recordings are not repeats. And the laptop heats up over consecutive runs, which changes latency and frame rate. That is why I report compute per run and flag the thermal confound.

**Slide 52 — Plan** [0:20]
On the plan, the study, design, integration and data collection are done. What remains is the integrated live test at the design speed of 1.5 metres per second in both domains, and completing the report.

**Slide 53 — Thank you** [0:10]
Thank you. I am happy to take your questions.

**Clock: 27:35**

---

## Likely questions (one-line answers)

- **Why not just use ORB-SLAM3's own map?** It has no occupancy grid for Nav2 and no way to gate a wrong relocalisation; RTAB-Map takes external odometry and provides both (slides 15, 17).
- **Does traffic affect localisation?** It raises the per-second error two- to eightfold while a vehicle fills the view, under 10 cm in total; it did not drive the global error at the simulated density. Not "no effect" (slide 35).
- **Why does 3-D ATE miss 5 cm if RPE passes?** Accumulation and height: on the I layout one ORB run is 4.4 cm in the floor plane; on the U layout the planar error itself is 10–33 cm. Post-hoc split, report table 4.6.
- **Is the gate too strict, could it refuse a correct fix?** Yes, that is its cost: while corrections are refused the robot runs on odometry. The one 6.5 s excursion was such a stretch. The trade is short coasting for no multi-metre jump.
- **Why are the real ATEs not a target test?** 0.28 m/s instead of 1.5, and the tracker marker offset (0.2–0.4 m along track) is unmeasured.
- **Is a 0.10 m RPE target fair when the proposal said 1.5 m?** It was tightened after the results; the report records that as a deviation.
