# สคริปต์พูดประกอบสไลด์ WiL update 16 พ.ย. 2026

ความยาวรวมประมาณ 26 นาที เลขหน้าตรงกับ `update_161126.pdf` (43 หน้า)
หลักการพูดทุกหน้า: บอกข้อสรุปหนึ่งประโยค แล้วชี้ไปที่ตัวเลขหรือจุดบนรูปที่พิสูจน์มัน
ส่วน A (หน้า 3–11) คือ methodology ตามเปเปอร์สี่ฉบับ วาดเป็น flow ให้มีกล่องที่ผลการทดลองในส่วน B จะชี้กลับมา
กล่องขอบแดงในแผนภาพคือกล่องที่มีผลการทดลองชี้มา เลขมุมกล่องคือเลขหน้าของสไลด์นั้น
หน้า 33 กับ 41 เป็นหน้าสรุปหนึ่งหน้าของ Experiment 1 และ 4 ถ้าเวลาน้อยให้ข้ามรายละเอียดแล้วพูดสองหน้านี้แทน

---

**หน้า 1 – ชื่อเรื่อง**
รอบนี้สองส่วนครับ ส่วนแรกเป็น methodology: อ่านเปเปอร์ต้นทางของ ORB-SLAM3, VINS-Mono, online temporal calibration และ RTAB-Map แล้ววาด flow ของแต่ละระบบให้เห็นว่ากล่องไหนทำอะไร ส่วนที่สองเอาผล Experiment 1 กับ 4 มาวางทับ flow นั้น ทุก finding จะชี้กลับไปที่กล่องที่มันเกิด

**หน้า 2 – สารบัญ**
(อ่านหัวข้อ ไม่ต้องขยาย) ย้ำว่าเลขหน้าในวงเล็บของส่วน B ชี้กลับไปส่วน A

---

## ส่วน A: Methodology

**หน้า 3 – โปรเจกต์วางบน flow อย่างไร**
stereo 30 Hz + IMU 200 Hz (sim pinhole, real fisheye KB8) เข้า front end สองตัวที่ publish ของเหมือนกันคือ body pose บน TF: ORB-SLAM3 stereo-inertial (สลับ extractor CPU/GPU และ loop closer เปิด/ปิด) หรือ VINS-Fusion stereo-inertial (VIO อย่างเดียว) แล้ว RTAB-Map รับ pose จาก TF ไม่รู้ว่าใครส่งมา สร้าง prior DB ใน mapping แล้ว localize อ่านอย่างเดียวพร้อม hold 5 วินาที และ gate 1.5 เมตร ต่อให้ Nav2
ป้ายแดง: Experiment 1 ทดสอบกล่อง front end ทั้ง sim และ real, Exp. 3 ทดสอบ map, Exp. 4 ทดสอบ localise + navigate
ย้ำสองเรื่อง: Atlas, loop closer, IMU init ทั้งหมดอยู่ **ใน** กล่อง ORB-SLAM3 Experiment 1 สลับแค่ closer กับ extractor และเราไม่ใช้ visual odometry ของ RTAB-Map เอง ประเด็นของเปเปอร์ที่ว่า back end ไม่สน odometry คือสิ่งที่ทำให้เทียบสอง front end ได้ยุติธรรม

---

## ส่วน B: Experiment 1 (simulation)

**หน้า 4 – ทุกระบบคือ front end + back end + map เชื่อมด้วย loop closure**
flow ทั่วไปของ visual SLAM ครับ กล้องกับ IMU เข้า front end สกัดหรือติดตาม feature ประมาณการเคลื่อนที่ แล้ว back end optimise ใน window หรือ local BA ส่วน loop closure คือ place recognition บวก geometric check แล้ว global optimisation ส่งผลกลับเข้า map
คำสำคัญที่จะใช้ทั้ง deck มาจากเปเปอร์ ORB-SLAM3: data association สามแบบ short-term คือจับคู่กับไม่กี่วินาทีที่ผ่านมา (VO ล้วน drift ไม่หยุด) mid-term คือจับคู่กับ map point ใกล้ ๆ ที่ยัง drift น้อย (ทำให้ drift เป็นศูนย์ในที่ที่ map แล้ว) long-term คือจำสถานที่ได้ไม่ว่าจะ drift เท่าไร
ORB-SLAM3 ใช้ทั้งสาม VINS-Mono ใช้ short-term + long-term ไม่มี mid-term RTAB-Map คือ long-term block อย่างเดียวบน odometry อะไรก็ได้ บวก occupancy grid ให้ planner

**หน้า 5 – ORB-SLAM3: สามเธรดบน Atlas เดียว**
แถวบนคือ tracking thread ที่ frame rate: สกัด ORB + stereo match, preintegrate IMU, ทำนาย pose จาก frame ก่อนหรือ relocalise, track local map, ตัดสินว่าจะเป็น keyframe ไหม
แถวกลาง local mapping ที่ keyframe rate: ใส่ keyframe, cull map point, สร้าง point ใหม่, local BA บน window ของ keyframe, และ IMU initialisation กับ refinement อยู่ในเธรดนี้
แถวล่าง loop & map merging: query DBoW2 ทั้ง Atlas, คำนวณ Sim3/SE3, guided matching, ตรวจด้วย covisible keyframe 3 ตัว, welding BA, essential graph, full BA ในเธรดแยก
Atlas คือ map ที่ active หนึ่งอันกับ map ที่ไม่ active หลายอัน ถ้าหลุด track เกิน 5 วินาทีจะเปิด map ใหม่ แล้ว merge กลับตอนเจอที่เดิม
กล่องขอบแดงสามกล่องคือกล่องที่ Experiment 1 จะชี้มา: ORB extraction (หน้า 19), IMU initialisation (หน้า 17–18), place recognition (หน้า 16)

**หน้า 6 – IMU initialisation เป็น MAP สามขั้น แล้วหมุน map ไปตาม gravity**
ขั้นแรก visual-only ประมาณ 2 วินาที ใส่ keyframe 4 Hz จนได้ 10 keyframe ขั้นสอง inertial-only MAP แก้หา scale, ทิศ gravity, bias และความเร็ว โดยมี prior ดึง bias เข้าศูนย์ gravity ถูก parameterise ด้วยสองมุมรอบแกนแนวนอน stereo-inertial ล็อก scale = 1 ขั้นสาม visual-inertial MAP แล้ว **หมุน map ให้ z ตาม gravity** ทำ BA ซ้ำที่ 5 กับ 15 วินาที
ฝั่งขวาคือ robustness: หลุด track สั้น ๆ ใช้ IMU ทำนาย pose แล้วหา match ใน window ใหญ่ ได้ 5 วินาที ถ้าหลุดยาวเปิด map ใหม่ และ map ที่หลุดภายใน 15 วินาทีหลัง IMU init จะถูกทิ้ง
สามจุดที่จะกลับมา: การหมุน map คือ frame change ในค่า pose ที่ publish (หน้า 18), "slow motion ทำให้ observability แย่" คือหุ่นบนพื้นราบ (หน้า 17), และ heuristic ทิ้ง map ถูก tune มาสำหรับ sensor ถือมือหรือโดรน (หน้า 24)

**หน้า 7 – Place recognition แลก temporal check ของ DBoW2 กับ geometric check**
หกขั้น: DBoW2 ให้ candidate สามตัว, สร้าง local window รอบ candidate, หา Sim3/SE3 ด้วย RANSAC+Horn, guided matching สองรอบ, ตรวจด้วย covisible keyframe สามตัวใน active map แทนการรอ bag of words ยิงสามครั้งติด, และ gravity check เฉพาะ pitch/roll
ฝั่งขวาคือสิ่งที่เปเปอร์ยอมแลก: DBoW2 ดิบได้ precision/recall 50–80 % temporal check ดันไป precision 100 % แต่ recall 30–40 % และถ่วงทุก closure สาม keyframe ORB-SLAM3 ตัด delay นั้นทิ้ง แล้วพึ่ง geometric agreement แทน
ปัญหาคือ guard ที่สร้างจาก "point agree กัน" แยกวัตถุที่ **เหมือนกันเป๊ะ** ไม่ได้ และ gravity check คุมแค่ pitch/roll ถ้า match ผิดไปหนึ่ง aisle บนพื้นราบ มันผ่านทั้งสองอย่าง นี่คือหน้า 16

**หน้า 8 – VINS-Mono: preprocessing, initialisation, sliding-window VIO, relocalisation, pose graph 4-DoF**
กลุ่มซ้าย: KLT ติดตาม feature + Shi-Tomasi เติมให้ครบ 100–300 จุด undistort แล้วฉายขึ้น unit sphere ตัด outlier ด้วย RANSAC fundamental matrix เลือก keyframe จาก parallax (ชดเชยหมุนด้วย gyro) ด้านล่าง IMU preintegration กับ initialisation ที่เอา SfM มา align กับ IMU
กลุ่มกลาง tightly coupled VIO: sliding window ที่มี state IMU n ตัว, extrinsic กล้อง-IMU, feature เป็น inverse depth Ceres minimise prior + IMU residual + visual residual marginalise state เก่าเป็น prior และ time offset t_d เป็น state ด้วย
กลุ่มขวา: relocalisation ด้วย DBoW2 + BRIEF แล้ว pose graph 4-DoF เพราะ roll/pitch observe ได้จาก gravity drift จึงอยู่แค่ x, y, z, yaw
ประเด็นสำคัญ: fork ของเรา **ไม่รัน pose graph** VINS-Fusion ในโปรเจกต์นี้คือ VIO block อย่างเดียว คือ short-term data association ล้วน (หน้า 14)

**หน้า 9 – window ถูกจำกัดด้วย marginalisation และ visual residual อยู่บน unit sphere**
marginalisation ตาม Fig. 7: ถ้า frame รองสุดท้ายเป็น keyframe ให้ marginalise frame เก่าสุดออกเป็น prior ถ้าไม่ใช่ ทิ้ง visual measurement แต่เก็บ IMU preintegration ต่อไป เพื่อให้ keyframe กระจายใน space ราคาที่จ่ายคือ linearisation point ถูกล็อกเร็ว ซึ่งผู้เขียนยอมรับเพราะ "drift เล็กน้อยยอมรับได้สำหรับ VIO"
visual residual นิยามบน unit sphere ไม่ใช่ image plane เพื่อให้ fisheye และ omnidirectional ใช้สูตรเดียวกัน
สามผลที่ตามมา: error เป็น drift เรียบไม่มี jump (หน้า 14), texture dip จ่ายตรง ๆ (หน้า 20), และ stereo build ที่เรารันเก็บ unit-sphere residual ไว้หลัง compile flag แล้วหารด้วย z ใน tracker (หน้า 25)

**หน้า 10 – time offset กล้อง-IMU เป็น state: เลื่อน feature ไม่ใช่เลื่อน stream**
t_IMU = t_cam + t_d โดย t_d ไม่รู้ค่า มาจาก trigger delay, transmission delay, clock ไม่ sync ในช่วงไม่กี่ ms กล้องถือว่าเคลื่อนที่ความเร็วคงที่ feature จึงมีความเร็วคงที่บน image plane เอาความเร็วนั้นคูณ t_d ไปเลื่อน observation แล้วให้ t_d เป็นตัวแปรใน BA
ตัวเลขจากเปเปอร์: ถ้าไม่ calibrate RMSE ของ VINS-Mono บน MH03 เป็นพาราโบลาตาม t_d ทนได้แค่ ±6 ms พอ calibrate จะแบนตั้งแต่ −40 ถึง +40 ms converge ในไม่กี่วินาที RealSense ZR300 มี offset 12.74 ms และแปรตาม exposure time
ทำไมอยู่ใน deck: rig จริงของเราคือเคสเดียวกับเปเปอร์ (RealSense IMU + กล้อง fisheye แยก, host ประทับเวลา) sim มี clock แม่น ทุกผลใน sim จึงเป็น t_d = 0 โดยนิยาม บน rig จริง td handling เป็นหนึ่งในสอง defect ที่แก้เดือนนี้ แต่ยังไม่มีการวัดขนาดผลของมันแยกออกมา

**หน้า 11 – RTAB-Map คือ long-term block อย่างเดียว odometry เข้าทาง TF grid กับ correction ออก**
input คือ odometry บน TF จากแหล่งไหนก็ได้ + stereo/RGB-D sync แล้ว STM สร้าง node หนึ่งตัวต่อ DetectionRate: pose, visual words, และ local occupancy grid ที่สร้างครั้งเดียวใน robot frame
WM: loop closure detection ด้วย tf-idf + Bayes filter เทียบ LoopThr และ proximity detection สำหรับ node ใกล้ ๆ ใน graph ถ้าผ่านจะคำนวณ transform ด้วย PnP RANSAC ตัวเดียวกับ visual odometry โดยมี Vis/MinInliers เป็นธรณี แล้ว graph optimisation ด้วย GTSAM และ guard ที่สามคือ OptimizeMaxError ที่ปฏิเสธ closure ที่ทำให้ link ขยับเกิน variance
output คือ global map (2D grid, OctoMap, cloud) ที่ประกอบใหม่ทุกครั้งที่ graph ถูก optimise และ TF map→odom
localisation mode: graph อ่านอย่างเดียว output คือ correction map→odom ระหว่าง fix ที่รับ pose ที่ publish คือ dead reckoning ของ odometry
กล่องแดงสี่กล่องคือที่ Experiment 4 จะชี้มา: STM/map assembling (หน้า 39), detection → PnP (หน้า 37), TF map→odom (หน้า 38)

**หน้า 12 – การรับ fix หนึ่งครั้งต้องผ่านสองด่าน เพราะมันตอบคนละคำถาม**
รูปนี้คือแกนของทั้งเรื่องครับ ด่านที่ 1 ถามว่า "**ที่ไหน**ที่น่าจะใช่" ตอบด้วย bag of words + tf-idf + Bayes filter ด่านที่ 2 ถามว่า "**ใช่จริงไหม**" ตอบด้วย PnP + RANSAC เทียบ Vis/MinInliers
บรรทัดสีแดงใต้กล่องคือข้อจำกัดของแต่ละด่าน: ด่าน 1 ค้นทั้งแผนที่ได้ แต่แยกของที่หน้าตาเหมือนกันไม่ออก ด่าน 2 แยกของที่เหมือนกันได้ แต่ค้นเองไม่ได้ ต้องมีคนยื่น candidate มาให้
ไม่มีวิธีไหนทำได้ทั้งสองอย่าง สามหน้าถัดไปคือเหตุผลและราคาของแต่ละด่าน

---

**หน้า 13 – ด่าน 1 ทิ้ง geometry ทิ้งไป เพราะถ้าไม่ทิ้งก็ค้นไม่ไหว**
ตารางซ้ายคือราคาของการถามว่า "ที่ไหน" อย่างซื่อสัตย์: prior ของ U มี 148 node แต่ละ node มี 888 feature ถ้าจะตรวจเรขาคณิตกับทุก node ต้องเทียบ descriptor ราว 10⁸ ครั้ง **ต่อหนึ่งรอบการพิจารณา** ซึ่งทำไม่ไหว
ขวาคือสิ่งที่มันทำแทน สามอย่าง: (1) bag of words ลืมว่า feature อยู่ตรงไหนในภาพ จำแค่ว่ามีคำอะไรบ้าง แล้วใช้ inverted index ค้น ราคาจึงแทบไม่โตตามขนาดแผนที่ (2) tf-idf เพราะคำไม่ได้มีค่าเท่ากัน ขอบชั้นวางโผล่ทุกเฟรมไม่บอกอะไร คำที่หายากต่างหากที่บอกตำแหน่ง ซึ่งในโกดังที่โครงสร้างซ้ำกันหมดเรื่องนี้สำคัญที่สุด (3) Bayes filter เพราะเฟรมเดียวเชื่อไม่ได้ มันสะสมหลักฐานข้ามเฟรม ยอมให้ย้ายไปได้เฉพาะที่ใกล้เคียง และมี virtual place แปลว่า "ไม่ใช่ที่ไหนที่รู้จัก" คำตอบจึงเป็นการแจกแจงความน่าจะเป็น ไม่ใช่คำตอบเดียว

---

**หน้า 14 – ยุบสองด่านเป็นด่านเดียวไม่ได้ และข้อมูลของเราเองแสดงให้เห็นทั้งสองทาง**
ซ้าย — **ถ้าตัดด่าน 2 ออก**: จาก 346 ข้อเสนอ มี 64 ครั้งที่ inlier เป็นศูนย์ ทั้งที่มี word match มัธยฐาน 77 คำ ถ้าไม่มี geometry พวกนี้กลายเป็น fix หมด config เก่าที่ยังไม่มี gate คือภาพของสิ่งนั้น: สอง run เกินหนึ่งเมตรนาน 36 กับ 23.5 วินาที ไกลถึง 6.80 เมตร pose ตัดทะลุชั้นวางสามแถว
ขวา — **ถ้าดันให้ด่าน 1 ทำงานแทนด่าน 2**: sweep 11–12 พ.ย. ตั้ง Vis/MinInliers 35 บวก distribution test ผลคือ localise ได้ 0 จาก 81 pose กันของปลอมได้ แต่กันของจริงไปด้วย
สองด่านนี้พังคนละทาง นั่นคือเหตุผลที่ต้องมีทั้งคู่ สรุปหนึ่งบรรทัด: ด่าน 1 ทิ้งข้อมูลเพื่อให้ค้นหาได้ ด่าน 2 เอาข้อมูลกลับมาเพื่อตัดสิน

---

**หน้า 15 – มีสี่ด่านระหว่าง "เฟรมจากกล้อง" กับ "node ในกราฟ"**
ไล่จากซ้ายครับ (1) **อัตรา** DetectionRate 1 Hz พิจารณาแค่ 1 เฟรมต่อวินาที ที่เหลือถูกทิ้งตั้งแต่ต้นทาง mapping run ยาว 286 วินาที จึงมีผู้สมัครราว 286 เฟรม (2) **การเคลื่อนที่** LinearUpdate 0.1 m กับ AngularUpdate 0.1 rad (5.7°) จะเป็น node ก็ต่อเมื่อหุ่นขยับจริง จอดนิ่งไม่ทำให้แผนที่บวม ผ่านมา 148 node จาก ~286 คือ 52 % (3) **ความใหม่** RehearsalSimilarity 0.6 ถ้าเหมือนของที่อยู่ใน STM เกิน 60 % จะถูก merge แทนที่จะเก็บแยก — **ด่านนี้ไม่เคยทำงานเลย** ทั้ง 148 node มี weight = 0 (4) **งบ feature** ซึ่งมีสองก้อน
สองบรรทัดล่าง: node ห่างกันมัธยฐาน 2.67 เมตร (0.15–5.40) และเรื่องงบสองก้อนที่สำคัญ — จาก 888 feature ที่เก็บต่อ node มีแค่ ~491 ที่กลายเป็น "คำ" ในกระเป๋า (ชนเพดาน Kp/MaxFeatures 500 พอดี) อีก ~400 เก็บไว้ให้ด่าน 2 ใช้เท่านั้น คือ bag กับ geometry กินงบคนละก้อน

---

**หน้า 16 – config ทั้งหมด: เป็นค่า default ของไลบรารีหมด ยกเว้นสองตัว**
ตารางซ้ายคือค่าที่ node พิมพ์ออกมาใน log ของทุก run จริง ๆ ขวาคือสิ่งที่ต้องรู้สามข้อ
หนึ่ง **มีแค่สองค่าที่ต่างจาก RTAB-Map 0.23**: EstimationType 0 บังคับให้จุดที่ match กันต้องตรงกัน**เป็นเมตร** ไม่ใช่แค่ reproject ลงตำแหน่งเดียวกัน และ OptimizeMaxError 10 ที่ต้องดันขึ้นเพราะค่า default 3.0 ปฏิเสธทุก closure ที่ VIO ซึ่ง drift อยู่เสนอมา
สอง **มีสองอย่างที่ไม่ได้ทำงาน**: memory management ปิดอยู่ (TimeThr กับ MemoryThr เป็น 0) ไม่มี node ไหนเคยออกจาก working memory เลย — ยังไม่เป็นปัญหาที่ 148 node แต่จะเป็นเมื่อแผนที่โตขึ้น และการตรวจด้วย ICP (Reg/Strategy 2) ปิดอยู่ทั้งหกรัน ซึ่งเป็น guard ตัวเดียวที่ดูโครงสร้าง 3 มิติ ไม่ใช่แค่จุดที่ match กันได้
สาม **Experiment 4 ไม่ได้เพิ่มอะไรลงแผนที่เลย** รันแบบ read-only ทั้งหมด bag ทั้ง 148 ใบมาจาก mapping run หุ่นระหว่างวิ่งจึง "ถาม" ได้อย่างเดียว ไม่ได้ "เพิ่ม" — นี่คือเหตุผลที่รูปด่านของหน้า 37 นับเป็น proposals คือการยื่นคำถามไปที่คลังที่ถูกแช่แข็งไว้แล้ว

---

**หน้า 17 – Nav2 กินแค่ pose ที่ localise ได้ กับ grid ของ prior เท่านั้น**
บล็อกไดอะแกรมของชั้น navigation ครับ ซ้ายคือสิ่งที่ป้อนเข้ามา: front end ให้ TF odom→vio_body และ RTAB-Map ให้ map→odom พร้อม prior grid บน /map สองอันรวมกันเป็น TF chain map→odom→vio_body→base_footprint→base_link
กลางคือ costmap สองชั้น: global 15×22 เมตร ความละเอียด 5 ซม. ไม่เลื่อนตามหุ่น และ local 10×10 เมตรแบบ rolling ทั้งคู่มีสามเลเยอร์เหมือนกัน static + obstacle + inflation
ขวาคือ planner_server (Smac Hybrid, DUBIN, replan 1 Hz) กับ controller_server (Regulated Pure Pursuit, lookahead 1.3–2.0 เมตร) แล้วออกไปที่ velocity smoother และ /cmd_vel เข้า AckermannSteering ของ gz
กล่องสีชมพูด้านล่างคือ pose_gate.py ซึ่งเป็นของที่เราเขียนเอง มันตรวจสามข้อแล้วสรุปเป็น /localization_ok เดียว แล้วใช้สั่ง start/pause lifecycle ของ Nav2 — เพราะ Nav2 ตั้ง autostart false ไว้
ประเด็นที่ต้องย้ำ: **Nav2 ไม่เคยเห็น ground truth เลย** มันขับด้วย pose ที่ห้าหน้าก่อนหน้าผลิตออกมา บน grid จากฐานข้อมูลเดียวกัน ดังนั้น error ทุกตัวในหน้า 34–41 จะไหลมาถึงตรงนี้ในฐานะ input ของการวางแผน

---

**หน้า 18 – config ของ Nav2 ตามตัวรถ ยกเว้นตรงที่ SLAM เปลี่ยนสมมติฐาน**
ตารางซ้ายคือค่าที่ใช้จริง แบ่งเป็น chassis / planner / controller / costmap / lifecycle ส่วนครึ่งล่างซ้ายกับขวาคือสิ่งที่ต้องอธิบาย
ขวา — **สี่อย่างที่ต่างจาก config เดิมเพราะ pose มาจาก SLAM แล้ว**: (1) local costmap ต้องเป็น rolling เพราะ config เดิมใช้หน้าต่างคงที่ใน odom frame ได้ก็เพราะ map→odom เป็น identity เป๊ะ ๆ พอมาใช้ relocalisation ค่านั้นกระโดด หน้าต่างจึงต้องวิ่งตามหุ่น (2) autostart false เพราะ pose_gate เป็นคนตัดสินว่าจะเริ่มเมื่อไร covariance ต่ำไม่ได้แปลว่าถูก (3) allow_unknown false เพราะ prior ที่สร้างจาก stereo มีที่ที่ไม่เคยเห็นจริง ๆ ดังนั้นบาง goal จะถูกปฏิเสธ ซึ่งไม่ใช่ planner พัง (4) transform_tolerance 0.30 วินาที หลวมกว่าค่ามาตรฐาน เพราะ pose เดินทางผ่านสอง node คนละอัตรา
ซ้ายล่างสีแดงสองย่อหน้า — **การตรวจการชนอยู่ที่ planner ไม่ใช่ controller** ถ้าเปิดที่ lookahead นี้ sweep ก่อนหน้าเจอ "paralysis, 155 vetoes, 0.10 m/s" เพราะสิ่งที่มันไปห้ามคือการที่ controller ตัดโค้งของตัวเองชนวงแหวน inflation ผลที่ตามมาต้องพูดให้ชัด: controller จะไม่หยุดให้คนที่มันมองเห็น การหลบมาจากการ replan ที่ 1 Hz แทน
และข้อสุดท้าย — **ทั้งหมดนี้ยังไม่เคยถูกใช้งานจริง** ทั้งหกรันของ Experiment 4 เป็นการขับด้วยมือ ไม่เคยสั่ง goal เลยสักครั้ง

---

**หน้า 19 – สองตระกูล estimator บน 4 recording เดียวกัน หนึ่ง run ต่อ cell**
recording สี่ชุดในโลก realistic สอง layout (U, I) แต่ละ layout มี static กับ traffic เส้นทาง 103–171 เมตร
ตระกูล non-visual: wheel, IMU dead reckoning, EKF ที่ใช้ไฟล์ noise ของ visual system เอง และ EKF บน noise ที่วัดจริงเป็นแถว diagnostic ตระกูล visual: VINS-Fusion หนึ่ง run, ORB-SLAM3 สี่ run (CPU/GPU × closer เปิด/ปิด) replay rate 1.0 ไม่มี detector
กติกา: aligned ATE คือ RMSE หลัง rigid SE(3) fit ใน 3 มิติ RPE ที่ 1 วินาทีนับเฉพาะคู่ที่ไม่มี jump สถานะ closer อ่านจาก loop_closing.csv ไม่ใช่ชื่อ run traffic เป็นเงื่อนไขไม่ใช่ factor
ลิงก์ methodology: ทั้ง ORB-SLAM3 และ VINS ถ่วง preintegration residual ด้วย covariance ที่สร้างจากไฟล์ noise เดียวกันนี้ EKF ได้ตัวเลขชุดเดียวกัน ทั้งสามจึงรันบน inertial model เดียว

**หน้า 20 – VINS-Fusion ต่ำสุด 3 ใน 4 recording; EKF แพ้เพราะ noise model ไม่ใช่ architecture**
ตารางซ้าย: EKF ที่ noise ของ VSLAM ได้ 0.68–2.80 เมตร แพ้ VINS ทุก recording แต่ filter ตัวเดียวกันบน noise ที่วัดจริงได้ 0.21–0.58 ชนะ VINS 3 ใน 4 ดังนั้น deficit ทั้งหมดคือ noise model
กลไก: gyro density ใน config สูงกว่าที่วัด 5–7 เท่า process noise ของ heading พองไป 30–40 เท่า residual ของ wheel yaw-rate เลยไปบังคับ heading แทน bias yaw RMS ขึ้นจาก 0.4–5.7 องศาเป็น 4.5–28 องศา
ทุก covariance ของ proprioceptive overconfident: ANEES 8–3399 เทียบ 3.0
รูปขวาคือ trajectory ทั้งสี่ recording IMU ออกนอกอาคารทุกครั้ง EKF สีเขียวอ่อนเป๋ heading EKF สีเขียวเข้มอยู่บนเส้นทาง

**หน้า 21 – error ของ VINS-Fusion เป็น drift เรียบไม่มี jump ตามที่ sliding window ให้**
0.22–0.66 เมตร ศูนย์ jump ศูนย์ reset rotation 1.2–3.6 องศา RPE 0.023–0.070 เมตร worst case หลัง transient 0.46–1.14 เมตร repeat ขยับแค่ 5–7 %
ชี้ไปหน้า 7–8: ไม่มี map ให้ re-match จึงไม่มี correction แบบกระโดดเข้ามา prior จาก marginalisation พาอดีตมาเป็น constraint เรียบ ๆ นี่คือ short-term data association ล้วน "drift เล็กน้อยยอมรับได้" บนเส้นทาง 100–170 เมตร drift นั้นคือ 0.2–0.7 เมตร
pose graph ที่จะลบ drift คือ stage ที่ fork ของเราไม่รัน การไม่มีมันคือเหตุที่ VINS ไม่มี jump **และ** ไม่มี zero-drift revisit

**หน้า 22 – ORB-SLAM3 สองโหมด: 11 run อยู่ 5 run กระเด็น และ loop closer ตัดสินทั้งสองทาง**
ตารางเดิมจากรอบ 7 พ.ย. แต่คราวนี้คอลัมน์ขวาบอกว่า closer ทำอะไรในกล่องของหน้า 4
realistic_001 ปิด closer ได้ 0.5/0.4 เปิดแล้ว 6.02/4.78 เพราะ false closure 14 เมตรที่ 73.2 วินาทีทั้งสอง front end realistic_003 กลับกัน ปิดได้ 1.76/1.91 เปิดได้ 0.14/0.50 เพราะ closure ลบ vertical drift 5 เมตร U layout closer เกือบไม่ทำอะไร
clean RPE ของ run ที่กระเด็น (0.037–0.058) อยู่ในช่วงเดียวกับ run ที่อยู่ error คือเหตุการณ์ไม่กี่ครั้ง ไม่ใช่ tracking ไม่มี tracking loss ไม่มี reset ใน 20 run
สรุปบนกล่อง: 3 ใน 5 divergence เป็นของ loop & merging thread อีก 2 เป็นของ IMU initialisation (หน้า 17) และ repeat spread ของ ORB ถึง 0.277 เมตร ช่องว่างต่ำกว่าหนึ่งในสี่เมตรระหว่าง arm ของ ORB คือ noise

**หน้า 23 – false closure คือ place recognition รับวัตถุเหมือนกันสองชิ้นที่ห่างกันหนึ่ง aisle**
ทั้งสอง front end รับ closure เดียวกันที่ 73.2 วินาที correction 14.7 กับ 13.3 เมตร keyframe ที่ถูกเชื่อมห่างกัน 17.7 เมตรคนละฝั่งอาคาร ทั้งที่ drift สะสมแค่ 0.5 เมตร ไม่มี revisit จริงต้องการ 14 เมตร
รูปขวาบน: (a) keyframe ที่ถูกปิดฝั่งตะวันตก (b) keyframe ที่ match ฝั่งตะวันออก สิ่งที่มีร่วมกันคือกองกล่องเหลืองติดป้ายบน bin ซึ่งเป็น asset ตัวเดียวกันวางซ้ำ รูปล่างคือ timeline ที่ step ตอน 73 วินาที
ชี้ไปหน้า 6 ทีละขั้น: DBoW2 เห็นคำเดียวกัน (1) 3D alignment กับ guided matching agree เพราะวัตถุ**เหมือนกันจริง** (3–4) covisible keyframe สามตัวเห็นกองเดียวกัน (5) gravity check คุมแค่ pitch/roll (6) recall ที่ method ซื้อมาด้วยการตัด temporal check คือสิ่งที่โกดังที่มี asset ซ้ำจ่ายคืน

**หน้า 24 – vertical drift บน I layout คือทิศ gravity ที่ไม่เคย observable เพราะหุ่นไม่เลี้ยว**
realistic_003 run ที่ไม่มี closure ได้ 1.76/1.91 เมตร 3-D ATE ทั้งที่ planar error ไม่เกิน 0.7 เพราะมันลอยขึ้น 4.7 กับ 5.1 เมตร vertical error เทียบระยะทางเป็นเส้นตรง (r > 0.98) slope คือ tilt คงที่ 2.06/2.21 องศา สอดคล้องกับ 2.80/3.12 ที่วัดจาก orientation stream closure เดียวที่ run เปิด closer รับไปลบมันทิ้ง
ชี้ไปหน้า 5: R_wg อัปเดตด้วยสองมุมรอบแกนแนวนอน gravity กับ accelerometer bias จะแยกกันได้ก็ต่อเมื่อ body หมุน รูป (a): yaw ที่กวาดใน window ของการประมาณ gravity คือ 94 กับ 36 องศาบน U แต่ **0.00 องศา** บน I ทั้งสอง recording ซึ่งเลี้ยวครั้งแรกที่ 25 กับ 21 วินาที หลัง refinement สุดท้ายปิดไปแล้ว ส่วน remedy ที่เปเปอร์เสนอคือ scale-and-gravity refinement ซึ่ง build นี้ไปถึงเฉพาะ path monocular
ข้อจำกัด: excitation ปนกับ layout สอง route เทียบสอง route

**หน้า 25 – step 90 องศาที่ 2.4 วินาทีคือ initialisation ของ method ไม่ใช่ error**
local mapping บันทึก inertial init ที่ map size **10 keyframe พอดี** ทั้ง 16 run ที่ 2.38–2.51 วินาที ตรงกับขั้นแรกของหน้า 5 แล้ว orientation ที่ publish กระโดด 89.8–90.2 องศา y กับ z สลับกัน คือ map ถูกหมุนไปตาม gravity เป็นมุมติดตั้ง ไม่ใช่ residual
pose ก่อน step มีแค่ 2.0–3.6 % ของ run √(0.024 × 90²) = 13.9 องศา ตรงกับ 12.8–14.1 ที่คอลัมน์ unanchored รายงาน median จริงคือ 0.25–2.54 องศา BA ครั้งที่สอง (15–16 วินาที) ถอด bias prior แล้วตัด frame ใหม่อีกครั้ง median 0.535 องศา เทียบ 0.094 ครั้งแรก ใหญ่กว่าใน 16 จาก 16 run
วัดหลัง step: ORB 0.27–3.14 องศา เทียบ VINS 1.51–3.57 บนสาม recording ที่ไม่มี false closure กติกา: quote median ห้าม quote raw max

**หน้า 26 – front end เปลี่ยน throughput ไม่เปลี่ยน accuracy**
GPU extractor เก็บ frame 97–99 % CPU 72–94 % front-end median 30–37 ms เทียบ 49–55 ms แปดคู่ที่ถือ recording กับ closer คงที่แล้วสลับแค่ extractor: throughput ขึ้น 8 ใน 8 แต่ ATE ขึ้น 3 ลง 5 ภายใน 0.16 เมตรในห้าคู่ tracking time กับ frames kept สัมพันธ์ −0.73 แต่ tracked points กับ ATE แค่ −0.14 โซ่ขาดหนึ่งข้อก่อนถึง accuracy
ชี้ไปหน้า 4 กล่อง ORB extraction: ในเปเปอร์ extraction ใช้ 11–16 ms จาก budget 21–33 ms ที่ 752×480 20 Hz ของเรา CPU extractor อย่างเดียว 49–55 ms ที่ 30 Hz frame ที่หายไปเสีย mid-term match ไม่ใช่เหตุการณ์ discrete ที่กำหนด ATE
ข้อจำกัด: หนึ่ง run ต่อ cell เส้นเดี่ยวในรูปไม่ใช่หลักฐาน แค่ contrast ระหว่างสอง panel

**หน้า 27 – ใน run VINS จ่ายให้ texture dip ส่วน ORB ใช้ IMU สะพานข้าม**
VINS track ได้มากสุด 150 feature พอต่ำกว่า 120 planar error โตที่ 0.007–0.025 m/s เทียบ 0.0004–0.0025 ที่เพดาน realistic_002 aisle ตะวันออก 20–26 วินาที เหลือ 53 feature error +0.18 เมตร คือ excursion ทั้งหมดของ run นั้น run ที่สองทำซ้ำได้ภายใน 6 มม. กล้องหันเข้ากำแพงไม่มี texture stereo match rate ยังอยู่ (92 vs 91 %) tracker ไม่มีอะไรให้จับ ไม่ใช่ depth
ORB-SLAM3: error ขยับเฉพาะต่ำกว่า 250 tracked point (0.36 % ของ window) ที่กำแพงเดียวกันมันตกไปราว 110 point ไม่ถึงวินาทีโดย error ไม่ step
ชี้: KLT tracking (หน้า 7) เทียบกับ short-term-lost path ของหน้า 5 ที่ใช้ pose ทำนายจาก IMU ฉาย map ลง frame แล้ว search window จับกลับได้ในวินาทีถัดไป
ขอบเขต: วินาทีที่ feature น้อยคือ 8–20 % ของ run เป็นสาเหตุของ excursion ไม่ใช่ของ ATE

**หน้า 28 – traffic ทำร้ายทั้งสอง estimator ครั้งเดียว: pallet jack ที่วิ่งนำหน้าหุ่น**
วัตถุแค่อยู่ในมุมมองไม่มีต้นทุน วัตถุกวาดผ่านภาพเร็วทำให้ ORB เสีย 2.5 มม. ไม่ว่ามันขยับหรือไม่ เหตุการณ์จริงเดียวคือ realistic_003 31.5–35.6 วินาที jack อยู่ข้างหน้า 3.4 เมตร วิ่ง 1.3 m/s เทียบหุ่น 1.56 เต็มภาพ error ต่อวินาทีขึ้น 2.0–7.7 เท่าใน ORB ทุก run และ 3.0 เท่าใน VINS error สะสม step −0.03 ถึง +0.09 เมตร
ชี้ไป outlier rejection ของทั้งสอง front end: VINS ใช้ RANSAC fundamental matrix ORB ใช้ reprojection กับ map ผ่าน Huber ทั้งคู่สมมติว่าจุดผิดขยับ**ต่างจาก**ฉาก วัตถุที่วิ่งไปด้วยกันแทบไม่มี apparent motion ผ่านทั้งสองด่านแล้วถูกเขียนลง map jack ตัวเดียวกันที่ 6–20 เมตรเก้าวินาทีก่อนหน้าไม่มีผล ระยะสำคัญเท่าการเคลื่อนไหว
นี่คือ configuration เดียวที่ hypothesis ของโปรเจกต์ได้รับการสนับสนุน หนึ่ง episode บนหนึ่ง recording

**หน้า 29 – สอง estimator err คนละทิศตั้งฉากกัน ATE มองไม่เห็น**
สัดส่วน cross-track ของ planar error: VINS 0.67–0.73 ใน 3 จาก 4 recording ORB ทั้ง 11 run ที่อยู่ 0.05–0.28 บน realistic_000 สองตัวได้ 0.293 กับ 0.290 เมตร แยกไม่ออกแต่ตั้งฉากกัน
along-track ที่เกินของ ORB เป็น scatter ไม่ใช่ scale: path ยาวเกิน 1.7–5.4 % มาจาก spike ก่อน init ล้วน local scale 0.997–0.999 VINS มี scale จริงตัวเดียว 1.2 % บน realistic_002
ใน aisle 3 เมตร cross-track error ปิดช่องว่างกับชั้นวาง along-track ไม่
สถานะ: เป็น observation ยังไม่มีกลไก stereo ล็อก s = 1 ใน ORB (หน้า 5) VINS ประมาณ scale จาก IMU ทำไม**ทิศ**ต่างกันยังเปิดอยู่

**หน้า 30 – ต้นทุน: window ที่ถูกจำกัด เทียบ map ที่โต**
ตารางบน U mapping bag สาม repeat: ORB front end 37–42 ms back end 211–284 ms CPU 332–379 % RSS 1.7 GB VINS front end 38 ms back end **44 ms** CPU 819 % RSS **479 MB**
repeat accuracy: VINS spread 0.094/0.035 (5–7 %) ORB 0.089/0.277 CPU 0.016/0.236 GPU จาก binary เดียว config เดียว bag เดียว frames kept แกว่ง 70–98 % ไม่มี cost measure ไหนทำนาย divergence ได้ run ที่กระเด็นจ่ายเท่า run ที่อยู่
ชี้ไป back end ของหน้า 4 กับ 7: local BA ของ ORB ครอบ keyframe window **บวก** covisible keyframe ที่ fix ไว้ของ map ที่โตขึ้น (เปเปอร์ 121–217 ms) window ของ VINS มี n state ไม่ว่า map ใหญ่แค่ไหน solve แบน memory จำกัด แต่ทุก frame จ่าย dense solve
caveat: run ไม่ interleave แถว GPU บน I bag มี thermal confound ไม่ quote

---

## ส่วน B: Experiment 1 (real robot)

**หน้า 31 – หุ่นจริง: wheel+gyro กับ ORB-SLAM3 อยู่ class เดียวกัน หลังปิด heuristic ตัวหนึ่ง**
ตาราง: baseline wheel+gyro ที่ fit k = 0.325 บน realv7_000 แล้วถือไว้ ได้ 0.26/0.22/0.46 ORB-SLAM3 สี่ config ได้ 0.44–1.28 / 0.17–0.20 / 0.40–0.48 นำหนึ่ง เสมอหนึ่ง แพ้หนึ่ง VINS ไม่มี run ที่ใช้ได้ (หน้าถัดไป) ทุกตัวเลข provisional เพราะ lever arm ยังไม่วัด
ชี้ไปหน้า 5 robustness: upstream reset map ที่ keyframe ติดกันห่างน้อยกว่า 2 ซม. ใน 10 วินาทีแรกหลัง IMU init แล้ว re-init เฉพาะตอน queue ว่าง หุ่นที่ยืนนิ่งต้นทุก bag เข้าเงื่อนไขแรก keyframe ทุก 0.25 วินาทีทำให้เงื่อนไขสองไม่มีวันเกิด
patch 15 ปิดทั้งสองอย่าง IMU init ที่ 2.8–4.7 วินาทีทั้ง 16 run score ดิบ ย้ำ: มัน**ไม่ได้**ทำให้แม่นขึ้น แค่ทำให้ run มีอยู่

**หน้า 32 – VINS บน rig จริง: ทุก divergence คือ feature เกิน 77 องศาของเลนส์ที่ tracker ฉายเป็น x/z**
สิ่งที่ตัดทิ้งก่อน: g_norm เก่า, frame มา 40–45 %, reset 217 ครั้ง, IMU callback โดน JPEG decode แย่ง ทั้งหมดจริงแต่ไม่ใช่สาเหตุ เพราะ stereo-only ก็ diverge
โซ่หลักฐานบน realv7_001: พังตอนเริ่มขยับ 98 จาก 151 feature ถูก purge ใน frame เดียว offline KLT เดิมเก็บได้ 96–100 % parallax 6740 หน่วย normalised เป็นไปได้เฉพาะ |x/z| ~ 10 คือ field angle > 84 องศา เลนส์ KB8 ไปถึง ~100 องศา detector วางมุม 11–35 % เกิน 77 องศา และ 1–11 % ที่หรือเกิน 90 ที่ x/z → ∞ แล้วพลิกเครื่องหมาย
แก้: fov_mask_deg 70 ผ่าน setMask() ตัวเดียวกับ YOLO filter stereo-only ได้ 0.846 เมตรจากที่เคย 1e8 arm ที่มี IMU ยัง 1.04–1.31 และ under-scaled แปลว่า IMU coupling ยังมี defect
ชี้ไปหน้า 8: VINS-Mono นิยาม residual บน unit sphere **เพื่อ** fisheye โดยเฉพาะ โดรนในเปเปอร์ใช้เลนส์ 190 องศา แต่ stereo build ที่เรารันเก็บมันไว้หลัง UNIT_SPHERE_ERROR และ tracker map ray เป็น (x/z, y/z) ORB-SLAM3 KannalaBrandt8 ไม่เคยหารด้วย z จึง track rig เดียวกันได้ 0.17–0.63
สถานะ: provisional หนึ่ง recording หนึ่งค่า mask ยังไม่อยู่ใน write-up IMU rotation บน rig ก็วัดใหม่เป็น diag(−1,−1,1)

---

**หน้า 33 – สรุป Experiment 1 หนึ่งหน้า: noise model กับ loop closer เป็นตัวตัดสิน ไม่ใช่การเลือก estimator**
หน้านี้คือหน้าที่ใช้ได้ถ้ามีเวลาพูดเรื่อง Experiment 1 แค่หน้าเดียวครับ
ตารางซ้ายบนคือ sim: IMU 12–23 เมตร, wheel 1.0–4.8, EKF ที่ noise ของ VSLAM 0.68–2.80, EKF ที่ noise วัดจริง 0.21–0.58, VINS 0.22–0.66 ไม่มี jump, ORB 11 จาก 16 run อยู่ 0.14–0.52 อีก 5 run กระเด็น 1.39–6.02
ตารางซ้ายล่างคือหุ่นจริง: wheel+gyro 0.22–0.46, ORB ที่ดีที่สุดต่อ recording 0.17–0.44, VINS ไม่มี run ที่ใช้ได้ ทุกค่า provisional
เทียบเป้าหมายโปรเจกต์: ไม่มี estimator ไหนถึง 0.05 เมตร แต่ทุกตัวยกเว้น IMU อยู่ในขอบ RPE 1.5 เมตร
ขวาคือสี่ข้อสรุป แต่ละข้อผูกกับกล่องใน flow: (1) ไฟล์ noise สำคัญกว่าตัว estimator — filter ตัวเดิมขยับเป็น 0.21–0.58 แล้วแซง VINS 3 ใน 4 และ visual pipeline ทั้งสองอ่านไฟล์เดียวกันนี้ (2) error ของ ORB คือ loop closer ทั้งสองทาง ผูกกับหน้า 6 และหน้า 5 (3) front end เปลี่ยน throughput ไม่เปลี่ยน accuracy (4) บนหุ่นจริง vision ยังไม่ชนะ dead reckoning และ VINS ถูก root-cause แล้วว่าเป็นเรื่อง field angle
ปิดท้ายด้วยขอบเขต: หนึ่ง run ต่อ cell เทียบกับ spread ของ ORB เองที่ 0.277 เมตร, traffic ปนกับ route, lever arm ยังไม่วัด

---

## ส่วน B: Experiment 4

**หน้า 34 – Experiment 4: localise กับ prior ที่ไม่มี closure ขณะขับ มี gate บน correction**
prior คือ DB ที่ Experiment 3 score แล้ว: U 148 node, I 127 node graph แม่น 0.10–0.40 เมตร rms บนเส้น 365–381 เมตร localize_sim: graph อ่านอย่างเดียว Vis/EstimationType 0 (point ที่ match ต้องตรงกันเป็นเมตร) MinInliers 20 hold back end 5 วินาที และ correction_gate ปฏิเสธ correction ที่ใหญ่กว่า drift ของ front end เกิน 1.5 เมตร
ขับมือ 21–148 เมตร ตรง/สี่เหลี่ยม/เลขแปด ต่อ layout ไม่สั่ง goal ครึ่ง goal-reaching ยัง planned กติกาล็อกไว้ก่อนมี run
score: ระยะ planar ระหว่าง map→base_link ที่ 2 Hz กับ Gazebo truth **ไม่ใส่** lever arm (base_link ที่ publish อยู่ที่ model origin) แต่ fit map→world ใส่ เพราะ fit node ใน body frame
ชี้ไปหน้า 10: STM ยังสร้าง node แต่ graph ไม่โต loop closure detection กับ WM คือ localisation เอง hold กับ gate อยู่**หลัง** guard สองตัวของเปเปอร์ ระหว่าง RTAB-Map กับ TF Experiment 3 score prior พวกนี้เป็น geometry (precision 77.8–84.8 %, false obstacle 3.8–10.2 % ตกเกณฑ์ทั้งหมด) อันนี้ score เป็นของที่ localiser กับ planner กิน

**หน้า 35 – หก run track truth ที่ median 0.13–0.21 เมตร bounded 5 ใน 6 ไม่ชน**
ตาราง: median 0.128–0.210 p95 สูงสุด 1.15 max สูงสุด 1.24 (I เลขแปด) I ตรง จบที่ 0.763 และ blind 12.5 % เกินเกณฑ์ 10 % เป็น run เดียวที่ตกเกณฑ์ availability median ข้าม run 0.147 เมตร median ครึ่งแรก/ครึ่งหลังตรงกันยกเว้น I ตรง (0.11 → 0.53) clearance ต่ำสุด 0.68 เมตร
รูปขวา: planar error เทียบระยะทางตั้งแต่ fix แรก เส้นเทาคือ 0.10–0.40 ของ graph เอง 70 % ของ sample อยู่ระหว่างสองเส้น
ย้ำ: หนึ่ง run ต่อ cell ไม่มี repeat ไม่มีอะไรวัด run-to-run spread

**หน้า 36 – pose ดีเท่า map ที่มันไป localise: map-limited ไม่ใช่ estimator-limited**
pose 0.13–0.21 เทียบ graph ของ prior เดียวกัน 0.10–0.40 localiser ทำซ้ำ map ได้เท่าความแม่นของ map เอง error = offset คงที่ 0.04–0.34 + scatter 0.03–0.19 along-track mean สลับเครื่องหมายระหว่าง run จึงไม่ใช่ latency ORB-SLAM3 เดี่ยว ๆ ที่ rigid-align แล้ว drift median 0.013–0.307 localised pose ชนะสี่ route ที่ยาว 56.8 ขึ้นไป แพ้สองอันสั้นสุด จุดตัดอยู่ระหว่าง 30–57 เมตร
ชี้ไปหน้า 10: fix คือ PnP กับ feature ของ **node ที่เก็บไว้** ที่ pose ของ node นั้นใน graph estimate ไม่มีทางใกล้โลกกว่า node prior ให้ datum ใน frame ที่อยู่ข้าม run สิ่งที่ควรไล่คือ prior ที่ดีขึ้น ไม่ใช่ filter ที่ดีขึ้น
รูป: U เลขแปด 148 เมตร median 0.126 max 0.25 กำแพงขวาหนาใน grid คือ wall displacement ที่ Experiment 3 รายงาน

**หน้า 37 – vocabulary เสนอ geometry ตัดสิน: สองในสามของ fix ที่ถูกปฏิเสธมี inlier ศูนย์จากคำนับร้อย**
หก run: 346 proposal รับ 251 (73 %) ใน 95 ที่ปฏิเสธ 64 มี inlier **ศูนย์** ทั้งที่ word match median 77 สูงสุด 192 จำนวนคำแทบทำนาย geometric support ไม่ได้ (ρ = 0.24) ขยับ MinInliers เป็น 35 บวก distribution test หยุด warp ได้แต่หยุด localisation ด้วย: recording หนึ่ง localise ได้ 0 จาก 81 pose
ชี้ไปหน้า 10 และหน้า 6: Bayes filter บน tf-idf ตัดสินว่า**ใช่**ที่เดิมไหม PnP กับ MinInliers ตัดสินว่า**ที่ไหน** rack face ที่ซ้ำกันทำให้สอง aisle มีคำร่วมเป็นร้อย stage แรกจึง alias และ stage สองทำงาน discriminate ทั้งหมด อาคารเดียวกันนี้คือที่เอาชนะ place recognition ของ ORB-SLAM3 บน realistic_001 (หน้า 16) กลไกเดียว สองระบบ คราวนี้วัดจากในอัลกอริทึม
รูป: ซ้ายคือ proposal ที่ถูกปฏิเสธ แถวหนาด้านล่างคือพวก inlier ศูนย์ กระจายทั่วช่วงจำนวนคำ ขวาคือต่อ run แยกรับ/ปฏิเสธมี support/ปฏิเสธไม่มี support

**หน้า 38 – excursion เดียวคือ correction ที่ถูกปฏิเสธ pose ไหลบน odometry จน fix ผ่าน gate**
I เลขแปด เกินหนึ่งเมตร 6.5 วินาทีจาก t = 30 peak 1.24 จบ run ที่ 0.166 correction map→odom **ไม่ขยับ** ตลอดช่วง (yaw −3.40 องศา x −10.17 เมตร) ถ้าเป็น fix ผิดมันจะ step ตรงนี้ไม่มีอะไรถูกรับเลย heading error 4.93 ข้างในเทียบ 2.67 ข้างนอก คือ dead reckoning สะสม
config เดิมที่ไม่มี gate เคยพาสอง run เกินหนึ่งเมตร 36 กับ 23.5 วินาที ไกลถึง 6.80 เมตร แต่ route ต่างกัน จึงไม่ใช่การวัด gate แบบควบคุม
ชี้ไปหน้า 10 output map→odom: RTAB-Map publish correction ไม่ใช่ pose ระหว่าง fix pose คือ dead reckoning ของ front end การปฏิเสธ fix แปลง displacement ฉับพลันเป็นแบบช้า ๆ และ 1.24 เมตรคือราคาของการปฏิเสธหกวินาที หนึ่งเหตุการณ์ ไม่อ้าง taxonomy

**หน้า 39 – initialisation transient เดียวทำร้ายสองชั้น: phantom cell ใน map และ latch 73 องศาใน localiser**
ชั้น map (Experiment 3): 63–96 % ของ false obstacle ที่ block เส้นทางในสามจากสี่ map คือ cell ที่ graph node ห้าตัวแรกวาง
ชั้น localisation: orientation ของ front end ไม่นิ่งราวหนึ่งวินาทีหลังเริ่ม back end ที่รับ fix ใน window นั้น latch correction ผิด 73–76 องศาค้างไว้ราว 16 วินาที hold 5 วินาทีลบมันได้ correction yaw สูงสุดจาก 110–177 ลงเหลือ 25 องศาในสาม run wait_for_imu_init กับ init_ignore_s ช่วยไม่ได้ ตัวแรกจริงแล้วตั้งแต่ใน window ตัวสองทำงานหลัง correction เกิด
ชี้ไปหน้า 18 ป้อนหน้า 10: วินาทีแรกของ front end คือ frame ก่อน gravity ของ ORB init pose ถูกแต่แสดงใน camera optical frame จนกว่าจะหมุน 90 องศา STM ของ RTAB-Map สร้าง local grid **ครั้งเดียว** ใน robot frame แล้ววางด้วย graph pose global grid ประกอบใหม่เฉพาะตอน graph optimise pose ผิด frame หนึ่งวินาทีจึงวาง cell ที่ไม่มี correction ไหนแตะอีก ใน localisation วินาทีเดียวกันเสนอ node ที่ PnP ผ่าน MinInliers พร้อม rotation 73 องศา guard สองตัวของเปเปอร์ไม่มีตัวไหนดู orientation plausibility
transient เดียว สองอาการ ใบสั่งยาเดียว: อย่ากินวินาทีแรก รูปแบบที่ถ่ายไป rig ได้คือ orientation-stability condition ใน front end ไม่ใช่ 5 วินาทีตายตัว

**หน้า 40 – heading แยกตาม layout และผู้ต้องสงสัยปกติไม่ได้อธิบาย error**
heading median 0.21–0.49 องศาบน U เทียบ 1.31–2.83 บน I จับคู่ route แล้ว I แย่กว่า 3.6–11.5 เท่า เฉพาะ I ตรงที่เครื่องหมายเดียวตลอด frame residual คงที่จึงอธิบายไม่ครบ การเลี้ยวแพงกว่า 4–8 เท่าบน U แต่**ถูกกว่า**ขับตรงบน I ไม่ใช่สมบัติของการเคลื่อนไหว สาเหตุยังไม่รู้
error ไม่ขึ้นกับอายุของ fix ล่าสุด (median 0.8 วินาที) ไม่ขึ้นกับระยะถึง node ใกล้สุด (0.7–0.8 เมตร) และการเลี้ยวไม่ได้ทำให้ fix หาย ขึ้นกับ yaw rate ทั้งหก run แต่ yaw rate ปนกับ speed
timebase: time offset คงที่ fit pose ที่ publish ได้ที่ −0.04 ถึง −0.12 วินาที แต่ trajectory ของ front end เองที่ +0.20 ถึง +0.26 เครื่องหมายตรงข้ามบน run เดียวกัน แปลว่า publication path หนึ่งจัดการเวลาผิดราว 0.1 วินาที offset กับ frame รวมกันอธิบาย 18–77 % ของแต่ละ median ทุกตัวเลขในส่วนนี้เป็น upper bound

---

**หน้า 41 – สรุป Experiment 4 หนึ่งหน้า: pose ถูกจำกัดด้วย map และครึ่งการทดลองยังไม่มีข้อมูล**
หน้าสรุปของ Experiment 4 ครับ ตารางซ้ายบนบอกตรง ๆ ว่าคำถามไหนมีข้อมูลแล้ว: pose บน prior ขณะขับมีหก run ส่วน goal reaching, โลกที่มี traffic และหุ่นจริง ยังไม่มีเลยสักอัน สถานะรวมจึงเป็น partial
ซ้ายกลางคือสิ่งที่หก run ให้: median ข้าม run 0.147 เมตร, bounded 5 ใน 6, excursion เดียว 6.5 วินาที peak 1.24, ไม่ชนอะไร clearance ต่ำสุด 0.68 เมตร, heading 0.21–2.83 องศาที่แยกตาม layout โดยยังไม่รู้สาเหตุ
ขวาคือสามข้อสรุปผูกกับ flow ของ RTAB-Map หน้า 10: (1) map-limited ไม่ใช่ estimator-limited เพราะ fix คือ PnP กับ node ที่เก็บไว้ที่ pose ของ node นั้น estimate จึงไม่มีทางใกล้โลกกว่า node (2) vocabulary เสนอ geometry ตัดสิน 64 จาก 95 ที่ถูกปฏิเสธมี inlier ศูนย์ทั้งที่มีคำ match เป็นร้อย เป็นอาคารเดียวกับที่ทำให้ ORB เกิด false closure (3) gate เปลี่ยนรูปร่างของความล้มเหลว จากกระโดดเป็นไหลช้า และ transient เดียวทำร้ายทั้งสองชั้น
ซ้ายล่างคือขอบเขต: ไม่มี repeat เลยสักที่ การเทียบ world/layout/route จึงไม่ใช่หลักฐาน, route ปนกับรูปร่างและความยาว, และ timebase คลาด 0.1 วินาที ทำให้ทุกตัวเลขเป็น upper bound

---

**หน้า 42 – สิ่งที่เปิดอยู่ ตามลำดับที่ควรปิด**
Experiment 1: bound loop closer ด้วย max correction เทียบ inertial drift bound ทดสอบบน realistic_001 ที่ 14 เมตร reproduce ได้ทั้งสอง front end ให้ I layout เลี้ยวใน 15 วินาทีแรกหรือต่อ stereo-inertial path เข้า scale-and-gravity refinement fit inertial noise กับ platform เพราะ noise file มีค่ากว่า estimator บน rig: sweep fov_mask_deg รัน VINS ใหม่ทั้งสี่ bag พร้อม IMU rotation ที่แก้ วัด lever arm และ repeat cost run แบบ interleave
Experiment 4: repeat ก่อน เพราะ config นิ่งแล้วและทุก cell มี run เดียว เก็บ traffic world ทั้งสอง รัน goal set เพื่อตอบคำถามของ Experiment 3 ว่า false obstacle 3.8–10.2 % block route ไหม ใส่ timing reference ใน recorder และเริ่ม map หลัง transient ด้วย orientation-stability condition ใน front end

**หน้า 43 – สรุป**
Methodology: สามระบบคือ flow เดียวกันเติมกล่องต่างกัน ORB-SLAM3 มี data association ครบสามแบบบน Atlas VINS-Fusion แบบที่เรารันคือ sliding-window VIO อย่างเดียว RTAB-Map คือ long-term block บนอะไรก็ได้ บวก grid
Experiment 1 sim: VINS 0.22–0.66 ไม่มี jump ตามที่ window จำกัดให้ ORB อยู่ 0.14–0.52 ใน 11 จาก 16 run ส่วน 5 ที่กระเด็นเป็นสองกล่องใน flow ของมันเอง: place recognition รับวัตถุเหมือนกันคนละ aisle และทิศ gravity ที่หุ่นไม่เลี้ยวไม่เคยทำให้ observable extractor เปลี่ยน throughput ไม่เปลี่ยน accuracy EKF แพ้เพราะไฟล์ noise ที่ใช้ร่วมกัน
Experiment 1 real: wheel+gyro กับ ORB class เดียวกันหลังปิด low-motion reset VINS diverge เพราะ feature เกิน 77 องศาถูกฉายเป็น x/z จุดเดียวที่ stereo build ต่างจาก unit-sphere design ของเปเปอร์ provisional
Experiment 4: localisation บน prior ที่ไม่มี closure ได้ 0.13–0.21 เท่าความแม่นของ prior เอง map-limited bag of words alias ระหว่าง aisle geometry รับหน้าที่แยก gate เปลี่ยน fix ผิดเป็นการไหลช้า ๆ และ transient เดียวทำร้ายทั้ง map และ localiser
ยังไม่มี: filtered arm, map บนหุ่นจริง, goal reaching, repeat, traffic ใน Experiment 4
