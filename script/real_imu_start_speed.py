#!/usr/bin/env python3
"""IMU speed reached 1-4 s after each start from rest (bias and gravity from the rest window
just before, tilt tracked by the gyro). Reliable only for the first ~2 s.
    python3 script/real_imu_start_speed.py
"""
import sys, sqlite3, glob, numpy as np
sys.path.insert(0,"/opt/ros/humble/lib/python3.10/site-packages")
from rclpy.serialization import deserialize_message
from sensor_msgs.msg import Imu
B="/media/ambushee/New Volume/wil_rescue"
def skew(w): return np.array([[0,-w[2],w[1]],[w[2],0,-w[0]],[-w[1],w[0],0]])
def expm(w):
    th=np.linalg.norm(w)
    if th<1e-12: return np.eye(3)
    K=skew(w/th); return np.eye(3)+np.sin(th)*K+(1-np.cos(th))*K@K
for ds in ["dataset_realv7_000","dataset_realv7_001","dataset_realv7_002"]:
    c=sqlite3.connect(glob.glob(f"{B}/{ds}/*.db3")[0]); tid=dict((n,i) for i,n in c.execute("select id,name from topics"))["/hwt101ct_yaw_publisher"]
    rows=[(t/1e9,deserialize_message(d,Imu)) for t,d in c.execute("select timestamp,data from messages where topic_id=? order by timestamp",(tid,))]; c.close()
    t=np.array([r[0] for r in rows]); A=np.array([[m.linear_acceleration.x,m.linear_acceleration.y,m.linear_acceleration.z] for _,m in rows]); G=np.array([[m.angular_velocity.x,m.angular_velocity.y,m.angular_velocity.z] for _,m in rows])
    n=len(t); an=np.linalg.norm(A,axis=1); gn=np.linalg.norm(G,axis=1)
    still=np.zeros(n,bool)
    for i in range(0,n-50,5):
        if np.std(an[i:i+50])<0.02 and gn[i:i+50].max()<0.015: still[i:i+50]=True
    starts=[i for i in range(1,n) if still[i-1] and not still[i]]
    out=[]
    for s in starts:
        r0=max(0,s-100)
        if s-r0<60 or not still[r0:s].all(): continue
        g0=A[r0:s].mean(0); bg=G[r0:s].mean(0)            # gravity+accel bias, gyro bias, from the rest just before
        R=np.eye(3); v=np.zeros(3); sp=[]
        for k in range(s, min(n, s+800)):               # 4 s at 200 Hz
            dt=t[k]-t[k-1]
            R=R@expm((G[k]-bg)*dt)
            v+= (R@A[k]-g0)*dt                            # specific force in the rest frame minus rest reading
            if (k-s)%200==199: sp.append(np.linalg.norm(v))
        if len(sp)==4: out.append((t[s]-t[0],sp))
    print("%s: %d starts from rest"%(ds[8:],len(out)))
    for ts,sp in out[:6]:
        print("   start at %5.1f s -> speed after 1/2/3/4 s: %s m/s"%(ts," / ".join("%.2f"%x for x in sp)))
