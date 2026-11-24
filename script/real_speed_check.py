#!/usr/bin/env python3
"""How fast did the real robot drive? Command, wheel feedback, tracker GT, every reported
estimator run (position-derived and its own velocity state), and a whole-segment IMU
integration that is shown NOT to work (bias over 34-113 s). Bags must be mounted.
    python3 script/real_speed_check.py
"""
import sys, sqlite3, glob, numpy as np, csv, os
sys.path.insert(0,"/opt/ros/humble/lib/python3.10/site-packages"); sys.path.insert(0,"script")
from rclpy.serialization import deserialize_message
from sensor_msgs.msg import Imu
from std_msgs.msg import Float64
import eval_real_gt as E
B="/media/ambushee/New Volume/wil_rescue"
def topic(ds,name,typ):
    c=sqlite3.connect(glob.glob(f"{B}/{ds}/*.db3")[0]); tid=dict((n,i) for i,n in c.execute("select id,name from topics"))[name]
    out=[(t/1e9,deserialize_message(d,typ)) for t,d in c.execute("select timestamp,data from messages where topic_id=? order by timestamp",(tid,))]
    c.close(); return out
def win_speed(t,p):
    tq=np.arange(t[0]+5,t[-1]-1,0.2)
    a=np.c_[[np.interp(tq,t,p[:,i]) for i in range(p.shape[1])]].T; b=np.c_[[np.interp(tq+1,t,p[:,i]) for i in range(p.shape[1])]].T
    v=np.linalg.norm(b-a,axis=1); return v[v>0.05]
for ds in ["dataset_realv7_000","dataset_realv7_001","dataset_realv7_002"]:
    print("=====",ds[8:])
    cmd=topic(ds,"/cmd_velocity",Float64); fb=topic(ds,"/feedback/velocity",Float64)
    fv=np.array([m.data for _,m in fb]); print("  /cmd_velocity values:", [round(m.data,3) for _,m in cmd], "| /feedback/velocity raw median while moving: %.3f (x0.325 = %.3f m/s)"%(np.median(fv[np.abs(fv)>0.05]),0.325*np.median(np.abs(fv[np.abs(fv)>0.05]))))
    g=E.load_gt(f"dataset/gt/{ds}"); print("  tracker GT               : median %.3f m/s"%np.median(win_speed(np.asarray(g["t_rel"]),np.asarray(g["xy"]))))
    for lbl,path in [("ORB GPU LC on","output/output_orb/real/%s/loopclosing_rate1"%ds),("ORB GPU LC off","output/output_orb/real/%s/noloopclosing_rate1"%ds),
                     ("ORB CPU LC on","output/output_orb/real/%s/loopclosing_rate1_cpu"%ds),("ORB CPU LC off","output/output_orb/real/%s/noloopclosing_rate1_cpu"%ds),
                     ("VINS-Fusion","output/output_vins/real/%s/rate1_002"%ds)]:
        a=np.loadtxt(path+"/vio.csv",delimiter=","); t=a[:,0]/1e9
        v3=win_speed(t,a[:,1:4]); 
        vel=np.linalg.norm(a[:,8:11],axis=1); vel=vel[(t>t[0]+5)&(vel>0.05)]
        print("  %-24s : position-derived median %.3f m/s | own velocity state median %.3f m/s"%(lbl,np.median(v3),np.median(vel)))
    # IMU: zero-velocity-bounded integration of horizontal acceleration
    imu=topic(ds,"/hwt101ct_yaw_publisher",Imu)
    t=np.array([x for x,_ in imu]); acc=np.array([[m.linear_acceleration.x,m.linear_acceleration.y,m.linear_acceleration.z] for _,m in imu]); gyr=np.array([[m.angular_velocity.x,m.angular_velocity.y,m.angular_velocity.z] for _,m in imu])
    # stationary = low gyro and low accel variation over 0.5 s
    w=100; n=len(t); still=np.zeros(n,bool)
    an=np.linalg.norm(acc,axis=1)
    for i in range(0,n-w,10):
        if np.std(an[i:i+w])<0.03 and np.max(np.linalg.norm(gyr[i:i+w],axis=1))<0.02: still[i:i+w]=True
    # gravity direction from all stationary samples; horizontal plane = perpendicular to it
    gdir=acc[still].mean(0); gdir/=np.linalg.norm(gdir)
    print("  IMU: gravity axis in sensor frame", np.round(gdir,3), "| stationary %.0f%%"%(100*still.mean()))
    # integrate per moving segment, removing the mean residual so each segment starts and ends at rest
    seg_speeds=[]
    i=0
    while i<n:
        if still[i]: i+=1; continue
        j=i
        while j<n and not still[j]: j+=1
        if j-i>200 and i>0 and j<n:     # >1 s of motion, bounded by rest at both ends
            g0=acc[max(0,i-100):i].mean(0); g1=acc[j:min(n,j+100)].mean(0)
            frac=np.linspace(0,1,j-i)[:,None]; grav=g0*(1-frac)+g1*frac
            ah=acc[i:j]-grav; ah-= (ah@gdir)[:,None]*gdir          # horizontal component
            dt=np.diff(t[i:j],prepend=t[i])
            v=np.cumsum(ah*dt[:,None],axis=0)
            v-= frac*v[-1]                                       # zero-velocity at both ends
            sp=np.linalg.norm(v,axis=1); seg_speeds.append((t[j-1]-t[i], np.median(sp[len(sp)//4:3*len(sp)//4])))
        i=j
    seg_speeds.sort(key=lambda x:-x[0])
    print("  IMU (ZUPT, longest motion segments): " + ", ".join("%.0f s -> %.2f m/s"%(d,s) for d,s in seg_speeds[:4]))
