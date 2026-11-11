#!/usr/bin/env python3
"""Generic read-only ROS 1 topic inspector (no robotics commands or proprietary ASTRA code)."""
import json, math, time, os, threading, subprocess
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

UPSTREAM='c2edc7414cc90a6f0421c16702ef44c5895fd151'
values={'pose':None,'velocity':None,'clock':None}
stamps={'pose':None,'velocity':None,'clock':None}
counts={'pose':0,'velocity':0,'clock':0}
seq=[0]
LOCK=threading.Lock()
errors=[]

def subscribe():
    try:
        import rospy
        from geometry_msgs.msg import Pose, Vector3
        from rosgraph_msgs.msg import Clock
        rospy.init_node('astra_readonly_probe',disable_signals=True)
        def pose(m):
            arr=[m.position.x,m.position.y,m.position.z,m.orientation.x,m.orientation.y,m.orientation.z,m.orientation.w]
            with LOCK:values['pose']=arr;stamps['pose']=time.monotonic();counts['pose']+=1;seq[0]+=1
        def velocity(m):
            with LOCK:values['velocity']=[m.x,m.y,m.z];stamps['velocity']=time.monotonic();counts['velocity']+=1
        def clock(m):
            with LOCK:values['clock']=m.clock.to_sec();stamps['clock']=time.monotonic();counts['clock']+=1
        rospy.Subscriber('/OrbitPosition',Pose,pose,queue_size=3)
        rospy.Subscriber('/OrbitVelocity',Vector3,velocity,queue_size=3)
        rospy.Subscriber('/clock',Clock,clock,queue_size=3)
        rospy.spin()
    except Exception as e:
        with LOCK:errors.append('ROS_LISTENER:'+type(e).__name__+':'+str(e)[:150])

def nodes():
    try:
        import rosnode
        return rosnode.get_node_names()
    except Exception as e:
        return ['ERROR:'+type(e).__name__]

def processes():
    out={}
    for s in ('roscore','gazebo','orbit'):
        try: pid=int(Path('/tmp/onorbit/'+s+'.pid').read_text());os.kill(pid,0);out[s]='PROCESS_EXISTS'
        except Exception:out[s]='NOT_RUNNING'
    return out

def snapshot():
    with LOCK:
        vv=values.copy();ss=stamps.copy();cc=counts.copy();sequence=seq[0];ee=list(errors)
    now=time.monotonic();issues=ee[:]
    for k in ('pose','velocity','clock'):
        if vv[k] is None:issues.append('MISSING_'+k.upper())
        elif ss[k] is None or now-ss[k]>2:issues.append('STALE_'+k.upper())
    if isinstance(vv['pose'],list):
        if len(vv['pose'])!=7 or not all(isinstance(x,(int,float)) and math.isfinite(x) for x in vv['pose']):issues.append('INVALID_POSE')
        elif not 0.98 <= sum(q*q for q in vv['pose'][3:]) <=1.02: issues.append('INVALID_QUATERNION')
    if isinstance(vv['velocity'],list) and (len(vv['velocity'])!=3 or not all(math.isfinite(x) for x in vv['velocity'])):issues.append('INVALID_VELOCITY')
    if vv['clock'] is not None and (not math.isfinite(vv['clock']) or vv['clock']<0):issues.append('INVALID_CLOCK')
    names=nodes()
    if not any('orbit' in x.lower() and 'astra' not in x.lower() for x in names):issues.append('UPSTREAM_NODE_NOT_FOUND')
    if not any('gazebo' in x.lower() for x in names):issues.append('GAZEBO_NODE_NOT_FOUND')
    msg={'schema':'astra-onorbitros-telemetry/1','source':'OnOrbitROS/Simulation','source_commit':UPSTREAM,
         'status':'OBSERVED_TELEMETRY_ONLY' if not issues else 'SAFE_HALT', 'issues':issues,
         'allows_robot_control':False,'units_verified':False,'sequence':sequence,
         'sim_time_s':vv['clock'] if vv['clock'] is not None else None,
         'position_eci':vv['pose'][:3] if vv['pose'] else None,
         'velocity_eci':vv['velocity'],
         'orientation_xyzw':vv['pose'][3:] if vv['pose'] else None}
    return {'status':'LIVE_ROS_GAZEBO_OBSERVED' if not issues else 'ROS_NOT_YET_READY',
            'sourcePin':UPSTREAM,'rosNodes':names,'topicCounters':cc,'processes':processes(),
            'frame':msg,'realRobotCommandCount':0,'simulation':True,
            'observationSource':'ROS 1 Noetic topics from compiled upstream C++ publisher and Gazebo /clock; not an actual space robot'}

class Handler(BaseHTTPRequestHandler):
    def do_GET(self):
        if self.path not in ('/','/health','/frame','/evidence'):self.send_response(404);self.end_headers();return
        value=snapshot()
        if self.path=='/health':value={k:value[k] for k in ('status','sourcePin','rosNodes','topicCounters','processes','realRobotCommandCount')}
        if self.path=='/frame':value=value['frame']
        data=json.dumps(value,ensure_ascii=False,allow_nan=False).encode()
        self.send_response(200);self.send_header('Content-Type','application/json');self.send_header('Content-Length',str(len(data)));self.send_header('Cache-Control','no-store');self.end_headers();self.wfile.write(data)
    def log_message(self,fmt,*args):print('ACCESS',fmt%args,flush=True)

if __name__=='__main__':
    threading.Thread(target=subscribe,daemon=True).start()
    ThreadingHTTPServer(('0.0.0.0',int(os.environ.get('PORT','10000'))),Handler).serve_forever()
