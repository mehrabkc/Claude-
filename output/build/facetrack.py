import cv2, numpy as np, subprocess, json
SRC="/root/.claude/uploads/28cefa92-73e1-5625-b696-5063f9b1eaf4/ed0a8c14-copy_CED29BB7-182F-40F1-ABE0-8B69F31CDEFD.mov"
FPS=30; W,H=270,480
p=subprocess.Popen(["ffmpeg","-v","error","-i",SRC,"-t","15.9","-vf",f"fps={FPS},scale={W}:{H},format=gray","-f","rawvideo","-"],stdout=subprocess.PIPE)
raw=p.stdout.read(); n=len(raw)//(W*H); fr=np.frombuffer(raw,np.uint8).reshape(n,H,W)
cas=[cv2.CascadeClassifier(cv2.data.haarcascades+f) for f in ["haarcascade_frontalface_default.xml","haarcascade_frontalface_alt2.xml","haarcascade_profileface.xml"]]
res=[]
for i in range(n):
    g=cv2.equalizeHist(fr[i]); best=None
    for c in cas[:2]:
        d=c.detectMultiScale(g,1.1,4,minSize=(30,30))
        if len(d): best=max(d,key=lambda r:r[2]*r[3]); break
    if best is None:
        d=cas[2].detectMultiScale(g,1.1,4,minSize=(30,30))
        if len(d): best=max(d,key=lambda r:r[2]*r[3])
    res.append(None if best is None else [(best[0]+best[2]/2)/W,(best[1]+best[3]/2)/H,best[2]/W])
json.dump(res,open("faces_raw.json","w"))
print(n, sum(r is None for r in res))
for i in range(0,n,15): print(i/FPS, res[i])
