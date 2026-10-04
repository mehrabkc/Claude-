import json, numpy as np
from scipy.ndimage import median_filter, gaussian_filter1d
FPS=30
raw=json.load(open("faces_raw.json")); n=len(raw)
cuts=[0,int(6.0*FPS),int(11.917*FPS+0.5),n]
out=np.zeros((n,2))
for a,b in zip(cuts[:-1],cuts[1:]):
    idx=[i for i in range(a,b) if raw[i] and 0.08<raw[i][1]<0.5 and 0.15<raw[i][2]<0.5]
    x=np.array([raw[i][0] for i in idx]); y=np.array([raw[i][1] for i in idx]); t=np.array(idx)
    for _ in range(2):
        mx=median_filter(x,9,mode='nearest'); my=median_filter(y,9,mode='nearest')
        k=(abs(x-mx)<0.07)&(abs(y-my)<0.06); x,y,t=x[k],y[k],t[k]
    tt=np.arange(a,b)
    xs=gaussian_filter1d(np.interp(tt,t,x),5,mode='nearest'); ys=gaussian_filter1d(np.interp(tt,t,y),5,mode='nearest')
    out[a:b,0]=xs; out[a:b,1]=ys
    print(a,b,len(t),'/',b-a, xs.min().round(2),xs.max().round(2),ys.min().round(2),ys.max().round(2))
json.dump(out.tolist(),open("faces.json","w"))
