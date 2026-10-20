import sys, numpy as np, torch
sys.argv=["x"]; torch.set_default_dtype(torch.float64)
import importlib.util as u
src=open("experiments/param_geometry/lmi_fingerprint.py").read().split("res = []")[0]
exec(src)
from torch.func import functional_call
def blocks(m):
    names,vals=zip(*[(k,v.detach().clone()) for k,v in m.named_parameters()])
    class W(nn.Module):
        def __init__(s,m): super().__init__(); s.m=m
        def forward(s): return torch.cat([t.reshape(-1) for t in s.m._frame()])
    w=W(m); f=lambda *v: functional_call(w,dict(zip(["m."+k for k in names],v)),())
    Js=torch.autograd.functional.jacobian(f,vals)
    out={}
    for k,j in zip(names,Js):
        j=j.reshape(j.shape[0],-1); s=torch.linalg.svdvals(j); out[k.replace("parametrizations.","").replace(".original","")]=f"{s[0]:.1e}"
    return out
for kind in ("hinf_sqrtm","hinf_riccati"):
  for pd in ("geotorch","bm"):
    K,CP,RE=[],[],[]
    for seed in range(8):
        torch.manual_seed(seed); m=build(kind,6,2,2,pd)
        fp=fingerprint(m); A=m._frame()[0].detach()
        K.append(fp["kappa_range"]); CP.append(float(torch.linalg.cond(m.P.detach()))); RE.append(float(torch.linalg.eigvals(A).real.max()))
    r=np.corrcoef(np.log(K),np.log(CP))[0,1]
    print(kind,pd,"med kappa %.1e"%np.median(K),"med cond(P) %.1e"%np.median(CP),"corr log k/log condP %.2f"%r,"med maxRe(A) %.3f"%np.median(RE))
    print("   block top sv:",blocks(m))
