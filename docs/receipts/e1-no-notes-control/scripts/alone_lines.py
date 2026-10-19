import re,sys,json
# per night: the last alone_asked "done" lines before the night summary
def nights(path):
    lines=open(path,errors="replace").read().split("\n")
    out={}; buf=[]; cur=None
    for i,l in enumerate(lines):
        m=re.search(r"notes promotion: night (\d+) begins",l)
        if m: cur=int(m.group(1)); buf=[]; continue
        m=re.search(r"notes promotion: night (\d+): (\d+) due.*alone (\d+) of (\d+) right",l)
        if m:
            n=int(m.group(1)); k=int(m.group(4))
            # check the lines between last done and summary
            out[n]=dict(right=int(m.group(3)),asked=k,done=buf[-k:] if k else [], before=buf[-k-3:-k] if k else [])
            continue
        if cur is not None and "done " in l and "sha=" in l:
            mm=re.search(r"prompt=(\d+) .*tokens=(\d+) sha=([0-9a-f]{12}) finish=(\w+)",l)
            buf.append((int(mm.group(1)),int(mm.group(2)),mm.group(3),mm.group(4),i))
    return out
if __name__=="__main__":
    o=nights(sys.argv[1])
    json.dump({k:v for k,v in o.items()},open(sys.argv[2],"w"))
    for n,v in sorted(o.items())[:12]:
        print(n, v["right"],"/",v["asked"], "lines:",len(v["done"]), "finish:", sum(1 for x in v["done"] if x[3]!="stop"),"length-capped")
