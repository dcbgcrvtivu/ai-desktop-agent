"""
Experiment 4: Task Orchestration Runner
"""
from __future__ import annotations
import argparse, html, json, time
from pathlib import Path
from PIL import Image, ImageDraw, ImageFont
from dataclasses import asdict
from src.orchestrator import OUTPUT_DIR, ROOT, RUN_LOG, TaskResult, now_text, run_task
from src.scene_registry import ALL_SCENES, get_scene
DEMO_HTML = OUTPUT_DIR / "demo.html"

def font(size, bold=False):
    for c in ["C:/Windows/Fonts/msyhbd.ttc" if bold else "C:/Windows/Fonts/msyh.ttc","C:/Windows/Fonts/simhei.ttf","C:/Windows/Fonts/arialbd.ttf" if bold else "C:/Windows/Fonts/arial.ttf"]:
        try: return ImageFont.truetype(c,size=size)
        except: pass
    return ImageFont.load_default()

FONT_T,FONT_H,FONT_B,FONT_S = font(32,True),font(22,True),font(17),font(14)

def rd(d,b,f,o="#cbd5e1",w=1): d.rounded_rectangle(b,8,f,o,w)
def tx(d,xy,v,fnt,c="#111827"): d.text(xy,v,font=fnt,fill=c)

def wl(text,w):
    L=[]
    for rl in text.splitlines() or [""]:
        c,cu="",0
        for ch in rl:
            u=2 if ord(ch)>127 else 1
            if c and cu+u>w: L.append(c);c=ch;cu=u
            else: c+=ch;cu+=u
        L.append(c)
    return L

def panel(d,b,t,lines,f="#ffffff"):
    rd(d,b,f)
    tx(d,(b[0]+18,b[1]+16),t,FONT_H)
    y=b[1]+58
    for l in lines:
        for wl2 in wl(l,54): tx(d,(b[0]+18,y),wl2,FONT_B,"#334155"); y+=26
        y+=6

def dp(d):
    rd(d,(74,405,550,720),"#f8fafc")
    rd(d,(112,435,512,499),"#ffffff","#cbd5e1")
    tx(d,(120,449),"\u65e0\u6807\u9898 - \u8bb0\u4e8b\u672c",FONT_B,"#111827")
    for i,l in enumerate(["\u5b9e\u9a8c\u56db\uff1a\u4efb\u52a1\u7f16\u6392\u4e0e\u7cfb\u7edf\u96c6\u6210\u5b9e\u6218","\u5b66\u53f7\uff1a0000000000  \u59d3\u540d\uff1aDemo User","","\u672c\u5b9e\u9a8c\u901a\u8fc7\u96c6\u6210\u5c4f\u5e55\u8bc6\u522b\uff08\u5b9e\u9a8c1\uff09\u3001","\u81ea\u52a8\u5316\u64cd\u4f5c\uff08\u5b9e\u9a8c2\uff09\u548c\u5927\u6a21\u578bAPI\uff08\u5b9e\u9a8c3\uff09\u3001","\u5b9e\u73b0\u4e86\u5b8c\u6574\u7684\u95ed\u73af\u3002"][:5]): tx(d,(120,515+i*28),l,FONT_S,"#334155")

def dm(d):
    rd(d,(74,405,550,720),"#f8fafc")
    rd(d,(200,450,430,680),"#e2e8f0","#94a3b8")
    tx(d,(240,430),"\u626b\u96f7 9x9 (10\u9897\u96f7)",FONT_B)
    for r9 in range(9):
        for c9 in range(9): rd(d,(210+c9*24,460+r9*24,232+c9*24,482+r9*24),"#cbd5e1","#94a3b8",1)

def si(scene,result):
    img=Image.new("RGB",(1280,760),"#eef2f7"); d=ImageDraw.Draw(img)
    d.rectangle((0,0,1280,88),fill="#101828")
    tm={}; tm["notepad"]="\u573a\u666f1\uff1a\u8bb0\u4e8b\u672c - \u6253\u5f00->\u8f93\u5165->\u4fdd\u5b58\u5230\u684c\u9762"
    tm["minesweeper"]="\u573a\u666f4\uff1a\u626b\u96f7\u81ea\u52a8\u70b9\u51fb"
    tx(d,(34,20),tm.get(scene.name,scene.name),FONT_T,"#ffffff")
    ts="%.1f"%result.total_time_sec
    tx(d,(34,58),"\u95ed\u73af %s | \u6b65\u9aa4 %d | \u7528\u65f6 %ss | %s"%(result.status,result.total_steps,ts,result.final_observation),FONT_S,"#cbd5e1")
    panel(d,(34,120,594,382),"\u5c4f\u5e55OCR\u8f93\u5165",wl(scene.initial_screen_text,58))
    log=[("%02d. [%s] %s: %s"%(i["step"],i["phase"],i["action"],i["result"])) for i in result.execution_log[:8]]
    if len(result.execution_log)>8: log.append("... \u5171 %d \u6761\u8bb0\u5f55"%len(result.execution_log))
    panel(d,(630,120,1246,720),"\u95ed\u73af\u6267\u884c\u65e5\u5fd7",log)
    if scene.name=="notepad": dp(d)
    else: dm(d)
    p=OUTPUT_DIR/("%s.png"%scene.name); img.save(p); return p

def bh(results):
    cards=[]
    for n,(s,r,ip) in results.items():
        rel=str(ip.relative_to(ROOT).as_posix())
        if r.status=="PASS": sty="background:#dcfce7;color:#166534"
        elif r.status=="FAIL": sty="background:#fee2e2;color:#991b1b"
        else: sty="background:#fef9c3;color:#854d0e"
        meta=('<div class="meta" style="%s;padding:8px 14px;border-radius:6px;margin:8px 0">\u72b6\u6001: %s | \u6b65\u9aa4: %d | \u6210\u529f: %d | \u5931\u8d25: %d | \u7528\u65f6: %.1fs</div>'%(sty,r.status,r.total_steps,r.successful_steps,r.failed_steps,r.total_time_sec))
        cards.append('<section><h2>%s</h2>%s<div class="meta" style="margin:4px 0;font-size:13px">%s</div><img src="../%s" alt="%s"></section>'%(n,meta,html.escape(r.final_observation),html.escape(rel),html.escape(n)))
    sj=json.dumps({k:{"status":v[1].status,"steps":v[1].total_steps,"time":"%.1fs"%v[1].total_time_sec} for k,v in results.items()},ensure_ascii=False,indent=2)
    htm='<!doctype html><html lang="zh-CN"><meta charset="utf-8"><title>\u5b9e\u9a8c\u56db\uff1a\u4efb\u52a1\u7f16\u6392\u4e0e\u7cfb\u7edf\u96c6\u6210\u5b9e\u6218</title><style>body{margin:0;background:#eef2f7;color:#111827;font-family:"Microsoft YaHei",Arial,sans-serif}main{max-width:1120px;margin:0 auto;padding:28px}h1{font-size:28px;margin:0 0 18px}section{background:white;border:1px solid #d7dee8;border-radius:8px;margin:18px 0;padding:16px}h2{font-size:18px;margin:0 0 12px}.meta{font-size:14px;padding:4px 0}img{width:100%%;height:auto;border:1px solid #cbd5e1;border-radius:6px}pre{background:#f8fafc;padding:12px;border-radius:6px;font-size:13px;overflow:auto}</style><main><h1>\u5b9e\u9a8c\u56db\uff1a\u4efb\u52a1\u7f16\u6392\u4e0e\u7cfb\u7edf\u96c6\u6210\u5b9e\u6218</h1><p style="margin:0 0 12px;font-size:15px;color:#334155">Branch B - \u57fa\u4e8e\u5927\u6a21\u578bAPI\u7684\u7cfb\u7edf\u96c6\u6210 | \u611f\u77e5->\u51b3\u7b56->\u6267\u884c->\u9a8c\u8bc1 \u95ed\u73af</p>%s<hr><pre>%s</pre></main></html>'%("".join(cards),html.escape(sj))
    DEMO_HTML.write_text(htm,encoding="utf-8")

def write_aggregate_log(results):
    payload={
        "experiment":"实验四：任务编排与系统集成实战",
        "branch":"Branch B",
        "summary":{
            "scenario_count":len(results),
            "pass_count":sum(1 for _,r,_ in results.values() if r.status=="PASS"),
            "total_steps":sum(r.total_steps for _,r,_ in results.values()),
            "generated_at":now_text(),
        },
        "results":[asdict(r) for _,r,_ in results.values()],
    }
    RUN_LOG.write_text(json.dumps(payload,ensure_ascii=False,indent=2),encoding="utf-8")

def ap():
    p=argparse.ArgumentParser()
    p.add_argument("--scenario",choices=["all","notepad","minesweeper"],default="all")
    p.add_argument("--mode",choices=["simulate","real"],default="simulate")
    p.add_argument("--confirm",default=""); p.add_argument("--loops",type=int,default=5)
    return p.parse_args()

def main():
    a=ap(); sn=list(ALL_SCENES.keys()) if a.scenario=="all" else [a.scenario]; rl={}
    for n in sn:
        sc=get_scene(n); print(); print("=== \u8fd0\u884c\u573a\u666f: %s ==="%sc.name)
        print("\u76ee\u6807: %s"%sc.goal)
        r=run_task(task_name=sc.name,goal=sc.goal,description=sc.description,
            initial_screen_text=sc.initial_screen_text,targets=sc.targets,
            mode=a.mode,confirm=a.confirm,max_loops=a.loops,scene_data=sc.data)
        print("  \u72b6\u6001=%s, \u6b65\u9aa4=%d, \u6210\u529f=%d, \u5931\u8d25=%d, \u7528\u65f6=%.1fs"%(
            r.status,r.total_steps,r.successful_steps,r.failed_steps,r.total_time_sec))
        print("  \u89c2\u5bdf: %s"%r.final_observation)
        ip=si(sc,r); rl[n]=(sc,r,ip)
    write_aggregate_log(rl); bh(rl); print("\n=== \u6f14\u793aHTML: %s ==="%DEMO_HTML)
    print("\u573a\u666f\u603b\u6570: %d, \u901a\u8fc7: %d, \u603b\u6b65\u9aa4: %d"%(len(rl),sum(1 for _,r,_ in rl.values() if r.status=="PASS"),sum(r.total_steps for _,r,_ in rl.values())))
    print("\u4efb\u52a1\u5b8c\u6210\u3002")

if __name__=="__main__": main()
