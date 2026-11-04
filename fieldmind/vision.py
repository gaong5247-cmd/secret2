"""Pixel-only perception. Ball template fallback + optional domain-trained YOLO ONNX.

Class IDs must correspond to the supplied model, not an imagined Roblox schema.
"""
from __future__ import annotations
from dataclasses import dataclass
from pathlib import Path
from typing import Optional
from .core import Vec2, Player

@dataclass
class Perception:
    ball: Optional[Vec2]
    confidence: float
    players: list[Player]


class TemplateBall:
    def __init__(self, path: str, threshold: float = .68):
        import cv2
        self.cv2 = cv2
        self.template = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
        if self.template is None:
            raise FileNotFoundError("Use calibrate-ball to create a ball template: " + path)
        self.threshold=threshold
        self.last_px: Optional[tuple[int,int]]=None

    def find(self, frame):
        cv2 = self.cv2
        gray=cv2.cvtColor(frame,cv2.COLOR_BGR2GRAY)
        h,w=gray.shape[:2]; th,tw=self.template.shape
        if h<th or w<tw: return None
        region=(0,0,w,h)
        if self.last_px:
            x,y=self.last_px
            rad=max(100,tw*5)
            left,top=max(0,x-rad),max(0,y-rad)
            right,bottom=min(w,x+rad+tw),min(h,y+rad+th)
            if right-left>=tw and bottom-top>=th:region=(left,top,right,bottom)
        def find_in(box):
            l,t,r,b=box
            result=cv2.matchTemplate(gray[t:b,l:r],self.template,cv2.TM_CCOEFF_NORMED)
            _,v,_,loc=cv2.minMaxLoc(result)
            return float(v),(loc[0]+l+tw//2,loc[1]+t+th//2)
        score, center=find_in(region)
        if score<self.threshold and region!=(0,0,w,h):score,center=find_in((0,0,w,h))
        if score<self.threshold:
            self.last_px=None
            return None
        self.last_px=center
        return Vec2(center[0]/w,center[1]/h),score


class OnnxYOLO:
    """YOLOv8/YOLO11 exported ONNX detection output [1,84,N] or [1,N,84].

    The ONNX weights are NOT included. Small Roblox balls/avatars usually need
    custom training; COCO defaults are not reliable here.
    """
    def __init__(self,path:str,ball_ids:list[int],player_ids:list[int],conf:float=.45):
        import cv2
        self.cv2=cv2
        self.net=cv2.dnn.readNetFromONNX(path)
        self.ball_ids=set(ball_ids)
        self.player_ids=set(player_ids)
        self.conf=conf

    def find(self,frame)->Perception:
        import numpy as np
        cv2=self.cv2; h,w=frame.shape[:2]
        blob=cv2.dnn.blobFromImage(frame,1/255.0,(640,640),swapRB=True,crop=False)
        self.net.setInput(blob)
        raw=np.asarray(self.net.forward())
        if raw.ndim!=3 or raw.shape[0]!=1:
            raise ValueError(f"Expected YOLO 3D output, got {raw.shape}")
        rows=raw[0]
        if rows.shape[0] < rows.shape[1]:rows=rows.T
        boxes=[]; scores=[]; classes=[]
        for row in rows:
            if len(row)<6: continue
            class_id=int(np.argmax(row[4:])); score=float(row[4+class_id])
            if score<self.conf or class_id not in self.ball_ids|self.player_ids:continue
            cx,cy,bw,bh=map(float,row[:4])
            x=max(0,int((cx-bw/2)*w/640));y=max(0,int((cy-bh/2)*h/640))
            boxes.append([x,y,int(bw*w/640),int(bh*h/640)])
            scores.append(score);classes.append(class_id)
        if not boxes:return Perception(None,0,[])
        selected=cv2.dnn.NMSBoxes(boxes,scores,self.conf,.45)
        ids=[int(i) for i in np.asarray(selected).reshape(-1)] if len(selected) else []
        ball=None; ball_conf=0.;players=[]
        for i in ids:
            x,y,bw,bh=boxes[i]
            p=Vec2(min(1,max(0,(x+bw/2)/w)),min(1,max(0,(y+bh)/h)))
            if classes[i] in self.ball_ids and scores[i]>ball_conf:
                ball=Vec2(p.x,min(1,max(0,(y+bh/2)/h)));ball_conf=scores[i]
            elif classes[i] in self.player_ids:
                players.append(Player(foot=p,team="unknown",confidence=scores[i]))
        return Perception(ball,ball_conf,players)


class Detector:
    def __init__(self, config: dict):
        v=config.get("vision",{})
        self.template=None
        self.onnx=None
        if v.get("ball_template") and Path(v["ball_template"]).exists():
            self.template=TemplateBall(v["ball_template"],float(v.get("threshold",.68)))
        if v.get("onnx_path"):
            self.onnx=OnnxYOLO(v["onnx_path"], v.get("ball_ids",[32]), v.get("player_ids",[0]))

    def find(self,frame)->Perception:
        found=self.onnx.find(frame) if self.onnx else Perception(None,0.0,[])
        if self.template and (not found.ball or found.confidence<.65):
            b=self.template.find(frame)
            if b and b[1]>found.confidence:
                found.ball,found.confidence=b
        return found


def calibrate_ball(config:dict):
    import cv2,mss,numpy as np
    path=config["vision"]["ball_template"]
    with mss.mss() as sct:
        frame=cv2.cvtColor(np.array(sct.grab(sct.monitors[1])),cv2.COLOR_BGRA2BGR)
    print("Drag a SMALL rectangle around the ball; press ENTER to confirm")
    x,y,w,h=map(int,cv2.selectROI("Select BALL only",frame,showCrosshair=True))
    cv2.destroyAllWindows()
    if w<5 or h<5:raise ValueError("No useful template selected")
    Path(path).parent.mkdir(parents=True,exist_ok=True)
    cv2.imwrite(path,frame[y:y+h,x:x+w])
    print("Template saved:",path)
