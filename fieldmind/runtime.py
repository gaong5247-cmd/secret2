"""Windows screen capture, preview, opt-in *local input* with emergency stop.

No Roblox process memory access, hooks into games, injection or anti-cheat bypass.
"""
from __future__ import annotations
import ctypes
import json
import time
from pathlib import Path
from .core import BallDynamics, TacticalPlanner, World, Vec2, Decision, features
from .vision import Detector


def read_config(path:str)->dict:
    return json.loads(Path(path).read_text(encoding="utf-8"))


def focused_window_contains(title: str)->bool:
    if not title:return False
    try:
        hwnd=ctypes.windll.user32.GetForegroundWindow()
        buf=ctypes.create_unicode_buffer(512)
        ctypes.windll.user32.GetWindowTextW(hwnd,buf,512)
        return title.casefold() in buf.value.casefold()
    except (AttributeError,OSError):
        return False


def f8_pressed()->bool:
    try:return bool(ctypes.windll.user32.GetAsyncKeyState(0x77)&0x8000)
    except (AttributeError,OSError):return False


class Actuator:
    def __init__(self, cfg:dict):
        import pydirectinput
        self.io=pydirectinput
        self.cfg=cfg
        self.last_action="idle"
        self.last_pulse=0.0
        self.pressed:set[str]=set()

    def release(self):
        for key in list(self.pressed):
            self.io.keyUp(key)
        self.pressed.clear()

    def apply(self,d:Decision,ball:Vec2|None,own:Vec2):
        keys=self.cfg.get("controls",{})
        desired=set()
        if d.action in ("intercept","block","dribble") and d.target:
            dx=d.target.x-own.x;dy=d.target.y-own.y
            # Only an INITIAL SCREEN-SPACE HEURISTIC: camera movement breaks mapping!
            if abs(dx)>.065:desired.add(keys.get("right","d") if dx>0 else keys.get("left","a"))
            if abs(dy)>.07:desired.add(keys.get("back","s") if dy>0 else keys.get("forward","w"))
        desired.discard(None)
        for key in self.pressed-desired:self.io.keyUp(key)
        for key in desired-self.pressed:self.io.keyDown(key)
        self.pressed=desired
        now=time.monotonic()
        if d.action in ("shoot","pass","tackle") and now-self.last_pulse>0.8:
            # User must explicitly verify and set the key or mouse button in game Help.
            control=keys.get(d.action)
            if control:
                if control=="mouse_left":self.io.click(button="left")
                elif control=="mouse_right":self.io.click(button="right")
                else:self.io.press(control)
                self.last_pulse=now
        self.last_action=d.action


def run(config_path:str,armed:bool=False,student_path:str|None=None):
    import cv2,mss,numpy as np
    cfg=read_config(config_path)
    if armed and not cfg.get("allow_input",False):
        raise RuntimeError("Set allow_input=true AND use --arm to enable input; private/test environments only")
    detector=Detector(cfg)
    dyn=BallDynamics();planner=TacticalPlanner()
    student=None
    if student_path:
        from .student import Student
        student=Student(student_path)
    sct=mss.mss()
    user_title=cfg.get("target_window_title","Roblox")
    actuator=Actuator(cfg) if armed else None
    fps=max(1,min(30,int(cfg.get("fps",12))))
    last_log=0.0;last_plan=0.0
    plan=Decision("idle",reason="initializing")
    try:
        while True:
            start=time.monotonic()
            if armed and f8_pressed():
                print("EMERGENCY F8 STOP")
                break
            if armed and not focused_window_contains(user_title):
                actuator.release()
                time.sleep(.10)
                continue
            monitor=sct.monitors[1]
            frame=cv2.cvtColor(np.asarray(sct.grab(monitor)),cv2.COLOR_BGRA2BGR)
            p=detector.find(frame)
            dyn.update(start,p.ball)
            world=World(ball=p.ball,ball_velocity=dyn.velocity,ball_confidence=p.confidence,
                        self_pos=Vec2(*cfg.get("self_screen_point",[.5,.82])),players=p.players,
                        attack_goal=Vec2(*cfg["attack_goal"]) if cfg.get("attack_goal") else None,
                        defend_goal=Vec2(*cfg["defend_goal"]) if cfg.get("defend_goal") else None)
            if start-last_plan>.20:
                plan=planner.decide(world,dyn.predict(.25))
                if student is not None:
                    action,confidence=student.predict(features(world))
                    # safety gate: don't attempt ungrounded ball-dependent actions
                    if p.ball is not None and p.confidence>=.60 and confidence>=.85:
                        if action not in ("pass","shoot") or cfg.get("allow_student_strikes",False):
                            plan=Decision(action,plan.target,"trained student",confidence)
                last_plan=start
            if armed and actuator:
                actuator.apply(plan,p.ball,world.self_pos)
            else:
                if p.ball is not None:
                    cx=int(p.ball.x*frame.shape[1]);cy=int(p.ball.y*frame.shape[0])
                    cv2.circle(frame,(cx,cy),16,(0,0,255),2)
                    projected=dyn.predict(.25)
                    if projected:
                        cv2.circle(frame,(int(projected.x*frame.shape[1]),int(projected.y*frame.shape[0])),8,(0,220,220),2)
                for player in p.players:
                    cx=int(player.foot.x*frame.shape[1]);cy=int(player.foot.y*frame.shape[0])
                    cv2.rectangle(frame,(cx-12,cy-35),(cx+12,cy),(255,200,0),2)
                cv2.putText(frame,f"{plan.action} / ball_conf={p.confidence:.2f} / {plan.reason}",(15,40),
                            cv2.FONT_HERSHEY_SIMPLEX,.65,(20,240,30),2)
                cv2.imshow("FieldMind XPU - OBSERVE ONLY (ESC to close)",frame)
                if cv2.waitKey(1)&0xff==27:break
            if start-last_log>2:
                print(f"action={plan.action} reason={plan.reason} conf={p.confidence:.2f} device="+
                      (student.device if student else "cpu-rules"))
                last_log=start
            lag=time.monotonic()-start
            if lag<1/fps:time.sleep(1/fps-lag)
    finally:
        if actuator:actuator.release()
        sct.close()
        if not armed:cv2.destroyAllWindows()
