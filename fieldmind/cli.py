"""CLI entrypoint. Nothing controls a game unless explicitly --arm and config opt-in."""
from __future__ import annotations
import argparse,json,random,time
from .core import Vec2,Player,World,BallDynamics,TacticalPlanner,features


def main():
    ap=argparse.ArgumentParser(prog="fieldmind")
    sub=ap.add_subparsers(dest="cmd",required=True)
    check=sub.add_parser("check",help="inspect installed Intel XPU availability")
    demo=sub.add_parser("simulate",help="run a deterministic lightweight strategy simulation")
    demo.add_argument("--steps",type=int,default=15)
    mk=sub.add_parser("make-toy",help="generate synthetic RULE-teacher examples, not expert demonstrations")
    mk.add_argument("--out",default="data/toy.jsonl")
    mk.add_argument("--count",type=int,default=2000)
    train=sub.add_parser("train",help="train tiny PyTorch policy from labeled JSONL")
    train.add_argument("--dataset",required=True);train.add_argument("--output",default="models/student.pt")
    train.add_argument("--epochs",type=int,default=30);train.add_argument("--cpu",action="store_true")
    calib=sub.add_parser("calibrate-ball",help="select game ball from screenshot, save local template")
    calib.add_argument("--config",default="configs/tps_ultimate.json")
    run=sub.add_parser("run",help="preview only, unless BOTH --arm and config allows")
    run.add_argument("--config",default="configs/tps_ultimate.json")
    run.add_argument("--student",default=None);run.add_argument("--arm",action="store_true")
    args=ap.parse_args()
    if args.cmd=="check":
        try:
            import torch
            print("torch:",torch.__version__)
            print("Intel XPU:",bool(hasattr(torch,"xpu") and torch.xpu.is_available()))
        except ImportError:print("PyTorch not installed (CPU rules still work)")
    elif args.cmd=="simulate":
        dyn=BallDynamics();plan=TacticalPlanner()
        for i in range(args.steps):
            p=Vec2(.25+.03*i,.48+.004*i)
            dyn.update(i*.08,p)
            w=World(ball=p,ball_velocity=dyn.velocity,ball_confidence=.95,
                    players=[Player(Vec2(.55,.51),"enemy"),Player(Vec2(.7,.25),"ally")],
                    attack_goal=Vec2(.5,0),defend_goal=Vec2(.5,1))
            choice=plan.decide(w,dyn.predict(.22))
            print(f"{i:02d} ball=({p.x:.2f},{p.y:.2f}) action={choice.action:10} target={choice.target}")
    elif args.cmd=="make-toy":
        from pathlib import Path
        rng=random.Random(32);plan=TacticalPlanner()
        path=Path(args.out);path.parent.mkdir(parents=True,exist_ok=True)
        counts={}
        with path.open("w",encoding="utf-8") as f:
            for _ in range(args.count):
                ball=Vec2(rng.uniform(.05,.95),rng.uniform(.06,.93))
                w=World(ball=ball,ball_velocity=Vec2(rng.uniform(-.1,.1),rng.uniform(-.1,.1)),
                        ball_confidence=.95,self_pos=Vec2(.5,.82),
                        players=[Player(Vec2(rng.random(),rng.random()),"enemy"),
                                 Player(Vec2(rng.random(),rng.random()),"ally")],
                        attack_goal=Vec2(.5,0),defend_goal=Vec2(.5,1))
                action=plan.decide(w).action
                counts[action]=counts.get(action,0)+1
                f.write(json.dumps({"features":features(w),"action":action},ensure_ascii=False)+"\n")
        print("Synthetic data:",path,"labels:",counts)
    elif args.cmd=="train":
        from .student import train as fit
        print(json.dumps(fit(args.dataset,args.output,args.epochs,not args.cpu),indent=2))
    elif args.cmd=="calibrate-ball":
        from .vision import calibrate_ball
        from .runtime import read_config
        calibrate_ball(read_config(args.config))
    elif args.cmd=="run":
        from .runtime import run as live
        live(args.config,args.arm,args.student)

if __name__=="__main__":main()
