from fieldmind.core import ACTIONS,FEATURE_DIM,Vec2,Player,World,BallDynamics,TacticalPlanner,features


def test_physics_predicts_direction():
    d=BallDynamics()
    for i in range(15):d.update(i*.1,Vec2(.20+i*.01,.4))
    pred=d.predict(.2)
    assert pred is not None and pred.x > d.last.x
    assert abs(pred.y-.4)<.02


def test_missing_ball_refuses_action():
    assert TacticalPlanner().decide(World()).action=="idle"


def test_dribble_with_possession():
    w=World(ball=Vec2(.5,.8),ball_confidence=.99)
    assert TacticalPlanner().decide(w).action=="dribble"


def test_shoot_when_goal_close():
    w=World(ball=Vec2(.5,.8),ball_confidence=.99,attack_goal=Vec2(.5,.66))
    assert TacticalPlanner().decide(w).action=="shoot"


def test_block_carrier_lane():
    w=World(ball=Vec2(.4,.5),ball_confidence=.99,
            players=[Player(Vec2(.40,.51),"enemy")],defend_goal=Vec2(.5,1))
    assert TacticalPlanner().decide(w).action=="block"


def test_no_fabricated_team_data():
    w=World(ball=Vec2(.3,.5),ball_confidence=.99,
            players=[Player(Vec2(.4,.5),"unknown")])
    assert TacticalPlanner().decide(w).action=="intercept"


def test_features_dimension():
    assert len(features(World()))==FEATURE_DIM
    assert set(ACTIONS)=={"idle","intercept","block","tackle","dribble","pass","shoot"}


def test_dynamics_discontinuous_reset():
    d=BallDynamics()
    d.update(0,Vec2(.1,.1));d.update(.1,Vec2(.2,.1));d.update(10,Vec2(.2,.2))
    assert d.velocity==Vec2(0,0)
