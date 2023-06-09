from package import Agent, Env
import numpy as np

def test_agent_update_model():
    a = Agent()
    a.update_model(6,1,7,0)
    a.update_model(11,3,10,0)
    a.update_model(111,0, 110, 1)

    assert a.model ==  {6: {1: (7, 0)}, 
                    11: {3: (10, 0)}, 
                    111: {0: (110, 1)}}
    
def test_argmax():
    """
    Tests the random tie breaking and the reproducibility 
    of the experiments
    """
    a = Agent()
    q_values = [0,2,3,4,1,2,4,4,3,4]
    selection = [a.argmax(q_values) for _ in range(10)]
    assert selection == [9, 6, 9, 7, 7, 6, 3, 6, 7, 9]

def test_coord_to_state():
    a = Agent()
    rows, cols = a.env.grid.index, a.env.grid.columns
    for row in rows:
        for col in cols:
            assert a.coord_to_state((col,row)) == col*10 + row

def test_state_coord_identity():
    a = Agent()
    rows, cols = a.env.grid.index, a.env.grid.columns
    for row in rows:
        for col in cols:
            assert a.state_to_coord(a.coord_to_state((col,row))) == (col,row)


def test_update_state_movement():
    a = Agent()
    # attempt to leave the grid from the bottom left corner
    assert a.update_state(7, 3) == 7
    assert a.update_state(7, 2) == 7 
    # attempt to leave the grid from the top right corner
    assert a.update_state(110, 0) == 110
    assert a.update_state(110, 1) == 110
    # test normal movement in the center of the grid
    assert a.update_state(35, 0) == 34
    assert a.update_state(35, 1) == 45
    assert a.update_state(35, 2) == 36
    assert a.update_state(35, 3) == 25

def test_epsilon_greedy_selection():
    """
    Test greedy selection and random tie breaking
    """
    a = Agent()
    a.q_values[107] = [0,1,0,1]
    assert [a.epsilon_greedy(107) for _ in range(10)] == [3, 3, 2, 3, 3, 3, 1, 3, 1, 3]

def test_agent_start():
    a = Agent()
    past_action = a.agent_start(a.coord_to_state(a.env.coordinates.get('G')[0]))
    assert past_action == 3
    past_action = a.agent_start(a.coord_to_state(a.env.coordinates.get('P')[1]))
    assert past_action == 1
    past_action = a.agent_start(74)
    assert past_action == 2

def test_planning_step():
    a = Agent(planning_steps=10)
    a.update_model(0,2,1,1)
    a.update_model(2,0,1,1)
    a.update_model(0,3,0,1)
    a.update_model(1,1,-1,1)

    expected_model = {
        0: {2: (1,1), 3: (0,1)},
        1: {1: (-1,1)},
        2: {0: (1,1)},
    }

    assert expected_model == a.model

    a.planning_step()
    expected_q_values = [np.array([0.    , 0.    , 0.1271, 0.2   ], dtype=np.float32),
                        np.array([0.    , 0.3439, 0.    , 0.    ], dtype=np.float32),
                        np.array([0.3152, 0.    , 0.    , 0.    ], dtype=np.float32)]

    assert np.all(np.isclose(expected_q_values, list(a.q_values.values())[:3])) 

def test_agent_start_step_stop():
    a = Agent(planning_steps=2)

    # ----------------
    # test agent start
    # ----------------
    action = a.agent_start(a.start_position)
    assert action == 3
    assert a.model == {}
    for action_values in list(a.q_values.values()):
        assert np.all(action_values == 0)
    
    # ----------------
    # test agent step
    # ----------------
    action = a.step(1,2)
    assert action == 1
    action = a.step(0,1)
    assert action == 0

    expected_model = {
        61: {3: (1,2)},
        
    }