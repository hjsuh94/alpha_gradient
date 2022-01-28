# Alpha-Ordered Gradients
Code for "Do Differentible Simulators Give Better Policy Gradients"? 

# Setup
Add the git repo to your `PYTHONPATH`. Then test by `import alpha_gradient` 

# Running Examples
We provide multiple examples that can be run.

To visualize the per-coordinate bias and variance on simple one-step examples, 
- BallWithWall example: `python3 examples/ball_with_wall/alpha_coordinate_sweep.py`
- Pivot example: `python3 examples/pivot/alpha_coordinate_sweep.py`

We include some trajectory optimization examples.
- Friction example: `python3 examples/friction/friction_test.py`
- Pushing with stiffness 10:  `python3 examples/curling/curling_10.py` 
- Pushing with stiffness 1000: `python3 examples/curling/curling_1000.py`
- Robot motion planning: `python3 examples/motion_planning/roomba_test.py`

Closed-loop policy optimization examples are:
- Finite-Horizon Static-Policy LQR: `python3 examples/linear_system/linear_test.py`
- Tennis: `python3 examples/breakout/run_bc_policyopt.py`
