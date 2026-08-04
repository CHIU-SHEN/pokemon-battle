# Primary budgeted MCTS candidate

This isolated candidate uses the frozen `crustle_kangaskhan_cage` policy with
deadline-aware belief-PUCT. It does not modify or authorize replacement of the
formal Abomasnow submission.

Default runtime limits are 8 simulations, 1 belief particle, depth 4, 30 ms per
decision, and 2 seconds of cumulative search per game. The `PTCG_MCTS_*`
environment variables documented in `main.py` may override these values during
evaluation.
