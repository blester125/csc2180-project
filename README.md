# CSC2180 - Automatic Reasoning with ML
##

Code for our course project.


### Development

1) Install requirements `pip install -r requirements.txt`
2) Install our package `pip install -e .`
3) Install pre-commit hooks `pre-commit install`

When you make a commit, pre-commit will run isort, black, and whitespace trimming on your code. If you get failures when
you run `git commit`, the changes will be available for staging. Verify them with `git add -p` and then run `git commit`
again.
