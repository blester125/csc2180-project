# CSC2180 - Automatic Reasoning with ML
##

Code for our course project.

Project Proposal on Overleaf: https://www.overleaf.com/project/67217e8a5a61dae58bff3d1e


### Development

1) Install requirements `pip install -r requirements.txt`
2) Install our package `pip install -e .`
3) Install pre-commit hooks `pre-commit install`

When you make a commit, pre-commit will run isort, black, and whitespace trimming on your code. If you get failures when
you run `git commit`, the changes will be available for staging. Verify them with `git add -p` and then run `git commit`
again.

### Previous Ideas

* Interactive Interactive Theorem Prover
  Chatbots are bad a lean. Can we feed the lean state to the chatbot to make is better at lean.
  Would need to make our own dataset (there is a step-by-step dataset on hugging face "lean workbook").
  ChatGPT tends to get stuck in a loop and makes a bunch of changes but never fixes the main issue.
  Multiple people said this was basially their project (with less cool branding?) during paper presentations on LLMs+Lean so it's good we aren't.
* ML For Code
  Tell the LLM to generate multiple implementations and then pick the "best" one. Like synthesis search

  Run Fuzzy testing on code outputs from an LLM. Feed the errors back and see if it fixes them. Build a dataset with Rosetta code? Would need to make sure rosetta code isn't in the training data?
* Finding LLM consistency with SAT between multiple runs of the LLM
  The LLM is a black box function where we query it multiple times (different inputs or uses the LLM randomness), convert the output to SAT and make sure that the outputs are consistent.
* Can SAT solving be used for VC-dimensions?
  Can you SAT dim n and UNSAT dim n + 1 as proof that the VC dim = n?
* SMT/SAT solving for a model/dataset instead of using gradient descent?
  Need to use z3. Can we implement things like AND/OR/XOR?
  Can we use the UNSAT core as a way to find labels that aren't linearly separable?
  Performance scaling for: #features, #examples, #model size?
  Real vs Int models?
  Implement different loss functions? (max margin, etc)
  Use MaxSAT to solve non-separable datasets
  Random feature project to solve non-separable? Depends on how #features scale
