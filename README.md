# Swiper

Solver for weight reduction problems.
See the Swiper paper for details and formal definitions.

## Running the solver

On Unix systems, clone the repository and run the following command inside the repository folder:
```bash
pip install -r requirements.txt
```

You can verify the installation and see some useful options by running:
```bash
./main.py wr --help  # for the weight restriction problem
./main.py wq --help  # for the weight qualification problem
./main.py ws --help  # for the weight separation problem
```

You can run the tests by executing:
```bash
./scripts/test-all.sh
``` 

Solver usage examples (see the help message for more details):
```bash
# -v enables verbose logging. -vv enables debug logging. --debug verifies the solution.
./main.py wr --tw 1/3 --tn 1/2 ./examples/aptos.dat -vv --debug
# --tw and --tn can be fractions or decimals
./main.py wr --tw 0.3 --tn 1/3 ./examples/tezos.dat -v --debug
# --sum-only only prints the total number of assigned tickets instead of the assignment itself
./main.py wq --tw 1/3 --tn 1/4 ./examples/filecoin.dat --sum-only
# --linear makes the solver run in quasilinear time
./main.py ws --alpha 0.3 --beta 0.35 ./examples/algorand.dat --sum-only --linear
```
