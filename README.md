# My ML Project: Iris Classification (Debugged)

This project takes a machine learning script that had several bugs, fixes it, and adds unit tests and an automatic
test run on GitHub Actions.

## What the program does
`fixed_program.py` loads the iris flower dataset, scales the four measurements with `StandardScaler`, splits the data
80/20 into training and test sets, trains a logistic regression model, and prints the accuracy.

## Bugs that were fixed
- Restored the missing `load_iris` import.
- Used the real column names (`sepal length (cm)` and so on) instead of `sepal_length`.
- Used `fit_transform` instead of `fit`, so the scaler returns data and not a scaler object.
- Added `max_iter=1000` so the model converges, and made `train_model` return the accuracy so it can be tested.

## Files
| File | Purpose |
|---|---|
| `fixed_program.py` | The corrected machine learning script |
| `test_fixed_program.py` | Unit tests written with `unittest` |
| `requirements.txt` | Python packages needed |
| `.github/workflows/python-tests.yml` | GitHub Actions workflow that runs the tests on every push |

## How to run
```bash
pip install -r requirements.txt
python fixed_program.py
```

## How to run the tests
```bash
python -m unittest discover -v
```

## Continuous integration
Every push to GitHub triggers the **Python Tests** workflow, which installs the requirements and runs the unit
tests. A green tick next to the latest commit (and in the **Actions** tab) means all tests passed.
