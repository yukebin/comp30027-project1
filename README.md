# COMP30027 Project 1: Scam detection with naïve Bayes

The University of Melbourne
School of Computing and Information Systems
COMP30027 Machine Learning, 2025 Semester 1

In this project, I implemented supervised and semi-supervised naïve Bayes models to detect scam SMS messages using text features. The [dataset](./data/) is derived from [real-world SMS messages](https://data.mendeley.com/datasets/f45bkkt8pr/1) and has been split into supervised training, unlabelled and test sets.

The full **project specification** is available in the [`specification/`](./specification/) folder. The **written report** can be found in [`report/`](./report/), and the implementation is contained in [`COMP30027_2025_asst1.ipynb`](COMP30027_2025_asst1.ipynb).

## Environment Setup

If you are using the full Anaconda distribution, no installation is needed - required packages are already installed.

However, if you're using minimal Python setup, to ensure reproducibility and avoid dependency conflicts, it's **recommended to run this notebook in a virtual environment**.

### Step 1: Create and activate viertual environment

```bash
python -m venv myenv
source myenv/bin/activate # On Windows: myenv\Scripts\activate
```

### Step 2: Install dependencies

```bash
pip install -r requirements.txt
```