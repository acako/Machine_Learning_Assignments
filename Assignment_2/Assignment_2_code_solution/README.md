# CPS803 Problem Set Instructions

## Setup for Coding Parts

1. Install [Miniconda](https://docs.conda.io/en/latest/miniconda.html)
  - Conda is a package manager that sandboxes your project’s dependencies in a virtual environment
  - Miniconda contains Conda and its dependencies with no extra packages by default (as opposed to Anaconda, which installs some extra packages)
2. Extract the zip file and run `conda env create -f environment.yml` from inside the extracted directory.
  - This creates a Conda environment called `cps803`
3. Run `source activate cps803`
  - This activates the `cps803` environment
  - Do this each time you want to write/test your code
4. (Optional) If you use PyCharm:
  - Open the `src` directory in PyCharm
  - Go to `PyCharm` > `Preferences` > `Project` > `Project interpreter`
  - Click the gear in the top-right corner, then `Add`
  - Select `Conda environment` > `Existing environment` > Button on the right with `…`
  - Select `/Users/YOUR_USERNAME/miniconda3/envs/cps803/bin/python`
  - Select `OK` then `Apply`
5. Notice some coding problems come with `util.py` file. In it you have access to methods that do the following tasks:
  - Load a dataset in the CSV format provided in the problem
  - Add an intercept to a dataset (*i.e.,* add a new column of 1s to the design matrix)
  - Plot a dataset and a linear decision boundary. Some plots might require modified plotting code, but you can use this as a starting point.
7. Notice that start codes are provided in each problem directory (e.g. `gda.py`, `posonly.py`)
  - Within each starter file, there are highlighted regions of the code with the comments ** START CODE HERE ** and ** END CODE HERE **. You are strongly suggested to make your changes only within this region. You can add helper functions within this region as well.
8. Once you are done with all the code changes, run `make_zip.py` to create a `submission.zip`.
  - You must upload this `submission.zip` to D2L along with your assignment written report.
