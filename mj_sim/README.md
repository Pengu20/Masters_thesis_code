# mj_sim

A repository to demonstrate, implement, test and learn control in MuJoCo.

## Table of Contents

- [mj\_sim](#mj_sim)
  - [Table of Contents](#table-of-contents)
  - [Setup](#setup)
    - [Getting Repository](#getting-repository)
    - [Setting up a Virtual Environment](#setting-up-a-virtual-environment)
    - [Activating a Virtual Environmentv](#activating-a-virtual-environmentv)
    - [Installing Dependencies](#installing-dependencies)
  - [Usage](#usage)
    - [Running Demo](#running-demo)
- [Learning](#learning)
  - [Demo](#demo)
- [Docs](#docs)
- [Cite this work](#cite-this-work)

## Setup

### Getting Repository
To get the repository clone it into your directory of choice

```bash
git clone git@gitlab.sdu.dk:sdurobotics/teaching/mj_sim.git
```
then go into the repository by
```bash
cd mj_sim
```

### Setting up a Virtual Environment

To isolate the project's dependencies, it's recommended to use a virtual environment. If you haven't already **installed Python's venv module**, you can do so by running:

```bash
sudo apt install python3.8-venv   # For Linux (Debian/Ubuntu)
```

```bash
pip install virtualenv            # For Windows, assuming pip is installed (guide: https://phoenixnap.com/kb/install-pip-windows)
```

```bash
brew install virtualenv           # For MacOS
```

Once installed, you can **create a virtual environment** by executing the following commands:

```bash
python3.8 -m venv venv            # Linux (Ubuntu) and MacOS
```
```bash
virtualenv --python C:\Path\To\Python\python.exe venv            # Windows
```

### Activating a Virtual Environmentv
```bash
source venv/bin/activate          # Activate the virtual environment (Linux/Macos)
```
```bash
venv\Scripts\activate             # Activate the virtual environment (Windows)
```

### Installing Dependencies

After activating the virtual environment, you can install the project dependencies using pip. It's common practice to store dependencies in a `requirements.txt` file. Install the dependencies with the following command:

```bash
pip install -r requirements.txt
```

## Usage

To use the repository, simply create a new controller in the `ctrl/` directory. Currently, two examples are present to use as guidelines. Ensure your controller inherits the step function from `BaseController` to enforce consistency.

### Running Demo

To run a demo of the project, execute the following command:

```bash
python3 main.py  # Linux
mjpython main.py # MacOS
Python main.py   # Windows
```

The default controller is an operational space impedance controller. A hotkey (`Space`) is set to move the robot down to make contact with the table. </par>

This can be seen here

![test](public/opspace_demo.gif)

The manipulator is equipped with a force-torque sensor to measure forces as shown in the bottom right of the GIF above.

# Learning

## Demo

The learning demo can be run after installing the additional learning package.
```bash
python -m learning.demo
```

# Docs

A static website is generatd using [`pdoc`](https://pdoc.dev/) and can be found in `public/`. Access the documentation through 
```bash
<your-webbrowser-of-choise> public/index.html # e.g. firefox public/index.html
```
or online through the link found [here](https://sdurobotics.pages.sdu.dk/teaching/mj_sim/).

# Cite this work
To cite this work using `bibtex` please use the following
```bibtex
@misc{staven2024mjsim,
  author       = {Staven, Victor M},
  title        = {mj\_sim},
  version      = {1.1.0},
  year         = {2024},
  howpublished = {\url{https://gitlab.sdu.dk/sdurobotics/teaching/mj_sim}},
  note         = {A repository to demonstrate, implement, test and learn control in MuJoCo.},
}
```
