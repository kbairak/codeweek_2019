# Installation

## Python

To see if you already have Python installed, open a terminal and type
`python -V`:

```sh
➜ python -V
Python 3.7.4
```

If you don't have Python, you have to install it.

If you are on Linux, use your package manager

```sh
➜ sudo apt update -y
➜ sudo apt install python
```

If you are on Windows or MacOSX, go to https://www.python.org/downloads/

## Virtual environments

Virtual environments are a way to isolate the dependencies of a Python project
in your machine so that projects don't interfere with each other. When starting
a new Python project, it's a good practice to create a new virtual environment
and install your dependencies there.

### Installation

Recent versions of Python have the `venv` module which creates virtual
environments. To see if it is available to you, type `python -m venv`:

```sh
➜ python -m venv
usage: venv [-h] [--system-site-packages] [--symlinks | --copies] [--clear]
            [--upgrade] [--without-pip] [--prompt PROMPT]
            ENV_DIR [ENV_DIR ...]
venv: error: the following arguments are required: ENV_DIR
```

If `venv` is not available to you, then you can install the `virtualenv`
package. If you are on Linux, try using your package manager:

```sh
➜ sudo apt install virtualenv
```

Otherwise, download it as a Python dependency:

```sh
➜ sudo pip install virtualenv
```

### Usage

`venv` and `virtualenv` work the same way. To create a virtual environment:

```sh
➜ python -m venv codeweek_venv
```

or

```sh
➜ vitrualenv codeweek_venv
```

These commands will create the `codeweek_venv` folder which will house your
virtual environment.

To "enter" (ie activate) your new virtual environment, run:

```sh
➜ source codeweek_venv/bin/activate
```

Now you are "inside" your virtual environment; any packages you install using
pip will be installed there and will not "contaminate" the rest of your system.

To "exit" (ie deactivate) your new virtual environment, run:

```sh
➜ deactivate
```

## Tensorflow, Jupyter notebook and Matplotplib

While inside your virtual environment, run:

```sh
➜ pip install tensorflow jupyter matplotlib
```

# Running

To start a Jupyter server, run:

```sh
➜ jupyter notebook
```

This will create a server process and open a browser tab to the Jupyter
dashboard.

# Interesting links

- [Very good playlist explaining the basic idea of deep networks and the math](https://www.youtube.com/playlist?list=PLZHQObOWTQDNU6R1_67000Dx_ZCJB-3pi)
- [Youtube channel covering many topics](https://www.youtube.com/channel/UCWN3xxRkmTPmbKwht9FuE5A)
- [Video that shows what a CNN "sees"](https://www.youtube.com/watch?v=AgkfIQ4IGaM)
- [Site with many interesting articles](https://towardsdatascience.com/)
- [Very cool guy](http://karpathy.github.io/)

And of course:

- [What most of today's presentation is based on](https://www.tensorflow.org/tutorials)
