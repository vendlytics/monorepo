# Vendlytics

This is Vendlytics' monorepo. More detail project descriptions and setup instructions are in `src/`.

## Pipenv

`Pipenv` manages our deps for us.

```shell
pip3 install pipenv
python3.6 -m pipenv shell
pip install -r requirements
```

## Linting

Python autolint with:

```shell
autopep8 --in-place --aggressive --aggressive **/*.py
```