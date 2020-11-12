# ITOV

- TOV方程式に関連する計算を行う数値計算コード
- [Sphinx manual](https://applejxd.github.io/ITOV/index.html)

## How to run

```bash
$ pipenv install
$ pipenv run python3 main.py
```

## How to build Sphinx manual

```bash
$ pipenv install
$ pipenv run sphinx-apidoc -f -o ./sphinx .
$ pipenv run sphinx-build -a ./sphinx ./docs
```