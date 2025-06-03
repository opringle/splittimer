
## Prerequisites

```bash
pyenv install 3.12

pyenv virtualenv 3.12 splittimer

pyenv local splittimer

pip install -r ./requirements.txt
```

## Preprocess GoPro runs

```bash
python youtube_preprocess.py https://www.youtube.com/watch?v=jUfJyZFpAoY&t=63s&ab_channel=GoProBike
```