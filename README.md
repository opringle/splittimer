
## Notes

- [x] download youtube videos with split data included
- [ ] compute resnet50 features for all clips and store on disk
- [ ] generate training data from image features and splits
- [ ] train an MLP and compute F1 score per label

## Prerequisites

- install pyenv
- configure python environment

```bash
pyenv install 3.12

pyenv virtualenv 3.12 splittimer

pyenv local splittimer

pip install -r ./requirements.txt
```

- install ffmpeg

```bash
brew install ffmpeg
```

## Preprocess GoPro runs

Download 2 youtube videos of Loudenvielle DH runs

```bash
python youtube_preprocess.py https://www.youtube.com/watch?v=jUfJyZFpAoY&t=63s&ab_channel=GoProBike --split-times 38.0 --keep-video
```

```bash
python youtube_preprocess.py https://www.youtube.com/watch?v=AClbgHAvAZ4&ab_channel=GoProBike --split-times 42.0 --keep-video
```

Select a random frame from Amaury's run, then search for the most similar frame in Vali's run

```bash
python resnet_dot_product_test.py processed_clips/gopro_amaury_pierron_takes_2nd_and_overall_points_lead__loudenvielle__25_uci_dh_mtb_world_cup processed_clips/gopro_vali_holl_takes_2nd_place__loudenvielle__25_uci_dh_mtb_world_cup
```
