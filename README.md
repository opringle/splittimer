
## Notes

- [x] download youtube videos and create X,Y pairs containing preprocessed frames and labelled splits
- [x] compute resnet50 features for all clips and store on disk
- [ ] try simply finding the frame with the highest cosine similarity
- [ ] efficiently generate training data from splits
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

Download 2 youtube videos of Loudenvielle DH runs (vali holl & amaury pierron) and preprocess them into 5s clips for ML

```bash
python youtube_preprocess.py https://www.youtube.com/watch?v=jUfJyZFpAoY&t=63s&ab_channel=GoProBike --split-times 39.0 --keep-video
```

```bash
python youtube_preprocess.py https://www.youtube.com/watch?v=AClbgHAvAZ4&ab_channel=GoProBike --split-times 43.0 --keep-video
```

Project each clip to vector representation (using pretrained image model) and save them to disk

```bash
python extract_clip_features.py processed_clips/gopro_amaury_pierron_takes_2nd_and_overall_points_lead__loudenvielle__25_uci_dh_mtb_world_cup/ --batch-size 16
```

```bash
python extract_clip_features.py processed_clips/gopro_vali_holl_takes_2nd_place__loudenvielle__25_uci_dh_mtb_world_cup/ --batch-size 16
```

Use cosine similarity to predict the split frame in the target video (Vali's run). Compute eval metrics and display predictions.

```bash
python find_splits.py processed_clips/gopro_amaury_pierron_takes_2nd_and_overall_points_lead__loudenvielle__25_uci_dh_mtb_world_cup processed_clips/gopro_vali_holl_takes_2nd_place__loudenvielle__25_uci_dh_mtb_world_cup
```
