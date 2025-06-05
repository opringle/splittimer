
## Notes

- [x] download youtube videos and create X,Y pairs containing preprocessed clips and labelled splits
- [x] compute resnet50 features for all clips and store on disk
- [x] try simply finding the frame with the highest cosine similarity - doesn't work well
- [ ] intelligently generate training data from splits https://grok.com/share/bGVnYWN5_b89bd234-5e76-4ca3-9ef3-3dd2c1054768 
- [ ] train a dedicated model
- [ ] refactor the code to allow different model types (resnet cosine similarity, mlp on resnet features etc)
- [ ] keep refining until accuracy is 100% on unseen videos

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

Download 2 youtube videos of Loudenvielle DH runs (vali holl & amaury pierron) and preprocess them into 5s clips for ML.

```bash
python youtube_preprocess.py --config video_config.yaml --keep-video
```

Compute vector representations of each clip (using pretrained resnet 50 model) and save to disk

```bash
python extract_clip_features.py processed_clips --batch-size 32
```

Generate training data for dedicated model

```bash
python generate_training_samples.py processed_clips --log-level DEBUG
```

Inspect the training samples

```bash
python inspect_training_data.py training_data/training_metadata_loudenvielle_2025_amaury_pierron_vali_holl.npz --num_positive 5 --num_negative 5
```

Use cosine similarity to predict the split frame in the target video (Vali's run). Compute eval metrics and display predictions.

```bash
python find_splits.py processed_clips/gopro_amaury_pierron_takes_2nd_and_overall_points_lead__loudenvielle__25_uci_dh_mtb_world_cup processed_clips/gopro_vali_holl_takes_2nd_place__loudenvielle__25_uci_dh_mtb_world_cup
```