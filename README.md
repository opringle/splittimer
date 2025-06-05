
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

Download youtube videos based on config file

```bash
python youtube_download.py --config video_config.yaml
```

Generate generalized training data for identifying splits

```bash
python generate_training_samples.py --config video_config.yaml --max_negatives_per_positive 10 --num_augmented_positives_per_segment 50 --log-level DEBUG
```

Inspect the training data

```bash
python inspect_training_data.py training_data/training_metadata.csv --num_samples 3 && open ./training_data_inspection/index.html
```

Preprocess videos into training samples and save to disk

```bash
python preprocess_videos_into_samples.py
```

Train and evaluate a model on the data

```bash
python ____.py
```