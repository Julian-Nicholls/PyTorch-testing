# Local data

`raw/` contains canonical Kaggle TFRecord shards; `prepared/` contains the local
NPZ representation. Both are ignored by Git. Each NPZ keeps 12 float32 feature
planes, a binary target plane, and a boolean validity plane. See the project
README for provenance and commands.
