# Challenge

Code for the artistic style image generation challenge. The project trains one LoRA for each of eight style buckets and generates test images from the test captions, styles, and predicted dimensions.

## Contents

- [Repository Layout](#repository-layout)
- [Installation](#installation)
- [Data Preparation](#data-preparation)
- [Training](#training)
- [Checkpoints](#checkpoints)
- [Reproduce Test Results](#reproduce-test-results)

## Repository Layout

```text
Challenge/
├── configs/
│   └── ratio.json                  
├── dataset/
│   ├── train/
│   │   ├── dataset/<style>/        
│   │   └── text/
│   │       └── train_<bucket>.jsonl 
│   └── test/
│       ├── raw/
│       │   └── captions.json       
│       └── test.jsonl              
├── checkpoints/                   
├── results/                        
├── scripts/
│   ├── train.sh                    
│   ├── train_multi.sh              
│   └── generate_all.sh             
├── tools/
│   ├── common.py                   
│   └── generate.py                 
├── model/
│   └── Zimage/                     
├── requirements.txt                
└── README.md
```

## Installation

Run the following commands from the `Challenge` root directory.

### Python environment

```bash
conda create -n challenge python=3.10 -y
conda activate challenge
pip install -r requirements.txt
```
### DiffSynth-Studio

Clone DiffSynth-Studio from the official repository:

```bash
git clone https://github.com/modelscope/DiffSynth-Studio.git ./diffsynth-studio
pip install -e ./diffsynth-studio
```

### Z-Image

The Z-Image base weights come from [Tongyi-MAI/Z-Image](https://huggingface.co/Tongyi-MAI/Z-Image):

```bash
huggingface-cli download Tongyi-MAI/Z-Image \
  --local-dir ./model/Zimage
```

## Data Preparation

The training data comes from [EmoArt-130k](https://huggingface.co/datasets/printblue/EmoArt-130k). The local images and annotations are organized by style bucket:

```text
./dataset/train/
├── dataset/
│   ├── Abstract Art/
│   ├── Abstract Expressionism/
│   ├── Baroque/
│   ├── China_images/
│   ├── Early Renaissance/
│   ├── Gongbi/
│   ├── High Renaissance/
│   ├── Impressionism/
│   ├── Ink and wash painting/
│   ├── Mannerism (Late Renaissance)/
│   ├── Socialist Realism/
│   └── Ukiyo-e/
└── text/
    ├── train_abstract.jsonl
    ├── train_baroque.jsonl
    ├── train_gongbi.jsonl
    ├── train_impressionism.jsonl
    ├── train_inkwash.jsonl
    ├── train_renaissance.jsonl
    ├── train_sovrealism.jsonl
    └── train_ukiyoe.jsonl
```

The original challenge captions are stored in `./dataset/test/raw/captions.json`. Before generation, use `./configs/ratio.json` to predict the aspect ratio, `width`, and `height` for each test item, then save the results to `./dataset/test/test.jsonl`. Generation reads this file, whose rows contain:

```json
{"sample_id": "track1_0001", "caption": "...", "bucket": "ukiyoe", "predicted_aspect_ratio": "4:3", "width": 688, "height": 512}
```

## Training

Train one style on one GPU:

```bash
bash ./scripts/train.sh baroque 0
```

Train one style on multiple GPUs:

```bash
bash ./scripts/train.sh baroque 0,1,2
```

The first argument is the style bucket and the second argument is a comma-separated GPU list. The available buckets are `abstract`, `baroque`, `gongbi`, `impressionism`, `inkwash`, `renaissance`, `sovrealism`, and `ukiyoe`.

Schedule multiple styles:

```bash
bash ./scripts/train_multi.sh gongbi:0 ukiyoe:1 baroque:2,3
```
Common settings can be changed with environment variables:

```bash
NUM_EPOCHS=3 \
TRAIN_IMAGE_ROOT=./dataset/train/dataset \
bash ./scripts/train.sh baroque 0
```

## Checkpoints

LoRA checkpoints are saved under the corresponding style directory:

```text
./checkpoints/
├── abstract/
├── baroque/
├── gongbi/
├── impressionism/
├── inkwash/
├── renaissance/
├── sovrealism/
└── ukiyoe/
```

## Reproduce Test Results

After all eight style checkpoints are ready, generate the complete test set with a chosen number of GPUs:

```bash
bash ./scripts/generate_all.sh --num-gpus 4
```

You can also specify GPU IDs directly:

```bash
bash ./scripts/generate_all.sh --devices 1,3,5
```

GPU 0 is used by default. Generated images are saved to:

```text
./results/images/
```

To generate one style directly:

```bash
python ./tools/generate.py \
  --test-data ./dataset/test/test.jsonl \
  --checkpoint-dir ./checkpoints \
  --out-dir ./results \
  --devices 0 \
  --buckets inkwash
```
