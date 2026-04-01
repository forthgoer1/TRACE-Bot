# TRACE-Bot

TRACE-Bot is a unified dual-channel framework capable of jointly modeling semantic artifacts and behavioral patterns enhanced by AI-Generated Content (AIGC). TRACE-Bot constructs fine-grained representations from heterogeneous sources, including personal information data, interaction behavior data, and tweet data. The framework employs a dual-channel architecture: one channel captures linguistic artifacts via a pretrained language model, while the other captures behavioral irregularities through multidimensional activity features, augmented by signals from state-of-the-art AIGC detectors. Subsequently, the fused representations are classified via a lightweight prediction head.

![项目截图](https://raw.githubusercontent.com/forthgoer1/TRACE-Bot/main/model.svg)

The Data Preparation module handles data collection and preprocessing to ensure structured inputs. The Feature Processing module extracts multidimensional textual and behavioral features, capturing semantic traces and interaction patterns. Subsequently, the Feature Fusion module integrates these heterogeneous representations to enhance discriminative expressiveness. Finally, the Detection module utilizes the fused vectors for precise identification and classification of LLM-driven social bots.

## Project Structure

```
TRACE-Bot/
├── src/              # Source code directory
│   ├── data_process.py         # Data processing and cleaning
│   ├── behavior_sequence.py    # Behavior sequence extraction
│   ├── GLTR_detection.py       # GLTR model detection
│   ├── fast_detectgpt.py        # Fast DetectGPT model detection
│   ├── feature_integration.py  # Feature integration
│   └── fusion_detection.py       # Feature fusion and model detection
├── README.md         # Project documentation
└── requirements.txt  # Dependencies
```

## Datasets

### Fox8-23
> https://zenodo.org/records/8035290

### BotSim-24
> https://github.com/QQQQQQBY/BotSim/tree/main/BotSim-24-Dataset


## Functional Modules

### 1. Data Processing and Cleaning (`src/data_process.py`)
- Processes NDJSON format data
- Converts data to CSV format
- Flattens nested JSON structures
- Performs data cleaning and preprocessing

### 2. Behavior Sequence Extraction (`src/behavior_sequence.py`)
- Extracts user tweet types (original, retweet, reply)
- Constructs behavior sequences
- Calculates features such as sequence compression ratio

### 3. AIGC Detection - GLTR (`src/GLTR_detection.py`)
- Calculates text probabilities using BERT and GPT-2 models
- Generates GLTR detection scores as features

### 4. AIGC Detection - Fast DetectGPT (`src/fast_detectgpt.py`)
- Detects text using the Fast DetectGPT model
- Generates detection scores as features

### 5. Feature Integration (`src/feature_integration.py`)
- Extracts user personal information features
- Integrates all features into a single feature data file

### 6. Feature Fusion and Model Detection (`src/fusion_detection.py`)
- Uses GPT-2 as the text semantic encoder
- Fuses behavioral features and text features
- Trains and evaluates the social bot detection model

## Workflow

1. **Data Processing**: Run `src/data_process.py` to process raw data.
2. **Behavior Sequence Extraction**: Run `src/behavior_sequence.py` to extract behavior sequence features.
3. **AIGC Detection**: Run `src/GLTR_detection.py` and `src/fast_detectgpt.py` separately to generate AIGC detection features.
4. **Feature Integration**: Run `src/feature_integration.py` to integrate all features.
5. **Model Training and Detection**: Run `src/fusion_detection.py` (Note: corrected from `feature_fusion.py` in original text to match file structure) to train the model and perform detection.

## Dependencies

- Python 3.8+
- pandas
- numpy
- torch
- transformers
- scikit-learn
- tqdm

## Notes

- Running AIGC detection models requires significant computational resources; execution on a GPU environment is recommended.
- Required pretrained models will be automatically downloaded upon the first run.
- Data processing and feature extraction steps may take considerable time, depending on the data scale.
