# Fake News Detection on the LIAR Dataset

Course project for Deep Learning I (Neural Networks). We train a binary fake-news
classifier on the LIAR dataset and compare several neural models: an MLP on hand-crafted
text features, a BiLSTM on the raw text, a fine-tuned DistilBERT, and a hybrid that combines
the text encoder with speaker metadata. It covers the full pipeline: data preparation, the
models, controlled experiments, evaluation (precision/recall, ROC/PR, calibration), and
error analysis.

## Contents

- `fake_news_detector.ipynb` - the complete project; runs top to bottom with all outputs and plots saved.
- `Project_Report_Task1.pdf` - the written report.
- `requirements.txt` - Python dependencies.

## Running it

The notebook is self-contained: it downloads the LIAR dataset and fixes all random seeds. A
GPU is recommended for the DistilBERT model; the rest runs fine on CPU.

Open `fake_news_detector.ipynb` in Jupyter (or Google Colab with a GPU runtime) and run all
cells. To run locally:

```bash
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
jupyter notebook fake_news_detector.ipynb
```

## Results in brief

On LIAR's short statements the MLP and BiLSTM perform about the same. A fine-tuned DistilBERT
is the strongest text model, and the speaker metadata alone carries most of the predictive
signal. Full numbers and figures are in the notebook and the report.

## Dataset

William Yang Wang. *"Liar, Liar Pants on Fire": A New Benchmark Dataset for Fake News
Detection.* ACL 2017.
