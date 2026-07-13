# Kaggle setup — getting the real datasets

MedChron V1 trains on **RSNA Pneumonia** (classification + ROI boxes) and uses
the **Montgomery + Shenzhen lung masks** for anatomical lung-field ROI. Both are
hosted on Kaggle and need a free Kaggle API token.

## 1. Create a Kaggle account
Sign up at <https://www.kaggle.com> (free). Verify your email/phone if prompted.

## 2. Create an API token
1. Go to <https://www.kaggle.com/settings/account>.
2. Under **API**, click **Create New Token**.
3. This downloads a file called **`kaggle.json`** (your username + key).

## 3. Put the token where the tools expect it
On Windows, place `kaggle.json` here:

```
C:\Users\ADITYA\.kaggle\kaggle.json
```

Create the `.kaggle` folder if it doesn't exist. (Alternatively set the
`KAGGLE_CONFIG_DIR` env var to whatever folder holds the file.)

> ⚠️ Treat `kaggle.json` like a password. It's already covered by `.gitignore`
> (`.env*` / secrets), but never paste its contents anywhere or commit it.

## 4. Download the data
From the project root, with the venv active:

```bash
python ml/scripts/download_data.py --list                 # see what's available
python ml/scripts/download_data.py --dataset rsna_pneumonia
python ml/scripts/download_data.py --dataset montgomery_shenzhen
```

Files land in `ml/data/<dataset>/` (git-ignored).

## 5. Build a train/val/test manifest
If the download produced an `ImageFolder`-style layout
(`<root>/<class_name>/<images>`), build a leakage-safe split:

```bash
python ml/scripts/prepare_data.py --root ml/data/rsna_pneumonia \
    --out ml/data/rsna_pneumonia/manifest.csv
```

If instead it produced a **labels CSV** (common for RSNA), tell me the column
names and I'll wire up the CSV → manifest step (`build_manifest_from_csv`) — the
exact layout depends on which Kaggle mirror you pulled.

## 6. Train
```bash
python ml/scripts/train.py --manifest ml/data/rsna_pneumonia/manifest.csv --backbone vgg16
python ml/scripts/evaluate.py --manifest ml/data/rsna_pneumonia/manifest.csv \
    --checkpoint ml/artifacts/model_vgg16.pt
```

## Datasets that need their own portal (not Kaggle)
- **VinDr-CXR** — PhysioNet credentialed access (training + a data-use agreement).
- **CheXpert** — Stanford registration.

These give the strongest professional scope but require an approval step; RSNA is
the fastest path to a working, explainable model.
