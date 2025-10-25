# Dataset Information

## TS2Vec-Ensemble ETT Datasets

The Electricity Transformer Temperature (ETT) datasets are publicly available collections of transformer data originally introduced by Zhou et al. (2021). The data cover two years (July 2016–July 2018) of measurements from electrical transformers in two separate regions of China.

### Dataset Overview

All three subsets (ETTh1, ETTh2, ETTm1) share a common format: each record includes a timestamp, six "load" features, and the transformer oil temperature (OT) as the target variable.

### Dataset Sources

- **Official Repository**: [https://github.com/zhouhaoyi/ETDataset](https://github.com/zhouhaoyi/ETDataset) (Zhou et al. 2021)
- **Google Drive Mirror**: Available from STAug repository

### Dataset Descriptions

#### ETTh1
- **Sampling Rate**: Hourly data (1-hour sampling)
- **Region**: Region 1
- **Duration**: 2 years of readings (July 2016–July 2018)
- **Target Variable**: Oil temperature (OT)
- **Input Features**: Six transformer load measurements (HUFL, HULL, MUFL, MULL, LUFL, LULL)

#### ETTh2
- **Sampling Rate**: Hourly data (1-hour sampling)  
- **Region**: Region 2
- **Duration**: 2 years of readings (July 2016–July 2018)
- **Target Variable**: Oil temperature (OT)
- **Input Features**: Six transformer load measurements (HUFL, HULL, MUFL, MULL, LUFL, LULL)
- **Note**: Structurally identical to ETTh1 but from a different transformer location

#### ETTm1
- **Sampling Rate**: 15-minute data (every 15 minutes)
- **Region**: Region 1
- **Duration**: 2 years of readings (July 2016–July 2018)
- **Target Variable**: Oil temperature (OT)
- **Input Features**: Six transformer load measurements (HUFL, HULL, MUFL, MULL, LUFL, LULL)

### Data Format

Each dataset is provided in CSV format with the following structure:
- **First Column**: Timestamp (date/time)
- **Columns 2-7**: Six load features (HUFL, HULL, MUFL, MULL, LUFL, LULL)
- **Last Column**: Target oil temperature (OT)

Example column names: `date`, `HUFL`, `HULL`, `MUFL`, `MULL`, `LUFL`, `LULL`, `OT`

### Usage in TS2Vec-Ensemble

In our TS2Vec-Ensemble experiments, we treat each ETT subset as a univariate forecasting problem focusing on the oil temperature series. The approach involves:

- **Forecasting Target**: Transformer oil temperature (OT) over time
- **Feature Usage**: Model learns from OT values, ignoring other features for the forecasting task
- **Forecasting Horizon**: Set according to standard practice (e.g., predict the next 24–720 hours)

#### Data Splitting

The datasets are split chronologically as follows:
- **Training**: 60% (approximately 12 months)
- **Validation**: 20% (approximately 4 months)  
- **Testing**: 20% (approximately 4 months)

#### Model Training

The TS2Vec-Ensemble model is trained in a self-supervised fashion on the training portion of OT values and evaluated on its ability to forecast future OT values.

### Data Access Instructions

1. **Download from GitHub**: Visit the official ETT datasets repository at [zhouhaoyi/ETDataset](https://github.com/zhouhaoyi/ETDataset)
2. **Download from Google Drive**: Use the mirror link from the STAug repository
3. **File Placement**: Place the downloaded CSV files under `ETT-small/` in your local dataset directory
4. **File Names**: Look for `ETTh1.csv`, `ETTh2.csv`, and `ETTm1.csv`

### Citation

When using these datasets, please cite the original ETT source:

```bibtex
@inproceedings{zhou2021informer,
  title={Informer: Beyond Efficient Transformer for Long Sequence Time-series Forecasting},
  author={Zhou, Haoyi and Zhang, Shanghang and Peng, Jieqi and Zhang, Shuai and Li, Jianxin and Xiong, Hui and Zhang, Wancai},
  booktitle={AAAI},
  year={2021}
}
```

### Additional Resources

- **Paper**: [Informer: Beyond Efficient Transformer for Long Sequence Time-Series Forecasting](https://arxiv.org/abs/2012.07436)
- **ArXiv**: [2012.07436] Informer: Beyond Efficient Transformer for Long Sequence Time-Series Forecasting
- **GitHub**: [STAug Repository](https://github.com/xiyuanzh/STAug) - Contains Google Drive mirror links

---

**Note**: All datasets are publicly available for research purposes. Ensure proper attribution when using these datasets in your research.