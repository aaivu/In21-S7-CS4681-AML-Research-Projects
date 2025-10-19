# ADE20K Dataset

The ADE20K dataset is a large-scale dataset for semantic segmentation, scene parsing, and object recognition tasks. It contains diverse images covering various scenes and objects, annotated at the pixel level.

**Dataset Path in Project:**

```python
ade20k_path = kagglehub.dataset_download("awsaf49/ade20k-dataset")
ade20k_path = os.path.join(ade20k_path, "ADEChallengeData2016")
```

- **Source:** [Kaggle: awsaf49/ADE20K-dataset](https://www.kaggle.com/datasets/awsaf49/ade20k-dataset)
- **Structure:** The folder `ADEChallengeData2016` contains images and corresponding annotations used for semantic segmentation.
- **Use Case:** Suitable for training and evaluating segmentation models such as SegFormer, U-Net, or DeepLab.
- **Notes:** Images are labeled at the pixel level with semantic categories, enabling fine-grained scene understanding.
