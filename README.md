# Robust Transformer with Locality Inductive Bias and Feature Normalization

This repo is the official implementation of ["Locality iN Locality"](https://arxiv.org/abs/2301.11553).

## Train & Test --- Prepare data
Please refer to the updated notebook [Instructions-best.ipynb](file:///d:/Locality-iN-Locality/Instructions-best.ipynb) for complete details on dataset preparation, optimized training, evaluation, and adversarial attacks (FGSM/PGD). The original notebook is also available at [Instructions.ipynb](file:///d:/Locality-iN-Locality/Instructions.ipynb).

## Trained Model Checkpoints
The repository includes the following trained model checkpoints on the GTSRB dataset:
- **`LNL_Ti_GTSRB_best.pt`**: The best performing LNL-Ti model checkpoint on GTSRB.
- **`model_checkpoint_epoch_5.pt`**: Standard LNL model checkpoint at epoch 5.
- **`model_checkpoint_LNL_MoEx_final_epoch_5.pt`**: LNL-MoEx model checkpoint at epoch 5.

For loading and evaluating these checkpoints, please refer to the evaluation cells in [Instructions-best.ipynb](file:///d:/Locality-iN-Locality/Instructions-best.ipynb).

## Citation
If you find this project useful in your research, please consider cite:
```
@article{manzari2023robust,
  title={Robust transformer with locality inductive bias and feature normalization},
  author={Manzari, Omid Nejati and Kashiani, Hossein and Dehkordi, Hojat Asgarian and Shokouhi, Shahriar B},
  journal={Engineering Science and Technology, an International Journal},
  volume={38},
  pages={101320},
  year={2023},
  publisher={Elsevier}
}
```

## Contact Information

For any inquiries or questions regarding the code, please feel free to contact us directly via email:

- Omid Nejaty: [omid.nejaty@gmail.com](mailto:omid.nejaty@gmail.com)
- Hossein kashiani: [hkashia@clemson.edu](mailto:hkashia@clemson.edu)
