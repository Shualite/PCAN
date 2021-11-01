# Scene Text Image Super-Resolution via Parallelly Contextual Attention Network (ACM Multimedia-2021)
[]()
"[Scene Text Image Super-Resolution via Parallelly Contextual Attention Network](https://doi.org/10.1145/3474085.3475469)" is under peer review at the ACM Multimedia Conference.
The code is built on [TSRN(pytorch)](https://github.com/JasonBoy1/TextZoom) and tested on Ubuntu 18.04 (Pytorch 1.1.0)

## Main Contents
### 1. Introduction
- **Abstract:**
Optical degradation makes text shapes and edges blurred, so the existing scene text recognition methods are difficult to achieve desirable results on low-resolution (LR) scene text images acquired in natural scenes. Therefore, efficiently extracting the sequential information for reconstructing super-resolution (SR) text images is challenging. In this paper, we propose a Parallel Contextual Attention Network (PCAN), which can effectively learn sequence-dependent features and focus more on high-frequency information of the reconstruction in text images. First, we explore the importance of sequence-dependent features in horizontal and vertical directions \emph{parallelly} for text SR, and design a parallel contextual attention block to adaptively select the key information in the text sequence that contributes to image reconstruction. Secondly, we propose a Hierarchically orthogonal texture-aware attention module and an edge guidance loss function, which can help to reconstruct high-frequency information in text images. Finally, we conduct extensive experiments on different test sets of TextZoom, and the results can easily incorporate into mainstream text algorithms to further improve their performance in LR image recognition. Compared with the SR images obtained by BICUBIC up-sampling, our method can respectively improve the recognition accuracy of ASTER, MORAN, and CRNN by 14.29\%, 14.35\%, and 20.60\%. Besides, our PCAN outperforms 8 state-of-the-art (SOTA) SR methods in improving the recognition performance of LR images. Most importantly, it outperforms the current optimal text-orient SR method TSRN by 3.19\%, 3.65\%, and 6.0\% on the recognition accuracy of ASTER, MORAN, and CRNN respectively.


### 2. Train
#### Prepare training datasets
- 1. Download the **TextZoom** dataset (1.7w+ LR-HR pair images) from the link [TextZoom](https://drive.google.com/drive/folders/1WRVy-fC_KrembPkaI68uqQ9wyaptibMh?usp=sharing).
- 2. Set '--dataset/lmdb/str/TextZoom' as the HR and LR image path.
- 3. download the Aster model from https://github.com/ayumiymk/aster.pytorch, Moran model from https://github.com/Canjie-Luo/MORAN_v2, CRNN model from https://github.com/meijieru/crnn.pytorch.
- 4. Set '--pth/crnn.pth', '--pth/demo.pth.tar', '--pth/moran.pth' as the file path of ocr metrics.


#### training
- 1. Change your own yaml file under 'src/config/all/own.yaml'
- 2. Run the following code.

`CUDA_VISIBLE_DEVICES=1 python3 main.py --STN --mask --edge --config 'all/own.yaml'`

### 3. Citation
If the the work or the code is helpful, please cite the following papers:

> @inproceedings{10.1145/3474085.3475469,
> 
> title = {Scene Text Image Super-Resolution via Parallelly Contextual Attention Network},
  author = {Zhao, Cairong and Feng, Shuyang and Zhao, Brian Nlong and Ding, Zhijun and Wu, Jun and Shen, Fumin and Shen, Heng Tao},
  booktitle = {Proceedings of the 29th ACM International Conference on Multimedia},
  pages = {2908–2917},
  year = {2021}
}


### 4. Acknowledge
The code is built on [TextZoom (Pytorch)](https://github.com/JasonBoy1/TextZoom). We thank the authors for sharing the codes.
