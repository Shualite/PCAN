# Scene Text Image Super-Resolution via Parallel Contextual Attention Network (ACM Multimedia-2021)
[]()
"[Scene Text Image Super-Resolution via Parallel Contextual Attention Network]()" is under peer review at the ACM Multimedia Conference.
The code is built on [TSRN(pytorch)](https://github.com/JasonBoy1/TextZoom) and tested on Ubuntu 18.04 (Pytorch 1.1.0)

## Main Contents
### 1. Introduction
- **Abstract:**
Optical degradation makes text shapes and edges blurred, so the existing scene text recognition methods are difficult to achieve desirable results on low-resolution (LR) scene text images acquired in natural scenes. Therefore, efficiently extracting the sequential information for reconstructing super-resolution (SR) text images is challenging. In this paper, we propose a Parallel Contextual Attention Network (PCAN), which can effectively learn sequence-dependent features and focus more on high-frequency information of the reconstruction in text images. First, we explore the importance of sequence-dependent features in horizontal and vertical directions \emph{parallelly} for text SR, and design a parallel contextual attention block to adaptively select the key information in the text sequence that contributes to image reconstruction. Secondly, we propose a Hierarchically orthogonal texture-aware attention module and an edge guidance loss function, which can help to reconstruct high-frequency information in text images. Finally, we conduct extensive experiments on different test sets of TextZoom, and the results can easily incorporate into mainstream text algorithms to further improve their performance in LR image recognition. Compared with the SR images obtained by BICUBIC up-sampling, our method can respectively improve the recognition accuracy of ASTER, MORAN, and CRNN by 14.29\%, 14.35\%, and 20.60\%. Besides, our PCAN outperforms 8 state-of-the-art (SOTA) SR methods in improving the recognition performance of LR images. Most importantly, it outperforms the current optimal text-orient SR method TSRN by 3.19\%, 3.65\%, and 6.0\% on the recognition accuracy of ASTER, MORAN, and CRNN respectively.


### 2. Train code
#### Prepare training datasets
- 1. Download the **TextZoom** dataset (1.7w+ LR-HR pair images) from the link [TextZoom](https://drive.google.com/drive/folders/1WRVy-fC_KrembPkaI68uqQ9wyaptibMh?usp=sharing).
- 2. Set '--dataset/lmdb/str/TextZoom' as the HR and LR image path.

<!-- #### Train the model
- You can retrain the model: 
  - 1. CD to 'TrainCode/code'; 
  - 2. Run the following scripts to train the models:
   -->
> 
>

<!-- ### 3. Test code
-  1. You can [Download the pretrained model first](https://pan.baidu.com/s/1aTYG4Wy72MI-gCRGnJgkvQ), password: eq1v
-  2. CD to 'TestCode/code', run the following scripts
> 
>  python main.py  --model san  --data_test MyImage  --save `save_name`  --scale 2 --n_resgroups 20 --n_resblocks 10 --n_feats 64 --reset --chop --save_results --test_only --testpath 'your path' --testset Set5  --pre_train ../model/SAN_BIX2.pt   -->
> 
### 4. Results
<!-- - Some of [the test results can be downloaded.](https://pan.baidu.com/s/1j0ZgfbGKyYZqsSCLOb3nUg)  Password:w3da -->

### 5. Citation
<!-- If the the work or the code is helpful, please cite the following papers

> @inproceedings{dai2019second,
> 
> title={Scene Text Image Super-Resolution via Parallel Contextual Attention Network},
  author={Dai, Tao and Cai, Jianrui and Zhang, Yongbing and Xia, Shu-Tao and Zhang, Lei},
  booktitle={Proceedings of the IEEE Conference on Computer Vision and Pattern Recognition},
  pages={11065--11074},
  year={2019}
}

> @inproceedings{zhang2018image,
> 
  title={Image super-resolution using very deep residual channel attention networks},
  author={Zhang, Yulun and Li, Kunpeng and Li, Kai and Wang, Lichen and Zhong, Bineng and Fu, Yun},
  booktitle={Proceedings of the European Conference on Computer Vision (ECCV)},
  pages={286--301},
  year={2018}
}

> @inproceedings{li2017second,
>  title={Is second-order information helpful for large-scale visual recognition?},
  author={Li, Peihua and Xie, Jiangtao and Wang, Qilong and Zuo, Wangmeng},
  booktitle={Proceedings of the IEEE International Conference on Computer Vision},
  pages={2070--2078},
  year={2017}
} -->

### 6. Acknowledge
The code is built on [TextZoom (Pytorch)](https://github.com/JasonBoy1/TextZoom). We thank the authors for sharing the codes.

## Code will be comming soon!