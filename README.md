# TrajDiff

1. Please download each dataset and place it in the `dataset/` directory under the corresponding dataset folder. Then, execute the corresponding `preprocessing_xxx` script to obtain the processed data.

2. python main.py --measure hausdorff --load_checkpoint 1 --gpu 0
   
   We have already placed the pre-trained model in the `/exp/snapshot` directory, which can be loaded using `load_checkpoint`.
