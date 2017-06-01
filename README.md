## Real-Time Multi-Person 2D Pose Estimation with Temporal Entity Alignment
Based on the PyTorch implementation of the pose estimation algorithm, now with
temporal entity tracking/alignment. This pipeline is only built for inference
and not for training, but may consider the possibility to provide a training interface.

### Source Implementations
* Caffe: https://github.com/ZheC/Realtime_Multi-Person_Pose_Estimation
* Pytorch: https://github.com/tensorboy/pytorch_Realtime_Multi-Person_Pose_Estimation

### References
    @InProceedings{cao2017realtime,
        title = {Realtime Multi-Person 2D Pose Estimation using Part Affinity Fields},
        author = {Zhe Cao and Tomas Simon and Shih-En Wei and Yaser Sheikh},
        booktitle = {The IEEE Conference on Computer Vision and Pattern Recognition (CVPR)},
        year = {2017}
    }

    @inproceedings{wei2016cpm,
        author = {Shih-En Wei and Varun Ramakrishna and Takeo Kanade and Yaser Sheikh},
        booktitle = {CVPR},
        title = {Convolutional pose machines},
        year = {2016}
    }
