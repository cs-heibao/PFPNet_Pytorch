# PFPNet_Pytorch
The unofficial implementation of ECCV 2018 paper "Parallel Feature Pyramid Network for Object Detection" in Pytorch.

And the paper could be downloaded from "http://openaccess.thecvf.com/content_ECCV_2018/papers/Seung-Wook_Kim_Parallel_Feature_Pyramid_ECCV_2018_paper.pdf"


# Contents
- [1.Preparation](#preparation)

- [2.Training](#training)

- [3.Demo](#demo)

# Preparation
1. Get the code. We will call the cloned directory as `$PFPNet_Pytorch`.  

    `https://github.com/junjieAI/PFPNet_Pytorch.git`  

2. Build the Cython modules, [ We can refer to Faster-Rcnn approach](https://github.com/rbgirshick/py-faster-rcnn).  
        ```  
        cd $PFPNet_Pytorch/lib  
        ```  
        ```  
        make  
        ```  
3. Download the [basenet model VGGNET.](https://s3.amazonaws.com/amdegroot-models/vgg16_reducedfc.pth) By default, we assume the model is stored in `$PFPNet_Pytorch/pretrained`.  
  
4. Download the [trained model](https://pan.baidu.com/s/1aa-Mar-DRESuihU3wbOgVQ) of myself, the Extract code: fh9uby. By default, we assume the model is stored in `$PFPNet_Pytorch/pretrained`.  
  
5. Prepare the data basic structure.   
        ```
        11  
        22  
        33 
        ```

