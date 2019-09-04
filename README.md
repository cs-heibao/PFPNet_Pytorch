# PFPNet_Pytorch
The unofficial implementation of ECCV 2018 paper "Parallel Feature Pyramid Network for Object Detection" in Pytorch.

And the paper could be downloaded from "http://openaccess.thecvf.com/content_ECCV_2018/papers/Seung-Wook_Kim_Parallel_Feature_Pyramid_ECCV_2018_paper.pdf"


# Contents
- [1.Environment](#environment)

- [2.Preparation](#preparation)

- [3.Training](#training)

- [4.Demo](#demo)
# Environment
        
        Python3.5
        
        Pytorch1.1.0
        
        GPU:1080Ti
        
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
        
        $VOCdevkit/                           # RootPath  
        
        $VOCdevkit/VOC2012                    # image sets, annotations, etc.  
        
        $VOCdevkit/VOC2012/Annotations                       # include .xml files. 
        
        $VOCdevkit/VOC2012/ImageSets/Main                    # include trainval.txt file.  
        
        $VOCdevkit/VOC2012/JPEGImages                        # include images.  
        
# Training
1. Train your model on PASCAL VOC Format.
                
                cd $PFPNet_Pytorch
                python3 train_PFPNet.py
                
2. Train results, it will create two types file, .pth model and loss log file.
                
                # It will create model definition files and save snapshot models in:
                #   - $PFPNet_Pytorch/weights/PFPNet{input_size}_VOC_{iteration}.pth/
                # and the loss log in:
                #   - $PFPNet_Pytorch/'{}_{}_{}_{}:{}:{}loss.txt'.format(log_time.year, log_time.month, log_time.day,log_time.hour,'%02d'%log_time.minute, '%02d'%log_time.second)
