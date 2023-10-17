# DeepLearning_Final
Predict the popularity of shelter pet photos with Swim Transformer.
## [Kaggle競賽資料集](https://www.kaggle.com/c/petfinder-pawpularity-score)
## TIPs
* pytorch  1.10.0 
* torchvision 0.11.1
* Cuda Memory > 15G (Swim Transformer)
## 分為兩部分
1. Resnet 模型  
   (1) Resnet50 [WANDB](https://wandb.ai/garymarz/PetFinder_my_Pawpularity%20Contest%20resnet50/runs/298tv2k6/overview?workspace=user-garymarz)  
       使用Resnet50進行數值預測，score:20.54167 ranking:2921  
   (2) Resnet18 [WANDB](https://wandb.ai/garymarz/PetFinder_my_Pawpularity%20Contest/runs/vqf5m6hp/overview?workspace=user-garymarz)  
       使用Resnet50進行數值預測，score:20.55673 ranking:2948  
2. [Swim Transformer](https://github.com/microsoft/Swin-Transformer)  
   使用Swim Transformer進行數值預測，score:17.85807 ranking:386
