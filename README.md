# CaLa (SIGIR 2024)

### CaLa: Complementary Association Learning for Augmenting Composed Image Retrieval
[![PWC]]([https://paperswithcode.com/sota/nlp-based-person-retrival-on-cuhk-pedes?p=towards-unified-text-based-person-retrieval-a](https://paperswithcode.com/paper/cala-complementary-association-learning-for))

This is the **official repository** for the [**paper**]([https://arxiv.org/abs/2308.11485]) "*CaLa: Complementary Association Learning for Augmenting Composed Image Retrieval*".

### Prerequisites
	
The following commands will create a local Anaconda environment with the necessary packages installed.

```bash
conda create -n cala -y python=3.8
conda activate cala
conda install -y -c pytorch pytorch=1.11.0 torchvision=0.12.0
conda install -y -c anaconda pandas=1.4.2
pip install comet-ml==3.21.0
pip install git+https://github.com/openai/CLIP.git
pip install salesforce-lavis
```

### Data Preparation

To properly work with the codebase FashionIQ and CIRR datasets should have the following structure:

```
project_base_path
└───  CaLa
      └─── src
            | blip_fine_tune.py
            | data_utils.py
            | utils.py
            | ...

└───  fashionIQ_dataset
      └─── captions
            | cap.dress.test.json
            | cap.dress.train.json
            | cap.dress.val.json
            | ...
            
      └───  images
            | B00006M009.jpg
            | B00006M00B.jpg
            | B00006M6IH.jpg
            | ...
            
      └─── image_splits
            | split.dress.test.json
            | split.dress.train.json
            | split.dress.val.json
            | ...

└───  cirr_dataset  
       └─── train
            └─── 0
                | train-10108-0-img0.png
                | train-10108-0-img1.png
                | train-10108-1-img0.png
                | ...
                
            └─── 1
                | train-10056-0-img0.png
                | train-10056-0-img1.png
                | train-10056-1-img0.png
                | ...
                
            ...
            
       └─── dev
            | dev-0-0-img0.png
            | dev-0-0-img1.png
            | dev-0-1-img0.png
            | ...
       
       └─── test1
            | test1-0-0-img0.png
            | test1-0-0-img1.png
            | test1-0-1-img0.png 
            | ...
       
       └─── cirr
            └─── captions
                | cap.rc2.test1.json
                | cap.rc2.train.json
                | cap.rc2.val.json
                
            └─── image_splits
                | split.rc2.test1.json
                | split.rc2.train.json
                | split.rc2.val.json
```


# Adjustments for dependencies

For finetuning blip2 encoderds, you need to comment out this code in lavis within your conda enviroment.
```python
# In lavis/models/blip2_models/blip2_qformer.py line 367
# @torch.no_grad() # commemt out this line.
```
Comment out this code to calculate the gradient of the blip2-model to update the parameters.

For finetuning clip encoders, you need to replace with these codes in the clip packages, thus RN50x4 features can interact with Qformers.
```python
# Replace CLIP/clip/models.py line 152-154 with the following codes.
152#    x = self.attnpool(x)
153#	
154#    return x

152#	y=x 
153#	x = self.attepool(x)
154#
155#	return x,y

```

### Training


```shell
# cala finetune 
CUDA_VISIBLE_DEVICES='GPU_IDs' python src/blip_fine_tune.py --dataset {'CIRR' or 'FashionIQ'} \
	--num-epochs 30 --batch-size 64 \
	--max-epoch 15 --min-lr 0 \
	--learning-rate 5e-6 \
	--transform targetpad --target-ratio 1.25 \
	--save-training --save-best --validation-frequency 1 \
	--encoder {'both' or 'text' or 'multi'} \
	--encoder-arch {clip or blip2} \
	--cir-frame {sum or artemis} \
	--tac-weight 0.45 \
	--hca-weight 0.1 \
	--embeds-dim {640 for clip and 768 for blip2} \
	--model-name {RN50x4 for clip and None for blip} \
	--api-key {Comet-api-key} \
	--workspace {Comet-workspace} \
	--experiment-name {Comet-experiment-name} \
```


### CIRR Testing


```shell
CUDA_VISIBLE_DEVICES='GPU_IDs' python src/cirr_test_submission_blip2.py --submission-name {cirr_submission} \
	--combining-function {sum or artemis} \
	--blip2-textual-path {saved_blip2_textual.pt} \
	--blip2-multimodal-path {saved_blip2_multimodal.pt} \
	--blip2-visual-path {saved_blip2_visual.pt} 

```

```shell
python src/validate.py 
   	--dataset {'CIRR' or 'FashionIQ'} \
   	--combining-function {'combiner' or 'sum'} \
   	--combiner-path {path to trained Combiner} \
   	--projection-dim 2560 \
	--hidden-dim 5120 \
   	--clip-model-name RN50x4 \
   	--clip-model-path {path-to-fine-tuned-CLIP} \
   	--target-ratio 1.25 \
   	--transform targetpad
```
