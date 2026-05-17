# What is your best single model LB score ?
URL: https://www.kaggle.com/competitions/birdclef-2026/discussion/683791
Total messages: 114
================================================================================

## ORIGINAL POST

--- Antoine Masq [CONTRIBUTOR] rank=63 votes=41 date=2026-03-22
Hi everyone,

As most of us are probably still improving our baseline models, I was curious to see what your best LB score is for a single model without using the unlabeled data.

So far, my best single model performance is 0.889 with an EfficientNetv2_b0 backbone


## COMMENTS

--- Salman Ahmed [EXPERT] rank=16 votes=12 date=2026-04-11
As of April 11. 
Updated a couple of things in my train settings.

Trained on 20 sec with Train Audio + Labelled Soundscapes.

No KD. No Claude for now. Just wanted to solve it / write it myself after a longtime relying on claude. [Just for the fun of it].

No Prior or additional features.

Ensemble of 4 scores 0.932.

Individual scores around 0.922.

10 seconds / 15 seconds scores almost the same for me / sometimes better [Didn't include those in my ensemble].

Will play with some cool attention for next week. 

  --- Jack [EXPERT] rank=106 votes= date=2026-04-11
  Impressive stuff! 

  --- Antoine Masq [CONTRIBUTOR] rank=63 votes= date=2026-04-11
  So this score is still without using the unlabeled soundscapes ? If so, that's super impressive!

  --- Gaurav Rawat [MASTER] rank=420 votes= date=2026-04-11
  Awesome dunno getting single models similar but infer too slow for me :/

  --- XuKong Ji [EXPERT] rank=1402 votes= date=2026-04-12
  That's impressive! 
  If you don't mind, could you please share some details about the data processing pipeline?
  Specifically, do you directly convert the entire 15-second audio clip into a mel spectrogram and feed it into the model for prediction and training? Or do you split the 15-second spectrogram into three 5-second segments, convert each segment into a mel spectrogram separately, perform model inference on each one individually, and then aggregate the three results?

    --- Salman Ahmed [EXPERT] rank=16 votes= date=2026-04-12
    xD. Yes it's trained on 20 seconds audio segments. Not the split and max logits. 

    --- Mahog [EXPERT] rank=1373 votes= date=2026-04-14
    Does training on longer audio segments increase training time by a lot for you? 20 second segments should in theory increase training time by 4x right?

  --- !!!!! [EXPERT] rank=888 votes= date=2026-04-12
  If you don’t mind me asking, what was the runtime for your submission?

    --- Salman Ahmed [EXPERT] rank=16 votes= date=2026-04-12
    Around 50 to 60 minutes I think. 

  --- Leon [GRANDMASTER] rank=1415 votes= date=2026-04-12
  Awesome! Is it still the efficientnetb0 backbone?

    --- Salman Ahmed [EXPERT] rank=16 votes=1 date=2026-04-13
    Yes. Solo B0 was around 0.922. 

    --- Jiacheng Ma [CONTRIBUTOR] rank=1195 votes= date=2026-04-17
    Great！It's still efficientnet_b0 or efficientnet_v2b0?

  --- Starry [MASTER] rank=1 votes= date=2026-04-13
  Hi Salman~ Nice to see you in BirdClef again. I wonder do you use external data in your current pipeline?

    --- Salman Ahmed [EXPERT] rank=16 votes=2 date=2026-04-13
    The data I am using is used in solutions for previous competitions. 
    
    I have a question for host @stefankahl ; Do we need to share that again? If yes, then where? and by what time / day?
    
    for example; no call / Background noise was used by 2nd place solution in 2023 [Background Noise(nocall in 2020, 2021 comp + rainforest + environment sound + nocall in freefield1010, warblrb, birdvox)] [https://www.kaggle.com/datasets/honglihang/background-noise](https://www.kaggle.com/datasets/honglihang/background-noise)
    
    
    

    --- Salman Ahmed [EXPERT] rank=16 votes= date=2026-04-13
    btw, I think I get same LB without any other data than this competition's.
    But I think using that previous competition's data, might be helpful to make my model generalize better. 

    --- lhwcv [GRANDMASTER] rank=1704 votes= date=2026-04-13
    @salmanahmedtamu Hi Salman, are you still on Discord?
    

    --- Salman Ahmed [EXPERT] rank=16 votes= date=2026-04-13
    Hi @lihaoweicvch,
    Great to see you in this year's competition as well.
    Yes, but not too active there. 😅

    --- Ali Ozan Memetoglu [CONTRIBUTOR] rank=13 votes= date=2026-04-22
    Thank you :)) Are you using the soundscapes too or only this?

    --- Salman Ahmed [EXPERT] rank=16 votes= date=2026-04-22
    Just Train Audios. Just started experimenting with Soundscapes, and for now, adding Soundscapes reduce my LB to around 0.85. 

    --- Salman Ahmed [EXPERT] rank=16 votes= date=2026-04-22
    As I was typing this, i remember something, I think there is a bug in Soundscapes training. 😂 I'll fix it soon.

    --- Ali Ozan Memetoglu [CONTRIBUTOR] rank=13 votes= date=2026-04-22
    No, I mean for the background noise I was only using soundscapes so I’m thinking of combining them with the dataset you provided

    --- Salman Ahmed [EXPERT] rank=16 votes= date=2026-04-23
    Ah I see. No I don't use the SS as background noise, because there are species in those SS recordings.

  --- MengYe [EXPERT] rank=213 votes= date=2026-04-18
  does your cv correlate with lb?

  --- Salman Ahmed [EXPERT] rank=16 votes=5 date=2026-04-20
  Finally. Single model LB 0.937 without using Unlabeled data. 
  Recently started paying attention to the modeling part in this competition, Apparently, it was not that important in previous competitions, but in this one, for me it is one of the most important part. 
  I have been investigating attention values on some examples with highest loss during training, and minimum loss during training, and now it's much better compared to previous models I had. 

    --- Leon [GRANDMASTER] rank=1415 votes= date=2026-04-20
    Wow that is amazing! May I ask is it still efficientnet b0 ns?

    --- Salman Ahmed [EXPERT] rank=16 votes= date=2026-04-20
    xD LB 0.937 is with Efficientnet v2B0 because it's faster to train compared to others.
    Will try other variants (B0 NS / mixnet / nfnet etc) soon. 

    --- Seeing Times [EXPERT] rank=1606 votes= date=2026-04-20
    an amazing score, May I ask did you change sed head?

    --- Leon [GRANDMASTER] rank=1415 votes= date=2026-04-20
    hh sorry for keeping asking this. I am working on making my b0 single model to match your score xD. I think you mentioned before b0 MBConv doesn't learn well, and ebv2 is using Fused-MBConv, might be reason why its better (not sure). and may I also ask if your b0 model (0.922) is with any post processing (I assume only model soup)?

    --- Salman Ahmed [EXPERT] rank=16 votes=1 date=2026-04-20
    xD No worries, For that previous model architecture / training strategy MBConvs were not learning, but now B0_NS is matching or performing even better than v2B0. 
    Yeah, I am not using Model Soup this time, simple model checkpoint. 

    --- Shiro [GRANDMASTER] rank=1565 votes= date=2026-04-21
    Impressive results, with efficientnetb1 I am not able to reach that results with single models, trying different architectures. May I ask if you include postprocessing in the inference step or is it just a vanilla prediction on 5 seconds?

    --- Salman Ahmed [EXPERT] rank=16 votes= date=2026-04-22
    Hi @ludovick Yes, I definitely rely on some post processing / TTA.

    --- D.M. [CONTRIBUTOR] rank=1602 votes= date=2026-04-26
    How can cv correlate with LB? The training data are "clean" clips not embedded in soundscapes. Mixup of clean clips with unlabeled soundscapes doesn't work because there is unlabeled signal in the soundscapes. Mixup with labeled soundscapes helps got from .80 to .86 but there's not enough of them to go further. I'm stumped.

--- hengck23 [GRANDMASTER] rank=634 votes=6 date=2026-04-09
why some people get lb0.87+ and why get as high as 0.91+ for simple CNN model?
(exclude effects, pseudo-labels, extreme tricks ... we focus on base model performance)

the reason are:
- long tail (some class has few train samples)
- domain shift. focal audio: one species, high snr soundscape: multiple species, low snr (background noise, bird far from mic)

how to solve:
- long tail: pretrained model, sampling, more information per sample (e.g. 10,15 sec instead of 5 sec), regularization
- domain shift: augmentation (others: using unlabelled, external data)

what should you monitor
- auc: sensitive to few samples
- bce:  can skews towards low sample (bce hacking  ... like reward hacking)
- logit std: monitor this as a "proxy to logit distribution"

how a robust model would look like?
-  agreement for train and validation
- lower  logit std
- low bce
- high auc (split into acu for normal, rare, problematic classes etc) 
- can train for every long epoch (if augmentation is good)." Good grad descent at high lr"
- low auc variance in different folds
- flat loss valley for low sample class

how a overfit model would look like?
-  disagreement for train and validation
- high train logit std (growing as training iteration pogress)
-  low train bce
-  kigh train auc 

finally, cut and paste this post into you favourite AI llm and ask for comment. If useful, add to the skills in agent.

  --- hengck23 [GRANDMASTER] rank=634 votes=2 date=2026-04-09
  how to learn to be an expert in modelling. distill ChatGPT knowledge to yourself.
  - instead of asking chaptgpt to "improve the model", ...
  - Pick a point from the above post, e.g. "flat loss valley for low sample class"
  - ask chagpt:
    - how to measure flatness. how to prove current model methods is flat or not
    - suggest methods to improve flatness
    - implement and carry experiments
    - verify flatness is improved and lead to better train,validation and public LB.
    - if it does not, ask chatgpt why and most importantly how to prove? 
  
  "turning insights into measurable, testable pipelines"

--- Salman Ahmed [EXPERT] rank=16 votes=7 date=2026-04-03
Alright, Finally have 2 SED models with ensemble of LB 0.923. and solo around 0.921
Simple training with mixup on waves. 
25 epochs. 
CE Loss but Sigmoid at Infer
Trained with train_audio and labelled soundscapes only.
Without priors it achieves 0.917 and with priors -> 0.923. 
Nothing fancy, simple model training setup. 
Another thing I noticed, for melspecs, MBConvs from EfficientNets are not learning that well, compared to other architectures. 

WITHOUT any Perch KD

  --- Boredom [MASTER] rank=2 votes= date=2026-04-03
  Interesting findings. In your experiments, did EfficientNet underperform compared to other architectures? Also, may I ask what you mean by "Priors" here? In my experiments, a single EfficientNet model can easily achieve an LB score above 925+.

    --- Salman Ahmed [EXPERT] rank=16 votes= date=2026-04-03
    Yes, for me Efficientnet NS (with MBConv Architecture) were not converging fast and scored much lower on LB, compared to other models.
    Not sure why. 
    By Priors I mean, prior table of species based on site and hours. 

    --- Murilo Gustineli [CONTRIBUTOR] rank=412 votes= date=2026-04-06
    Great score with a single EfficientNet! I'm struggling to get EfficientNet-B0 to go beyond 0.855 LB, even when trying multiple different heads and loss functions. Could you share what loss function you're using (BCE vs CE) and whether you're using an attention-based head or something simpler? Trying to understand what unlocks 925+ on EfficientNet specifically

  --- Murilo Gustineli [CONTRIBUTOR] rank=412 votes= date=2026-04-06
  Congrats on the 0.921! My best single model so far got 0.906 LB with HGNetV2-B0 + LSEHead + BCE. I tried switching to AttHead (with tanh attention, sigmoid-space clipwise, dual 50/50 supervision) but it got worse LB results both with BCE and CE loss. Two quick questions if you don't mind: (1) what backbone did you switch to from EfficientNet? and (2) are you using 5s or longer windows for training? Thanks!

    --- hengck23 [GRANDMASTER] rank=634 votes=1 date=2026-04-06
    i suppose your 0.906 comes from distillation. Comparing with your previous result, you would note that the improvement comes from distillation and not data.   
    What to distill? logit? embedding or spatial embedding or all?  
    
    if the results is from distillation, then maybe modeling is less impotant.
    
    ---
    
    if you look at  public notebook which is just based on perch2 emebdding, they are :  
    lb 0.930 = (perch2 + proxy class + temporal + PCA) + prior + state space + perclass normalisation ....
    
    you can estimate   
    lb 0.92+ = (perch2 + proxy class + temporal + PCA) 
    
    this means to get lb 0.92+ from  timm, you need to add temporal
     

    --- Murilo Gustineli [CONTRIBUTOR] rank=412 votes= date=2026-04-06
    I'm not using knowledge distillation at all actually. The 0.906 is just HGNetV2-B0 trained from ImageNet init on focal + labeled soundscapes with BCE loss. No Perch at all.
    
    Really interesting point about temporal. I've tried your 3x5s multi-context approach (backbone sees 15s, split into 3 chunks, LSEHead per-chunk) and got best-ever offline val metrics, but it regressed on LB (0.896 with 4×5s avg, vs 0.906 baseline). What exactly do you mean by "add temporal"? is it the multi-context you showed, or something else like temporal embeddings or sequence modeling?

  --- Aic [EXPERT] rank=221 votes= date=2026-04-08
  May I ask, do you experience any score fluctuations in your experiments due to different numbers of epochs or changes in other hyperparameters?...it's been giving me a headache lately.

    --- Mahog [EXPERT] rank=1373 votes= date=2026-04-16
    Have you managed to fix this? I'm having the same issue, even a seed change shifts LB by 0.005 :V

  --- D.M. [CONTRIBUTOR] rank=1602 votes= date=2026-04-25
  > 0.921 Simple training with mixup on waves. Trained with train_audio and labelled soundscapes only.
  No clue how you are doing that. I get .865 when I do that :/

--- hengck23 [GRANDMASTER] rank=634 votes=6 date=2026-04-04
how to use CNN only (no perch2, no post processing) to get 0.896 lb  
context and augmentation is the key. here we cheat by using CNN and long input wave to capture temporal information

hint: how to get even better results: multi-conext head

![](https://www.googleapis.com/download/storage/v1/b/kaggle-forum-message-attachments/o/inbox%2F113660%2F29fb5caf287a72c18a9783d462005f01%2FSelection_2776.png?generation=1775293726250796&alt=media)

  --- Tom [MASTER] rank=7 votes=1 date=2026-04-04
  This is similar to my current approach, but I made a slightly different approach which fits on 4x5sec. 

    --- hengck23 [GRANDMASTER] rank=634 votes= date=2026-04-04
    I checked your Claude report. He actually knows this trick the very first day. That is impressive 

  --- unknown [] rank= votes= date=2026-04-04
  

  --- D.M. [CONTRIBUTOR] rank=1602 votes= date=2026-04-05
  I sort of see the overall goal, but this is too sketchy for me. I don't know how many context_nets there are or what clip length they've been trained on, nor what the '3' is in the shape of prob3. I think the bottom loop is sliding width-4 windows within a 12 timestep clip and averaging the last 2 dimensions of prob3 in each of the 12 cells. But then I still don't know what the 3 dimension is, nor what 'probs' is at the end.

  --- goevian [CONTRIBUTOR] rank=181 votes= date=2026-04-15
  may i ask, what do you mean by"multi-context head"?

--- Murilo Gustineli [CONTRIBUTOR] rank=412 votes=6 date=2026-03-30
My best single SED model is 0.86 LB with EfficientNet-B0 - and I've been struggling to break past that despite extensive ablations.

**My setup:** EfficientNet-B0 SED (ImageNet init), GeM + attention head, n_fft=4096, n_mels=224, AmplitudeToDB norm, soft CE dual loss (clipwise + framewise max), 10s windows, spectrogram MixUp (α=0.5), 90/10 split, 50 epochs with CosineAnnealing. Trained on focal + labeled soundscapes (234 classes). 

I've run 15+ ablations (BCE, ASL, LSEHead, HGNetV2-B0, waveform MixUp, weight decay, StochasticDepth) and can't break past 0.86 - offline val improvements consistently fail to translate to LB gains.

I also reproduced [ttahara's HGNetV2-B0 recipe](https://www.kaggle.com/code/ttahara/birdclef-2026-hgnetv2-b0-baseline-training) (LSEHead, BCE, OneCycleLR, per-sample MixUp, 5s windows, 20 epochs) and got 0.872 val/auroc offline, but only 0.855 LB on EfficientNet with the same recipe. The val-to-LB disconnect has been consistent across all ablations - improvements in offline val rarely translate to LB gains.

Curious what's driving the gap to 0.9+ for other single EfficientNet-B0 SEDs. A few things we haven't explored:
- NS-JFT pretrained weights ([tf_efficientnet_b0.ns_jft_in1k](https://huggingface.co/timm/tf_efficientnet_b0.ns_jft_in1k)) vs standard ImageNet init
- 3-channel mel input (repeat mono to match pretrained stem) vs 1-channel with weight averaging
- Longer context windows (15-20s) for capturing insect/amphibian textures
- Training on focal only vs focal + soundscapes - currently training on **focal + soundscapes**

Would love to hear from folks who've broken 0.9+ @aliozanmemetoglu @hengck23 @tom99763 @cudacoding @salmanahmedtamu @hideyukizushi what was the single biggest change that got you there? Any pointers would be appreciated!

  --- Salman Ahmed [EXPERT] rank=16 votes=3 date=2026-03-31
  In my experiments, more epochs make the model too confident on it's predictions and that hurt the AUC (Evaluation metric). 
  
  In my experiments, SED simple Attn head's framewise max is better than lse head.
  
  I mixup waves rather than Spectrograms.
  
  For now, Efficientnet B0 didn't help. 
  
  Try to listen to a audio from train soundscape and then look at the attentionhead's output. That might help.

  --- hengck23 [GRANDMASTER] rank=634 votes=2 date=2026-03-31
  it is better to think of baseline  
  1) use perch embedding to get cv and public lb  
  - just train a linear or MLP probe on clip and evaluate on soundscape ss (in addition to clip). this give you an idea on effect of domain shift.    
  
  2) repeat (1) but using your model. train on clip only   
  - read perch paper and see how many train samples they use and compare that with kaggle data. this gives you an idea of how num and variety of data affects results.  
  
  
  from these two, it would be enough to see if your results is reasonable or no and other kagger top results is due to tricks, post processing, better modelling, better data, etc 
  

  --- yukiZ [GRANDMASTER] rank=144 votes=2 date=2026-03-31
  Hi! Regarding the contribution of CV/LB in my experiment, data augmentation was effective.
  
  I believe many publicly available notebooks include Mixup.
  On the other hand, data augmentation on raw waveforms is not seen very often, but it has been frequently used in solutions from past BirdCLEF competitions. For example, "Pink Noise" and "Time Stretch".

--- Salman Ahmed [EXPERT] rank=16 votes=13 date=2026-03-22
My best single model is SED Efficientnet B0 (Trained on 10 sec). 
Mixup on Raw Waves, and CrossEntropyLoss on clipwise and framewise max. Trained for 20+ epochs.
This single model scores 0.908

I have a ensemble of these 2 models and it scores 0.914

No other augmentation. No unlabeled data.

  --- unknown [] rank= votes= date=2026-03-28
  

  --- Murilo Gustineli [CONTRIBUTOR] rank=412 votes= date=2026-03-30
  Is it safe to assume that you're only training on `train_audio` or also using the `train_soundscapes` to achieve the 0.908 LB score?

    --- Salman Ahmed [EXPERT] rank=16 votes=1 date=2026-03-31
    Yup. Train audio + label soundscapes.

    --- Jiacheng Ma [CONTRIBUTOR] rank=1195 votes= date=2026-03-31
    What are label soundscapes and unlabel soundscapes? Thanks.

    --- Salman Ahmed [EXPERT] rank=16 votes=1 date=2026-03-31
    train_soundscapes_labels.csv are labeled soundscapes. 

  --- unknown [] rank= votes= date=2026-04-05
  

--- yukiZ [GRANDMASTER] rank=144 votes=7 date=2026-03-22
I believe our interest lies in whether there is a correlation between CV and LB, so I will share my experimental results.
However, due to the highly skewed distribution of the data, the validation strategy will likely differ depending on the participant.

The following experimental results show the results of training/submitting using only Fold0 with 5Fold SKF, assuming future Blending of the BB of the SED Model with B0 (approximately 5-7M parameters) or Alternatively, it uses a backbone of around 15M, similar to v2s.

|MyExpNo  |AUC(LocalValidation※OnlyFold1/5)  | LB(※OnlyFold1/5) |
| --- | --- |--- |
| #107| **0.9774** | **0.904** |
| #105| **0.9756** | **0.902** |
| #042v4| **0.9768** | **0.898** |
| #042 | **0.9744** | **0.888** |
| #041  | **0.9737** | **0.885** |
| #040 | **0.9722** | **0.878** |

* *[2026/03/27 Upd. Add Exp "#042v4","#105","#107"]*


  --- Amit Aharoni [EXPERT] rank=1224 votes=1 date=2026-03-23
  Thanks for sharing
  Based on the AUC score, is it safe to assume that your local validation is using train_audio and not train_soundscape?

    --- yukiZ [GRANDMASTER] rank=144 votes=1 date=2026-03-24
    Yes, your way of thinking is correct.

--- Boredom [MASTER] rank=2 votes=7 date=2026-03-22
Single model / Single Fold: 0.922 without using the unlabeled/extra data

  --- vion [EXPERT] rank=1198 votes=1 date=2026-03-26
  Amazing score considering that 0.940 is the maximum achievable with 206 of 234 classes. Congrats! 

  --- Jiacheng Ma [CONTRIBUTOR] rank=1195 votes=1 date=2026-03-31
  You only use train_audio? no train_soundscape?

    --- Boredom [MASTER] rank=2 votes= date=2026-04-03
    train_audio + train_soundscape (w/ label)，w/o pseduo label

    --- Boredom [MASTER] rank=2 votes= date=2026-04-03
     I guess without using unlabeled data, the highest LB (leaderboard score) we can achieve is around 0.94-0.945.

    --- !!!!! [EXPERT] rank=888 votes= date=2026-04-03
    May I ask you use Perch or just timm-based model? For me, I get low score (0.88) using pretrained timm model.

  --- Boredom [MASTER] rank=2 votes=3 date=2026-04-13
  update:   Single model: 0.947+   about 13min

    --- Kurise [MASTER] rank=10 votes=1 date=2026-04-13
    Amazing results, did you used the unlabeled data?

    --- Boredom [MASTER] rank=2 votes=4 date=2026-04-13
    Yes, absolutely. I would be very surprised if someone could achieve above 0.945 with a single model without using unlabeled data😭. My best single model without unlabeled data is around 0.937-0.938.

    --- Aic [EXPERT] rank=221 votes= date=2026-04-13
    Really impressive! Was your best single model based on Perch distillation, or was it a non-Perch model?

    --- Boredom [MASTER] rank=2 votes= date=2026-04-13
    non-Perch, maybe Perch can do better.

    --- Ali Ozan Memetoglu [CONTRIBUTOR] rank=13 votes=4 date=2026-04-13
    My single best model is 0.941 but I use last year's pretrained checkpoint so I believe it is possible :D 

    --- Aic [EXPERT] rank=221 votes= date=2026-04-13
    Impressive stuff!

    --- Boredom [MASTER] rank=2 votes= date=2026-04-13
    😱Did you use extra data?I've been struggling for a long time to break 0.94 without using unlabeled data..

    --- Kurise [MASTER] rank=10 votes= date=2026-04-13
    You did what I want to do tomorrow! May I ask how much will it improve by using last year's checkpoint?

    --- Kurise [MASTER] rank=10 votes= date=2026-04-13
    Not sure whether I'm right, using last year's checkpoint just like 2 stage training(pretrain on 25, retrain on 26), but there is no extra computation cost. So that we can get a model trained on both 25 and 26 data.

    --- Ali Ozan Memetoglu [CONTRIBUTOR] rank=13 votes=2 date=2026-04-13
    They used the entire Xeno-Canto dataset by filtering it and fixing a bug during pretraining, but I didn’t use anything beyond the checkpoint.

    --- Cody_Null [GRANDMASTER] rank= votes= date=2026-04-14
    Im not super excited about the idea of doing all these models 5 fold on the entire XC dataset but we gotta do what we gotta do 😂

    --- Ali Ozan Memetoglu [CONTRIBUTOR] rank=13 votes= date=2026-04-15
    it is pretraining, so there are no folds

  --- Boredom [MASTER] rank=2 votes=3 date=2026-04-21
  Update: Single model LB 0.950+ (11 minutes). After reading Salman’s latest post, I realized I might have been constrained by my experience from past competitions, which may have limited the optimal single-model performance of this pipeline to around 0.953–0.955. I will soon build the next pipeline from scratch for ensembling, hoping to push even higher.

    --- Starry [MASTER] rank=1 votes= date=2026-04-23
    Hi Boredom, it is an amazing result, may I ask if you use unlabeled train_soundscapes in your 0.950+ model. And if so, what the score with only train_audio + labeled train_soundscapes ?

    --- Boredom [MASTER] rank=2 votes= date=2026-04-23
    Yep！this is the result of the first iteration on unlabeled data. Maybe one or two more iterations are possible. A single model using only labeled data achieves an LB of 0.938–0.939. I've never been able to break 0.94 with a single model. Compared to others with solid work, there might still be a significant gap in potential.

    --- Shiro [GRANDMASTER] rank=1565 votes= date=2026-04-25
    Impressive! does it include TTA/postprocessing during inference? Is there a big gap between with/without those postprocessing in term of LB score?

    --- Boredom [MASTER] rank=2 votes=1 date=2026-04-25
    no tta，950+ lb include some simple pp，these can improve about 0.03-0.04

    --- Kurise [MASTER] rank=10 votes= date=2026-04-25
    Are you sure? I don't know how "simple" pp can enhance 0.04😨

    --- Boredom [MASTER] rank=2 votes=1 date=2026-04-25
    sorry，0.004。haha😂

  --- Gaurav Rawat [MASTER] rank=420 votes= date=2026-04-23
  nice SED or vision model ?

    --- Boredom [MASTER] rank=2 votes=1 date=2026-04-23
    timm backbone+SED 

--- Ali Ozan Memetoglu [CONTRIBUTOR] rank=13 votes=8 date=2026-03-22
Mine is at 0.925 without using the unlabeled data. I couldn’t make good use of it, and it might not even be that useful in this competition. But I might be wrong.

  --- Tom [MASTER] rank=7 votes=2 date=2026-03-22
  I think the floor of top scores would be very high in the end

  --- hengck23 [GRANDMASTER] rank=634 votes= date=2026-03-23
  "nd it might not even be that useful in this competition. But I might be wrong." 
  
  it should be very useful! in fact winng solution is how you make use of it

--- XuKong Ji [EXPERT] rank=1402 votes= date=2026-04-13
I have a single fold(one of 5folds) model(effecient b0+LSEHead + BCE) score 0.899 with 5s input/time smooth/time shift tta(shift 0s and 2.5s). I got something weird:
1. 5fold ensemble score 0.889;
2. After incresing input to 10s, score drop to 0.86(1fold)... 

--- OpPrime [CONTRIBUTOR] rank=289 votes= date=2026-04-12
I got 0.826 on EfficientNetB0 with 5-fold training, 0.835 on fold 0, 10 epochs per fold, feeding in 5 seconds clips only. 
Notebook and weights shared now. 

Following Salman's post below, I then got a 0.801 from SED HGNetV2 feeding it 20s of mel with a temporal attention head. 

I am now tracing how and where I am going wrong as my numbers don't match with others.


  --- EliKal [MASTER] rank=991 votes= date=2026-04-12
  I’d start by focusing on your 5s models. You can get 0.915+ with an ensemble of just two models.
  For 20s inputs, I’d try adding segment-wise labels from the labeled soundscapes. That said, you’ll probably need to adjust the backbone or increase the resolution, since a larger context usually means a larger model.
  It might be worth trying 10s inputs first with the same backbone and resolution, just to see how things behave. My 10s model (EfficientNet-B0 as a starting point) isn't ready yet, I hope my intuition will work.

--- Tucker Arrants [MASTER] rank=203 votes= date=2026-04-12
Frozen Perch embeddings + learned pool + GRU 

5 folds, 3x time-shift TTA

0.91 public leaderboard

**Edit**: I was able to score 0.925 without any deep learning, just some simple clustering on the frozen perch embeddings and metadata engineering. The learned pool + GRU was total over kill. 

--- Gaurav Rawat [MASTER] rank=420 votes= date=2026-04-07
- 0.82 cv lb 908
- 0.94 cv lb 913
- 0.97 cv lb 926
- 0.87 cv lb 916 :/
- 0.942 cv lb 915
- SED 0.87 - 0.75 :(

Weird but

  --- Aic [EXPERT] rank=221 votes= date=2026-04-07
  May I ask about your CV strategy? Mine is quite unreliable, even though my SS_AUC is closely aligned with the LB.

    --- Gaurav Rawat [MASTER] rank=420 votes=1 date=2026-04-08
    pretty simple groupk on site 

    --- EliKal [MASTER] rank=991 votes=1 date=2026-04-08
    Did you validate on xc, inat and ss data together?

    --- Gaurav Rawat [MASTER] rank=420 votes= date=2026-04-09
    No, I validate strictly on Soundscapes (SS) using GroupKFold by Site as of now . 

    --- EliKal [MASTER] rank=991 votes= date=2026-04-09
    Interesting. When I tried to group the Soundscapes by site I had many missing species in the validation set. Your cv scores are OOF scores? It would make sense much more if they are

  --- Anil Ozturk [MASTER] rank=59 votes=1 date=2026-04-09
  Soundscape-val-only GKF on sites OOF: 0.60775
  LB: 0.879
  
  lol

    --- Tucker Arrants [MASTER] rank=203 votes= date=2026-04-09
    GKF is overly harsh for this competition because the test set has site overlap with training. GKF models can't learn site-level species priors that would help at test time — they're forced to hold out entire sites during training, so they never see the "this site has these species" signal that exists in reality.
    But removing GKF entirely is also risky: the model might memorize site-specific artifacts that don't transfer, and you'd have no way to detect that the model relies on site memorization versus learning general features.
    
    But you can try dual validation on a single training run: train on everything, but hold out two disjoint val sets. Val-A holds out specific recordings from seen sites (tests the regime: new recordings at seen sites). Val-B holds out an entire site (tests whether the model has any ability to generalize to unseen sites as a hedge against test sets that include unseen sites). Combine them into a weighted metric for early stopping, weighted toward Val-A because it matches the test regime more directly.

--- EliKal [MASTER] rank=991 votes= date=2026-04-05
EfficientNet-B0 – 0.909
EfficientNet-B3 – 0.906
DINOv3-small – 0.906
ConvNeXt-small – 0.907
All models were trained using the same pipeline for 12 epochs (after 10 epochs of distillation with KLD loss), with mixup applied to spectrograms. After reading the comments, I’m considering replacing this with mixup on raw waveforms.
Inference was performed using only time-shift TTA.

  --- tennogh [EXPERT] rank=53 votes= date=2026-04-05
  Interesting that you have so little difference between backbones. With my pipeline I have up to .02 difference between model families so far.

    --- Jack [EXPERT] rank=106 votes= date=2026-04-08
    It's the distillation (assuming he's distilling perch) causing that stability. I experience the same swings as you with different backbones

    --- EliKal [MASTER] rank=991 votes= date=2026-04-08
    Correct. Perch embeddings for all the architectures.
    Did you solve this stability "issue"?

    --- tennogh [EXPERT] rank=53 votes= date=2026-04-08
    Not sure if this is an issue, I guess it could make the models more similar and ensembling less beneficial. I'll check when I have a stable KD pipeline.
    
    Edit: with KD and 4 different backbones, I still have a ~0.02 difference between the worst and the best (the ranking is a little bit different too).
    KD also reduces diversity to an extent (cosine similarity ~0.5->~0.67), to the point where is might be worth ensembling KD and non-KD models.

--- D.M. [CONTRIBUTOR] rank=1602 votes= date=2026-04-03
Mine is only 0.880, fine-tuning an imagenet-pretrained CNN backbone that takes spectrogram input. The backbone outputs low-temporal resolution BCT (T=10, C=1024), that I then map to 234 and pool with logit-LSE to get clip logits, trained with BCE loss. The .880 isn't really replicable, more like .875 only. Trained on train_audio and the labeled soundscapes. Using 5-second clips.

I tried lots of mixing schemes with the unlabeled soundscapes but that didn't transfer to the LB for me. I can't imagine how it's possible to get something like .922 using the same data I'm using.

I also trained a 4-layer residual mamba model from scratch (no pretrained network at all), but that topped out at .772.

  --- hengck23 [GRANDMASTER] rank=634 votes= date=2026-04-03
  " The .880 isn't really replicable, more like .875 only." this is due to small sample size in some class. bird clef is domain and long tail problem. the issue can be solved by better validation and training methods. Hence the potential of timm and "train from pretained CNN/scratch" can be higher than you think.
  
  
  although "train from pretained CNN/scratch" can achieve good results, using perchv2 may be faster and easier.
