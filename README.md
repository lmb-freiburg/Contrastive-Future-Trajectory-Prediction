# Contrastive-Future-Trajectory-Prediction

This repository corresponds to the official source code of the ICCV 2021 paper:

<a href="https://arxiv.org/pdf/2103.12474.pdf">On Exposing the Challenging Long Tail in Future Prediction of Traffic Actors</a>

### Requirements
We use the same requirements as the Trajectron++, see:
https://github.com/StanfordASL/Trajectron-plus-plus

Additionally, it is essential to download the Trajectron++ code, rename it to ```Trajectron_plus_plus``` and place it next to other folders (e.g., `data/, models/`).



### Data

#### ETH-UCY: 
The test data files are provided under ```data/```.
These are the result of running the processing script of the Trajectron++, see: 
https://github.com/StanfordASL/Trajectron-plus-plus/blob/master/experiments/pedestrians/process_data.py

#### nuScenes (Bird's-eye view):
For the processed files, you can run the processing script of nuScenes at:
https://github.com/StanfordASL/Trajectron-plus-plus/blob/master/experiments/nuScenes/process_data.py

### Pre-trained Models
All pretrained models (EWTA and with contrastive learning) are provided under ```models/```. 

### Testing
This is an example call of the testing script (test Trajectron++EWTA on ETH):

```
python test.py --model models/eth_ewta/ --checkpoint 490 --data data/eth_test.pkl --kalman kalman/eth_PEDESTRIAN_test_kalman.pkl --node_type PEDESTRIAN
```

Another example to test all vehicles on nuScenes dataset:

```
python test.py --model models/nuScenes_ewta/ --checkpoint 25 --data data/nuScenes_test_full.pkl --kalman kalman/nuScenes_VEHICLE_test_kalman.pkl --node_type VEHICLE
```

### Training
Coming soon...

### Citation
If you use our repository or find it useful in your research, please cite the following paper:


<pre class='bibtex'>
@InProceedings{MCMB21,
  author       = "O. Makansi and {\"O}. {\c{C}}i{\c{c}}ek and Y. Marrakchi and T. Brox",
  title        = "On Exposing the Challenging Long Tail in Future Prediction of Traffic Actors",
  booktitle    = "IEEE International Conference on Computer Vision (ICCV)",
  month        = " ",
  year         = "2021",
  url          = "http://lmb.informatik.uni-freiburg.de/Publications/2021/MCMB21"
}
</pre>