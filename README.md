# Deep Spatiotemporal Clutter Filtering Network
![Unet.pdf](https://github.com/mtab2020/Deep_Clutter_Filtering/files/10893047/Unet.pdf)
*Architecture of the proposed spatiotemporal clutter filtering network. It is a fully convolutional auto-encoder designed based on the 3D U-Net to generate filtered transthoracic echocardiographic (TTE) sequences which are coherent both in space and time. An input-output skip connection was added to the auto-encoder for preserving fine image structures, together with the attention gate (AG) modules to aid the network focusing on the clutter zones.*

## Results
![Filtered_eg.pdf](https://github.com/mtab2020/Deep_Clutter_Filtering/files/10893053/Filtered_eg.pdf)

*(a) Examples of the cluttered test frames and ((b)-(h)) the clutter-filtered frames of the six vendors generated by the examined deep networks. (b), (c) and (d) The proposed 3D filter trained with L_rec, L_rec&adv and L_rec&prc, respectively. (e), (f) and (g), the 3D benchmark networks. (h) the 2D benchmark network. (i) The clutter-free frames. For each vendor, absolute difference images computed from the clutter-filtered and clutter-free frames are shown in the row below the respective filtered frames.*

*Example video clips of (a) the cluttered and (d) clutter-free TTE sequences and the filtering results generated by (b) the proposed 3D filter and (c) the 2D filter (both trained with the in-out skip connection, AG module and L_2 loss). Absolute difference video clips computed from the clutter-filtered and clutter-free frames are shown in the row below.*  

https://github.com/MahdiTabassian/Deep-Clutter-Filtering/assets/73531266/bfd86eff-27f2-4418-b120-d1855daa4d5f

https://github.com/MahdiTabassian/Deep-Clutter-Filtering/assets/73531266/1691f0ba-f9a5-4bdb-aed2-f4c3b86ba9a0

https://github.com/MahdiTabassian/Deep-Clutter-Filtering/assets/73531266/153d988e-1d41-4535-a11c-b977001b58ff

https://github.com/MahdiTabassian/Deep-Clutter-Filtering/assets/73531266/2ee89f78-178e-4bfc-8a09-a2fcf375d912

https://github.com/MahdiTabassian/Deep-Clutter-Filtering/assets/73531266/8d05a599-f559-4076-acd2-d4d34698aa8e

https://github.com/MahdiTabassian/Deep-Clutter-Filtering/assets/73531266/8f565da7-3115-49dd-ac9b-c520806ea728

*(a) Example video clips of the in-vivo TTE sequences of four different subjects which are contaminated by the NF and/or RL clutter patterns. (b) The filtering results generated by the proposed 3D filter and (c) the 2D filt. Absolute difference video clips computed from the cluttered and clutter-filtered frames are shown below the filtered frames.*

https://github.com/MahdiTabassian/Deep-Clutter-Filtering/assets/73531266/cd23f8a7-cbb8-4a78-a20e-a716f8cd79b1

https://github.com/MahdiTabassian/Deep-Clutter-Filtering/assets/73531266/4bf2aca2-bb58-4978-89d2-394b0903fed9

https://github.com/MahdiTabassian/Deep-Clutter-Filtering/assets/73531266/8fbe9734-7436-4104-8734-5e177893082c

https://github.com/MahdiTabassian/Deep-Clutter-Filtering/assets/73531266/5489ad3c-0193-46ed-b388-54aa7f201766

## Installation

After cloning the repository, run the following command:
```
pip install -r requirements.txt
```
To train each filter using the synthetic data, provide directory of the data and a directory for saving the results in `config.json` of that filter. Then run the following command after changin directory to the filter directory (e.g. for the 3D filter with reconstruction loss):
```
python TrainClutterFilter3D.py --config config.json
```

## Citation
