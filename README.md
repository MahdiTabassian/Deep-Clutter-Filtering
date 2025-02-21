# Deep Spatiotemporal Clutter Filtering Network

*Architecture of the proposed spatiotemporal clutter filtering network. It is a fully convolutional auto-encoder designed based on the 3D U-Net to generate filtered transthoracic echocardiographic (TTE) sequences which are coherent both in space and time. An input-output skip connection was added to the auto-encoder for preserving fine image structures, together with the attention gate (AG) modules to aid the network focusing on the clutter zones.*
<img width="1236" alt="Clutter_filter" src="https://github.com/user-attachments/assets/c88fc727-fbb7-42dd-ae68-a8bdfb9149a1">

## Results

<img width="1143" alt="Filtered_eg1" src="https://github.com/user-attachments/assets/a54025a9-8f39-44c3-8c74-671693003016">

<img width="1140" alt="Filtered_eg2" src="https://github.com/user-attachments/assets/6071d3d5-aa3e-4e6f-ab12-cab6f4676562">

*(a) Examples of the cluttered test frames and ((b)-(h)) the clutter-filtered frames of the six vendors generated by the examined deep networks. (b), (c) and (d) The proposed 3D filter trained with L_rec, L_rec&adv and L_rec&prc, respectively. (e), (f) and (g), the 3D benchmark networks. (h) the 2D benchmark network. (i) The clutter-free frames. For each vendor, absolute difference images computed from the clutter-filtered and clutter-free frames are shown in the row below the respective filtered frames.*

*Example video clips of (a) the cluttered and (d) clutter-free TTE sequences and the filtering results generated by (b) the proposed 3D filter and (c) the 2D filter (both trained with the in-out skip connection, AG module and reconstruction loss). Absolute difference video clips computed from the clutter-filtered and clutter-free frames are shown in the row below. (Raw files can be found in `Filtering_results_videos
/synthetic`).*  


https://github.com/user-attachments/assets/cfa40692-7091-4e44-a533-d788cd30ef1f



https://github.com/MahdiTabassian/Deep-Clutter-Filtering/assets/73531266/1691f0ba-f9a5-4bdb-aed2-f4c3b86ba9a0

https://github.com/MahdiTabassian/Deep-Clutter-Filtering/assets/73531266/153d988e-1d41-4535-a11c-b977001b58ff

https://github.com/MahdiTabassian/Deep-Clutter-Filtering/assets/73531266/2ee89f78-178e-4bfc-8a09-a2fcf375d912

https://github.com/MahdiTabassian/Deep-Clutter-Filtering/assets/73531266/8d05a599-f559-4076-acd2-d4d34698aa8e

https://github.com/MahdiTabassian/Deep-Clutter-Filtering/assets/73531266/8f565da7-3115-49dd-ac9b-c520806ea728

*(a) Example video clips of the in-vivo TTE sequences of four different subjects which are contaminated by the NF and/or RL clutter patterns. (b) The filtering results generated by the proposed 3D filter and (c) the 2D filt. Absolute difference video clips computed from the cluttered and clutter-filtered frames are shown below the filtered frames. (Raw files can be found in `Filtering_results_videos
/in-vivo`).*

https://github.com/MahdiTabassian/Deep-Clutter-Filtering/assets/73531266/cd23f8a7-cbb8-4a78-a20e-a716f8cd79b1

https://github.com/MahdiTabassian/Deep-Clutter-Filtering/assets/73531266/4bf2aca2-bb58-4978-89d2-394b0903fed9

https://github.com/MahdiTabassian/Deep-Clutter-Filtering/assets/73531266/8fbe9734-7436-4104-8734-5e177893082c

https://github.com/MahdiTabassian/Deep-Clutter-Filtering/assets/73531266/c3c2c37d-d293-44ca-aaf6-16767b35f986


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
Please cite as:
```
@misc{tabassian2025deepspatiotemporalclutterfiltering,
      title={Deep Spatiotemporal Clutter Filtering of Transthoracic Echocardiographic Images: Leveraging Contextual Attention and Residual Learning}, 
      author={Mahdi Tabassian and Somayeh Akbari and Sandro Queirós and Jan D'hooge},
      year={2025},
      eprint={2401.13147},
      archivePrefix={arXiv},
      primaryClass={eess.IV},
      url={https://arxiv.org/abs/2401.13147}, 
}
```
