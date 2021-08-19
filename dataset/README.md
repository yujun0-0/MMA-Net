## VIL-100 Dataset: A Large Annotated Dataset of Video Instance Lane Detection

`VIL-100` dataset, is released on the article ***VIL-100: A New Dataset and A Baseline Model for
Video Instance Lane Detection*** [[pdf](https://arxiv.org/)], which can be downloaded free on [[baidu disk](https://pan.baidu.com/s/1T4XcZUIHY1kCd0Bo0OngGA)], password: **kw4l**.

### Intorduction
1. **Data Collection and Split**
    - It consists of `100 videos`, `100 frames per video`, 0-6 lanes per frame, in total `10,000 frames` (97 videos are collected by monocular forward-facing camera, 3 videos are from Internet).
    
    - It contains 10 typical scenarios: `normal, crowded, curved road, damaged road, shadows, road markings, dazzle light, haze, night, crossroad`.
    
    - Training set and test set are according to the ratio of `8:2`, all 10 scenarios are presented in both them.

2.  **Annotation**
   
    **PNG-format images**
    
    >The images are in **'P' model**, the points of each lane are fitted into a curve by third-order polynomial, and expanded into a lane-region with a certain width. Empirically, on 1,920 ¡Á 1,080 frames, the lane width is 30 pixels. For lower-resolution
frames, the width is reduced proportionally.

    **JSON-format files**
     ``` 
     *********** A sample of one json-file ***********
     {
	        "camera_id": 8272,
	        "info": {
		        "height": 1080 , 
		        "width": 1920,
		        "date": "2020-11-24",
		        "image_path": "0_Road014_Trim005_frames/XXXXXX.jpg"
	        },
	        "annotations": {
		        "lane": [{
			        "id": 1, 
			        "lane_id": 1,
			        "attribute": 1,
			        "occlusion": 0,
			        "points": [[412.6, 720],[423.7, 709.9], ...]
		        }, {...}, {...}, {...}]
	        }
       }
      ```
         
    **`height / width:`** the height and width of image in *image_path* .   
    **`id:`** The sequential index of lanes : 1, 2, 3, 4, 5, 6.    
    **`lane_id:`** A label reflects its relative position to the ego vehicle, i.e., an even label 2i indicates the i-th lane to the right of vehicle while an odd label 2i - 1 indicates the i-th lane to the left of vehicle (i = 1, 2, 3, 4), the example as follows:   
    
     <img src="https://z3.ax1x.com/2021/08/19/fHS83T.png" height = "230" alt="Í¼Æ¬Ãû³Æ" align=center />

    **`attribute:`** Linetype of each lane, total 10 linetypes, the correspondence is as follows:
   
    <table style="width: 1500px; text-align: left; margin-left:30px">
        <tr>
            <th >linetype</td>
            <th >attribute</td>
            <th >linetype</td>
            <th >attribute</td>
            <th >linetype</td>
            <th >attribute</td>
        </tr>
        <tr>
            <td>single white solid</td>
            <td>1</td>
            <td>single white dotted</td>
            <td>2</td>        
            <td>single yellow solid</td>
            <td>3</td>
        </tr>
        <tr>
            <td>single yellow dotted</td>
            <td>4</td>
            <td>double white solid</td>
            <td>5</td>        
            <td>double yellow solid</td>
            <td>7</td>
        </tr>
        <tr>
            <td>double yellow dotted</td>
            <td>8</td>
            <td>double white solid dotted</td>
            <td>9</td>        
            <td>double white dotted solid</td>
            <td>10</td>
        </tr>
        <tr>
            <td>double solid white and yellow</td>
            <td>13</td>
        </tr>
    </table> 
   
    **`occlusion:`** Whether the current lane line is occluded: 0(no) or 1(yes).    
    **`points:`** The set of coordinates of points positioned along the center line of each lane.


3. **Dataset Features and Statistics**

    <img src="https://z3.ax1x.com/2021/08/19/fHSgDH.png" width = "740" height = "250" alt="Í¼Æ¬Ãû³Æ" align=center />
    
    `Figure 2 (a) shows the frame-level frequency of such co-occurrence of the 10 scenarios in VIL-100.`        
    `Figure 2 (b) shows the total number of frames for each scenario ¨C a frame with co-occurred scenarios is counted for all present scenarios.`

    <img src="https://z3.ax1x.com/2021/08/19/fHSZjg.png" width = "900" height = "300" style="margin-bottom:20px" alt="Í¼Æ¬Ãû³Æ" align=center />

    `Figure 3 (a) shows the linetype distribution.`        
    `Figure 3 (b) shows the statistics of the number of lanes per frame.`    
    
        
    
4. **More information is on [pdf](https://arxiv.org/)** or see [Contacts].

### Documentation


The directory is structured as follows:

 * `ROOT/JPEGImages`: Set of input video sequences provided in the form of JPEG images. Video sequences are available at original resolutions: (960,480), (960, 448), (672, 378), (960, 474), (1920, 1080), (1280, 720), (640, 368), (960, 478).

 * `ROOT/Annotations`: Set of lane-region images of each video sequences in PNG-format.

 * `ROOT/JSON`: Set of the meta information of each video sequences in json-format.
 
 * `ROOT/data`: A yaml file about split of training/test sets.
 
 > Note that, the clip/sequence dir name under `ROOT/JPEGImages` is based on the 10 scenarios. For example:    
 > * **1269_Road022_Trim002_frames**: '1269' means the current scenarios are: crowded, curve, dazzle night and haze.
 > * **3_Road017_Trim007_frames**: '3' means the current scenario is: damage.
 > * The corresponding between scenarios and numbers is as follows:
 
   <table style="width: 1500px; text-align: left; margin-left: 20px">
     <tr>
        <th >scenario</td>
        <td >normal</td>
        <td >crowded</td>
        <td >curved road</td>
        <td >damaged road</td>
        <td >shadows</td>
        <td >road markings</td>
        <td >dazzle light</td>
        <td >haze</td>
        <td >night</td>
        <td >crossroad</td>
     </tr>
    <tr>
        <th>number</th>
        <td>0</td>
        <td>1</td>
        <td>2</td>
        <td>3</td>
        <td>4</td>
        <td>5</td>
        <td>6</td>
        <td>7</td>
        <td>8</td>
        <td>9</td>
    </tr>
  
   </table> 
 
 

   

### Credits
All sequences if not stated differently are owned by Tianjin University and Automotive Data of China (Tianjin) Co., Ltd and are licensed under Creative Commons Attributions 4.0 License, see [Terms of Use].

### Citation


Please cite `VIL-100` in your publications if they help your research:

    @inproceedings{VIL100_ICCV_2021,
      author    = {?????},
      title     = {VIL-100: A New Dataset and A Baseline Model for Video Instance Lane Detection},
      booktitle = {The IEEE International Conference on Computer Vision (ICCV)},
      year      = {2021}
    }

### Terms of Use

`VIL-100` is released under the Creative Commons License:
  [CC BY-NC-SA](https://creativecommons.org/licenses/by-nc-sa/4.0/) <img src="https://licensebuttons.net/l/by-nc-sa/4.0/88x31.png"/>


In synthesis, users of the data are free to:

1. **Share** - copy and redistribute the material in any medium or format.
2. **Adapt** - remix, transform, and build upon the material.

The licensor cannot revoke these freedoms as long as you follow the license terms.

### Contacts

- [Yujun Zhang](https://github.com/yujun0-0) < yujunzhang@tju.edu.cn > 

- Lei Zhu < lzhu@cse.cuhk.edu.hk > 

- Wei Feng < wfeng@ieee.org > 

