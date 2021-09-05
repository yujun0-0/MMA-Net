## Preparation
1. Please see **dataset/README.md** to get more details about our datasets-VIL100

2. Please see **INSTALL.md** to install environment and evaluation tools

3. Before training, we should download datasets-VIL100 and models
    - datasets (Baidu Disk) : https://pan.baidu.com/s/1NkP_5LMLTn6qsu9pSbyi0g - iy16       
    - models (Baidu Disk) : https://pan.baidu.com/s/1_o13TBbTf258-j7iACDS2Q - sgh2 
    
    (Google Drive) https://drive.google.com/drive/folders/178_SSeQ4M1hI3BrTonhiTrpOWTEAenLE
        
        - The first training stage loads the model: **initial_STM**
        - The second training stage loads the model: **resume STM** and **resume ATT**

4. Put them under this structure
    
    ```
      MMA-Net
           |----INSTALL.md
           |----README.md
           |----dataset
           |------|-----VIL100
           |----models
           |----evaluation
           |----options.py
           |----libs
           |----requirements.txt
           |----train.py
           |----test.py
     ```
    


## Training and Testing
1. To train the MMA network, run following command
    ```python3
    python3 train.py --gpu ${GPU-IDS}
    ```
2. To test the MMA network, run following command
    ```python3
    python3 test.py
    ```
    The test results will be saved as indexed png file at `${root}/${output}/${valset}`.

    Additionally, you can modify some setting parameters in `options.py` to change training configuration.

## Evaluation

1. generate **`accuracy`**, **`fp`**, **`fp`**

    ```
    python evaluate_acc.py      # Please modify `pre_dir_name` and `json_dir_name` in evaluate_acc.py
    ```

2. Install **`CULane evaluation tools`**, please see INSTALL.md

3. generate **`F`**, **`mIoU`evaluate_acc** <font color="red"> after the CULane evaluation tools are installed</font>

    1. all pred txt files will be generated under `MMA-Net/evaluation/txt/pred_txt` after this step
    
        ```
        python generate_iou_pred_txt.py      # Please modify `pre_dir_name` and `json_path` in  `generate_iou_pred_txt.py`
        ```
    
    2. `results_MMA` and `temp_MMA` will be generated under `MMA-Net/evaluation/txt/results_txt` after this step.
    
        `results_MMA`: evaluation results of each sequence
        
        `temp_MMA`: temporary files generated during evaluation, you can ignore them
        
        ```    
        python evaluate_iou.py      # `data_root` should be set as your VIL-100 dataset path in `evaluate_iou.py`
        ```
    
    3. **<font color="red">Attention!! if you want to evaluation results one more time, please delete all folders/files under `MMA-Net/evaluation/txt/results_txt` </font>.**



