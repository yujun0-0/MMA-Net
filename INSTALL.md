# Install

1. Clone the project

    ```
    git clone https://github.com/yujun0-0/MMA-Net
    cd MMA-Net
    ```

2. Create a conda virtual environment and activate it

    ```
    conda create -n mma-net python=3.6 -y
    conda activate mma-net
    ```

3. Install dependencies

    ```
    pip install -r requirements.txt
    ```
   
4. Data preparation

    To run the training and testing code, we require the following data organization format
    ```
    ${root}--
            |--${VIL100}
                |----JPEGImages
                |----Annotations
                |----Json
                |----data
                |------|-----db_info.yaml
                |------|-----test.txt
                |------|-----train.txt
    ```
    The `root` folder can be set in `options.py`.
    
5. Install **`CULane evaluation tools`** <font color="red">(Only required for evaluating mIoU)</font>

    If you just want to train a model or make a demo, this tool is not necessary and you can skip this step. If you want to get the evaluation results on CULane, you should install this tool.

    This tools requires OpenCV C++. Please follow [here](https://docs.opencv.org/master/d7/d9f/tutorial_linux_install.html) to install OpenCV C++. ***When you build OpenCV, remove the paths of anaconda from PATH or it will be failed.***
    ```Shell
    # First you need to install OpenCV C++. 
    # After installation, make a soft link of OpenCV include path.

    ln -s /usr/local/include/opencv4/opencv2 /usr/local/include/opencv2
    ```
    We provide three kinds of complie pipelines to build the evaluation tool of CULane.
evaluate_acc
    Option 1:

    ```Shell
    cd evaluation/culane
    make
    ```

    Option 2:
    ```Shell
    cd evaluation/culane
    mkdir build && cd build
    cmake ..
    make
    mv culane_evaluator ../evaluate
    ```

    For Windows user:
    ```Shell
    mkdir build-vs2017
    cd build-vs2017
    cmake .. -G "Visual Studio 15 2017 Win64"
    cmake --build . --config Release  
    # or, open the "xxx.sln" file by Visual Studio and click build button
    move culane_evaluator ../evaluate