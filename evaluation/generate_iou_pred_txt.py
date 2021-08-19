import os,json
import numpy as np
from skimage.io import imread, imshow


def generate_pred(pred_path, out_path, json_path):

    pred_dirs = os.listdir(pred_path) #['2_Road036_Trim003_frames','...']
    for pred_dir in pred_dirs: # 2_Road036_Trim003_frames

        print('generate sequence: ',pred_dir)
        pred_dir_path = os.path.join(pred_path, pred_dir)
        all_pred_files = os.listdir(pred_dir_path)
        all_pred_files.sort()
        all_json_files = os.listdir(os.path.join(json_path, pred_dir))
        all_json_files.sort()

        # out_pred_path
        if not os.path.exists(os.path.join(out_path, pred_dir)):
            os.makedirs(os.path.join(out_path, pred_dir))

        for index, img in enumerate(all_pred_files):
            img = imread(os.path.join(pred_dir_path, img), as_gray=True)
            img_gray = np.unique(img[np.where(img > 0)])
            y_pred = [(img.shape[0] - np.where(img == i)[0] - 1).tolist() for i in img_gray]
            x_pred = [(np.where(img == j)[1] + 1).tolist() for j in img_gray]
            param = [np.polyfit(y_pred[k], x_pred[k], 2).tolist() for k in range(len(x_pred))]

            out_file_txt_name = all_json_files[index].replace('jpg.json', 'lines.txt')
            with open(os.path.join(out_path, pred_dir, out_file_txt_name), "a+") as fp:
                for i, y in enumerate(y_pred):
                    txty = list(range(min(y), max(y), 3))
                    txtx = (param[i][0] * np.array(txty) * np.array(txty) + param[i][1] * np.array(txty) + param[i][2]).tolist()
                    if len(txtx) >= 2:
                        for index in range(len(txtx)):
                            if txtx[index] >= 0 and txtx[index] <= img.shape[1]:
                                fp.write('%d %d ' % (txtx[index], img.shape[0] - txty[index]))
                        fp.write('\n')
            fp.close()



if __name__ == "__main__":
    # change your pred_path at ${root}/${output}/${valset}
    pre_dir_name ='' # {pred_dir_name} is the path of your output results
    json_path = '../dataset/VIL100/Json' # {json_path} is the path of Data_ROOT/Json_ori
    pred_txt_path = './txt/pred_txt'
    if not os.path.exists(pred_txt_path):
        os.mkdir(pred_txt_path)
    generate_pred(pre_dir_name, pred_txt_path, json_path)








