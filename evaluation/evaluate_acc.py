import numpy as np
from sklearn.linear_model import LinearRegression
import json, os
from skimage.io import imread

class LaneEval(object):
    lr = LinearRegression()
    pixel_thresh = 20
    pt_thresh = 0.85

    @staticmethod
    def get_angle(xs, y_samples):
        xs, ys = xs[xs >= 0], y_samples[xs >= 0]
        if len(xs) > 1:
            LaneEval.lr.fit(ys[:, None], xs)
            k = LaneEval.lr.coef_[0]
            theta = np.arctan(k)
        else:
            theta = 0
        return theta

    @staticmethod
    def line_accuracy(pred, gt, thresh):
        pred = np.array([p if p >= 0 else -100 for p in pred])
        gt = np.array([g if g >= 0 else -100 for g in gt])
        return np.sum(np.where(np.abs(pred - gt) < thresh, 1., 0.)) / len(gt)


    @staticmethod
    def get_pred_lanes(filename):
        img = imread(filename, as_gray=True)

        # According to the input image, the corresponding curve of each lane line is obtained
        img_gray = np.unique(img[np.where(img > 0)])
        y_pred = [(img.shape[0] - np.where(img == i)[0] - 1).tolist() for i in img_gray]
        x_pred = [(np.where(img == j)[1] + 1).tolist() for j in img_gray]
        param = [np.polyfit(y_pred[k], x_pred[k], 2).tolist() for k in range(len(x_pred))]
        return param, img.shape[0]


    @staticmethod
    def get_gt_lanes(gt_dir, filename, height):
        gt_json = json.load(open(os.path.join(gt_dir, filename))).get('annotations')['lane']
        img_height = height
        lanex_points = []
        laney_points = []
        for i in gt_json:
            for key, value in i.items():
                if key == 'points' and value != []:
                    lanex = []
                    laney = []
                    for item in value:
                        lanex.append(item[0])
                        laney.append(img_height - item[1])
                    lanex_points.append(lanex)
                    laney_points.append(laney)
        return lanex_points,laney_points

    @staticmethod
    def calculate_results(param, gtx, gty):
        angles = [LaneEval.get_angle(np.array(gtx[i]), np.array(gty[i])) for i in range(len(gty))]
        threshs = [LaneEval.pixel_thresh / np.cos(angle) for angle in angles]
        line_accs = []
        fp, fn = 0., 0.
        matched = 0.

        for index, (x_gts,thresh) in enumerate(zip(gtx, threshs)):
            accs = []
            for x_preds in param:
                x_pred =  (x_preds[0] * np.array(gty[index]) * np.array(gty[index]) + x_preds[1] * np.array(gty[index]) + x_preds[2]).tolist()
                accs.append(LaneEval.line_accuracy(np.array(x_pred), np.array(x_gts), thresh))
            # print(accs)
            max_acc = np.max(accs) if len(accs) > 0 else 0.
            if max_acc < LaneEval.pt_thresh:
                fn += 1
            else:
                matched += 1
            line_accs.append(max_acc)
        fp = len(param) - matched
        if len(gtx) > 8 and fn > 0:
            fn -= 1
        s = sum(line_accs)
        if len(gtx) > 8:
            s -= min(line_accs)
        return s / max(min(8.0, len(gtx)), 1.), fp / len(param) if len(param) > 0 else 0., fn / max(min(len(gtx), 8.), 1.)


    @staticmethod
    def calculate_return(pre_dir_name, json_dir_name):
        Preditction = pre_dir_name
        Json = json_dir_name
        num, accuracy, fp, fn = 0., 0., 0., 0.
        list_preditction = os.listdir(Preditction)
        list_preditction.sort()
        for filename in list_preditction:
            pred_files = os.listdir(os.path.join(Preditction, filename))
            json_files = os.listdir(os.path.join(Json, filename))
            pred_files.sort()
            json_files.sort()

            for pfile, jfile in zip(pred_files, json_files):
                pfile_name = os.path.join(Preditction, filename, pfile)
                param, height = LaneEval.get_pred_lanes(pfile_name)
                print('pred_image_name:', pfile_name)
                print('json_file_name:', os.path.join(Json, filename, jfile))
                lanex_points, laney_points = LaneEval.get_gt_lanes(os.path.join(Json, filename), jfile, height)

                try:
                    a, p, n = LaneEval.calculate_results(param, lanex_points, laney_points)
                except BaseException as e:
                    raise Exception('Format of lanes error.')
                accuracy += a
                fp += p
                fn += n
                num += 1


        accuracy = accuracy / num
        fp = fp / num
        fn = fn / num
        return accuracy, fp, fn



if __name__ == '__main__':

    # pre_dir_name is the path to your output dir
    # pre_dir_name = '/home/ubuntu/Wmq/Task/LaneDetection/New_BaseLines/VisTR-little-data/output_vil'
    # pre_dir_name = '/media/ubuntu/HDD_2TB/wmq/RESA/results/png_results(299.pth)'
    # pre_dir_name = '/home/ubuntu/Wmq/Task/LaneDetection/New_BaseLines/LaneATT/experiments/laneatt_r18_vil/png_results/epoch_100'
    pre_dir_name = '/home/ubuntu/Wmq/Task/LaneDetection/New_BaseLines/LaneAF/results/pics/show/lane_img'

    # json_dir_name is the path of Data_ROOT/Json
    json_dir_name = '/home/ubuntu/Wmq/Task/LaneDetection/Data/LaneVideoData/Json_ori'

    accuracy, fp, fn = LaneEval.calculate_return(pre_dir_name, json_dir_name)
    print('accuracy:', accuracy)
    print('fp:', fp)
    print('fn:', fn)






