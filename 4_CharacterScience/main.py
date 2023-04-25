from keras.layers import Dense, Activation, Concatenate, GlobalAveragePooling2D
# from keras.callbacks import EarlyStopping, ModelCheckpoint, CSVLogger
from keras.callbacks import EarlyStopping, CSVLogger
from tensorflow.keras.callbacks import  ModelCheckpoint 
from keras.preprocessing import image as Kimage
from keras.applications import MobileNet
from keras.optimizers import Adam
from keras.models import Model
from keras import Input
# from tensorflow.python.keras.preprocessing.image import ImageDataGenerator
from keras.preprocessing.image import ImageDataGenerator
from matplotlib.offsetbox import OffsetImage, AnnotationBbox
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix
from sklearn.manifold import TSNE
from PIL import Image, ImageOps, ImageDraw
from pathlib import Path
import keras.backend as K
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import numpy as np
import random
import shutil
import cv2
import os

from tensorflow.keras.utils import load_img

import warnings
warnings.filterwarnings('ignore')


# config = K.tf.ConfigProto()
config = K.tf.compat.v1.ConfigProto()
config.gpu_options.allow_growth = True
# sess = K.tf.Session(config=config)
sess = K.tf.compat.v1.Session(config=config)
K.set_session(sess)
fontImage_path = Path("dataset").resolve()/"fontimage_preprocessed"
fontTag_path = Path("dataset").resolve()/"taglabel"
alphLower = [chr(i) for i in range(97, 97+26)]
alphUpper = [chr(i)*2 for i in range(65, 65+26)]


class ModelCheckpointPerEpoch(ModelCheckpoint):
    def __init__(self, filepath, verbose=2, save_weights_only=False, period=1):
        # super(ModelCheckpoint, self).__init__()
        super().__init__(filepath, verbose=2, save_weights_only=False, period=1)
        self.verbose = verbose
        self.filepath = filepath
        self.save_weights_only = save_weights_only
        self.period = period
        self.epochs_since_last_save = 0
        self.epoch_count = 0

        # add by me
        # self.save_freq = 'epoch'

    def on_epoch_end(self, epoch, logs=None):
        logs = logs or {}
        self.epochs_since_last_save += 1
        self.epoch_count += 1
        if self.epochs_since_last_save >= self.period:
            self.epochs_since_last_save = 0
            filepath = self.filepath.format(epoch=self.epoch_count, **logs)
            filepath = filepath.split("model")[0] + "model_epoch" + str(self.epoch_count) \
                       + filepath.split("model")[1]  ## added
            print('\nEpoch %05d: saving model to %s' % (epoch + 1, filepath))
            if self.save_weights_only:
                self.model.save_weights(filepath, overwrite=True)
            else:
                self.model.save(filepath, overwrite=True)


def Grad_Cam(model, x, layer_name):

    # 前処理
    X = np.expand_dims(x, axis=0)
    X = X.astype('float32')
    preprocessed_input = X / 255.0
    input_shape = model.input_shape[1:]

    # 予測クラスの算出
    predictions = model.predict(preprocessed_input)
    class_idx = np.argmax(predictions[0])
    class_output = model.output[:, class_idx]

    #  勾配を取得
    conv_output = model.get_layer(layer_name).output   # layer_nameのレイヤーのアウトプット
    grads = K.gradients(class_output, conv_output)[0]  # gradients(loss, variables) で、variablesのlossに関しての勾配を返す
    gradient_function = K.function([model.input], [conv_output, grads])  # model.inputを入力すると、conv_outputとgradsを出力する関数

    output, grads_val = gradient_function([preprocessed_input])
    output, grads_val = output[0], grads_val[0]

    # 重みを平均化して、レイヤーのアウトプットに乗じる
    weights = np.mean(grads_val, axis=(0, 1))
    cam = np.dot(output, weights)

    # 画像化してヒートマップにして合成
    cam = cv2.resize(cam, (input_shape[0], input_shape[1]), cv2.INTER_LINEAR) # 画像サイズは200で処理したので
    cam = np.maximum(cam, 0)
    cam = cam / cam.max()

    jetcam = cv2.applyColorMap(np.uint8(255 * cam), cv2.COLORMAP_JET)  # モノクロ画像に疑似的に色をつける
    jetcam = cv2.cvtColor(jetcam, cv2.COLOR_BGR2RGB)  # 色をRGBに変換
    jetcam = (np.float32(jetcam) + x / 2)   # もとの画像に合成

    return jetcam


def gradcam_plus_plus(model, img_array, layer_name):
    img_array = np.expand_dims(img_array, axis=0)
    img_array = img_array.astype('float32')
    img_array = img_array / 255.0

    cls = np.argmax(model.predict(img_array))
    y_c = model.output[0, cls]
    conv_output = model.get_layer(name=layer_name).output
    grads = K.gradients(y_c, conv_output)[0]
    # grads = normalize(grads)
    first = K.exp(y_c) * grads
    second = K.exp(y_c) * grads * grads
    third = K.exp(y_c) * grads * grads * grads
    gradient_function = K.function([model.input], [y_c, first, second, third, conv_output, grads])
    y_c, conv_first_grad, conv_second_grad, conv_third_grad, conv_output, grads_val = gradient_function([img_array])
    global_sum = np.sum(conv_output[0].reshape((-1, conv_first_grad[0].shape[2])), axis=0)
    alpha_num = conv_second_grad[0]
    alpha_denom = conv_second_grad[0] * 2.0 + conv_third_grad[0] * global_sum.reshape((1, 1, conv_first_grad[0].shape[2]))
    alpha_denom = np.where(alpha_denom != 0.0, alpha_denom, np.ones(alpha_denom.shape))
    alphas = alpha_num / alpha_denom
    weights = np.maximum(conv_first_grad[0], 0.0)
    alpha_normalization_constant = np.sum(np.sum(alphas, axis=0), axis=0)
    alphas /= alpha_normalization_constant.reshape((1, 1, conv_first_grad[0].shape[2]))
    deep_linearization_weights = np.sum((weights * alphas).reshape((-1, conv_first_grad[0].shape[2])), axis=0)
    cam = np.sum(deep_linearization_weights * conv_output[0], axis=2)
    cam = np.maximum(cam, 0)  # Passing through ReLU
    cam /= np.max(cam)  # scale 0 to 1.0
    return cam


def sigmoid_for_superimpose(x, a=50, b=0.5, c=1):
    return c / (1 + np.exp(-a * (x - b)))


def superimpose(original_img_path, cam, emphasize=False):
    img_bgr = cv2.imread(original_img_path)
    heatmap = cv2.resize(cam, (img_bgr.shape[1], img_bgr.shape[0]))
    if emphasize:
        heatmap = sigmoid_for_superimpose(heatmap)
    heatmap = np.uint8(255 * heatmap)
    heatmap = cv2.applyColorMap(heatmap, cv2.COLORMAP_JET)
    hif = 0.8
    superimposed_img = heatmap * hif + img_bgr
    superimposed_img = np.minimum(superimposed_img, 255.0).astype(np.uint8)  # scale 0 to 255
    superimposed_img_rgb = cv2.cvtColor(superimposed_img, cv2.COLOR_BGR2RGB)
    return superimposed_img_rgb


def plot_history_result(history, result_path, targetTag):
    plt.figure()
    plt.plot(history.history['loss'], label="loss for training")
    plt.plot(history.history['val_loss'], label="loss for validation")
    plt.title('loss')
    plt.xlabel('epoch')
    plt.ylabel('loss')
    plt.legend(loc='best')
    plt.grid()
    plt.savefig(str(result_path/"loss_{}".format(targetTag)))
    plt.close()

    plt.figure()

    plt.plot(history.history['accuracy'], label="accuracy for training")
    plt.plot(history.history['val_accuracy'], label="accuracy for validation")
    # plt.plot(history.history['acc'], label="accuracy for training")
    # plt.plot(history.history['val_acc'], label="accuracy for validation")
    plt.title('accuracy')
    plt.xlabel('epoch')
    plt.ylabel('accuracy')
    plt.legend(loc='best')
    plt.grid()
    plt.savefig(str(result_path/"accuracy_{}".format(targetTag)))
    plt.close()


def tsne_2d(x, y, figname, resize=True, alpha=1):
    if resize:
        x = x.reshape(x.shape[0], x.shape[1], x.shape[2])
        x = x.reshape(x.shape[0], -1)
    tsne = TSNE(n_components=2)
    proj = tsne.fit_transform(x)
    plt.figure(figsize=(10, 10))
    cmp = plt.get_cmap("Set1")
    for i, l in enumerate(sorted(list(set(y)))):
        select_flag = y == l
        plt_latent = proj[select_flag, :]
        plt.scatter(plt_latent[:, 0], plt_latent[:, 1], color=cmp(i), label=l, s=15, alpha=alpha)
    plt.legend(loc="best", borderpad=1)
    # plt.title(figname)
    plt.savefig(figname, bbox_inches="tight")
    plt.close()
    return proj


def imscatter(x, y, ylabel_list, image_list, figname, ax=None, zoom=0.2):
    def waku(photo, color):
        # 枠線の太さ
        waku_w2 = 30
        # 枠と写真を描画するキャンバス
        # 写真と枠線のサイズからサイズを計算
        canvas = Image.new('RGB', (photo.size[0] + waku_w2, photo.size[1] + waku_w2), (128, 128, 128))
        # 枠線を描く
        draw = ImageDraw.Draw(canvas)
        draw.line((0, 0, canvas.width, 0), fill=color, width=waku_w2)
        draw.line((canvas.width, 0, canvas.width, canvas.height), fill=color, width=waku_w2)
        draw.line((canvas.width, canvas.height, 0, canvas.height), fill=color, width=waku_w2)
        draw.line((0, canvas.height, 0, 0), fill=color, width=waku_w2)
        # canvasに写真をはめ込む
        canvas.paste(photo, (waku_w2 // 2, waku_w2 // 2, waku_w2 // 2 + photo.size[0], waku_w2 // 2 + photo.size[1]))
        return canvas

    if ax is None:
        plt.figure(figsize=(10, 10))
        ax = plt.gca()
    color_idx={"{}".format(x): i for i, x in enumerate(sorted(list(set(ylabel_list))))}
    cmp = plt.get_cmap("Set1")
    im_list = [OffsetImage(waku(photo=p, color=tuple(map(lambda x: int(x*255), cmp(color_idx[i])[:3]))),
                           zoom=zoom, cmap="gray")
               for i, p in zip(ylabel_list, image_list)]
    x, y = np.atleast_1d(x, y)
    artists = []
    for x0, y0, im in zip(x, y, im_list):
        ab = AnnotationBbox(im, (x0, y0), xycoords='data', frameon=False)
        artists.append(ax.add_artist(ab))

    ax.update_datalim(np.column_stack([x, y]))
    ax.autoscale()
    plt.savefig(figname, bbox_inches="tight")
    plt.close()
    return artists


def cm_sample_index(y_true, y_pred):
    TP = []
    FP = []
    TN = []
    FN = []

    for i, y in enumerate(zip(y_true, y_pred)):
        if y[0]==y[1]==1:
           TP.append(i)
        if y[1]==1 and y[0]!=y[1]:
           FP.append(i)
        if y[0]==y[1]==0:
           TN.append(i)
        if y[1]==0 and y[0]!=y[1]:
           FN.append(i)

    return (TP, FP, TN, FN)


def multiInputMN(input_shape, classes, weights, alphClass):
    base_model = MobileNet(input_shape=input_shape, classes=classes, weights=weights)
    convModel = Model(base_model.inputs, base_model.layers[-6].output, name="MobileNet")
    convModel.summary()
    input_list = [Input(input_shape, name="{}".format(alph)) for alph in alphClass]
    output_list = [convModel(input_n) for input_n in input_list]
    output = Concatenate()(output_list)
    output = Dense(units=classes, activation="softmax")(output)
    model = Model(input_list, output)
    model.summary()
    return model


def cleanup_directory(name):
    if os.path.isdir(name):
        shutil.rmtree(name)
    os.makedirs(name)


def prepare_data(df, targetTag, alphClass, result_path):

    def data_split(fn_all):
        fn_train, fn_test = train_test_split(fn_all, shuffle=True, random_state=0, test_size=0.2)
        fn_train, fn_val = train_test_split(fn_train, shuffle=True, random_state=0, test_size=0.25)
        return [fn_train, fn_val, fn_test]

    def save_img(fn_array, alphClass, path):
        if not path.exists():
            path.mkdir(parents=True)
        for fn in fn_array:
            for alph in alphClass:
                img_path = str(fontImage_path / "{}_{}.png".format(fn, alph))
                img = load_img(img_path, grayscale=True)
                img.save(str(path / "{}_{}.png".format(fn, alph)))

    # 結果のディレクトリ作成
    cleanup_directory(str(result_path))
    cleanup_directory(str(result_path / "weights"))
    cleanup_directory(str(result_path / "image" / "augImg"))

    # データ数を対象タグに揃える
    random.seed(0)
    target_fontNames = set(df[df["tagWord"] == targetTag]["fontName"].values)
    others_fontNames = set(df["fontName"].values) - target_fontNames
    target_fn = sorted(list(target_fontNames))
    others_fn = sorted(random.sample(list(others_fontNames), len(target_fn)))

    # クラスごとにtrain/val/test画像保存
    classes = ["others", targetTag]
    classes_fn = [others_fn, target_fn]
    for class_fn, class_name in zip(classes_fn, classes):
        for fn_array, fn_type in zip(data_split(class_fn), ["train", "val", "test"]):
            save_img(fn_array, alphClass, path=result_path / "image" / "{}".format(fn_type) / "{}".format(class_name))


def train_model(targetTag, result_path, model, batch_size, epochs, save_to_dir=False):

    train_dir = str(result_path / "image" / "train")
    val_dir = str(result_path / "image" / "val")
    augImg_dir = str(result_path / "image" / "augImg") if save_to_dir else None
    input_shape = model.input_shape[1:]
    classes = ["others", targetTag]

    train_datagen = ImageDataGenerator(
        rescale=1. / 255,
        width_shift_range=0.1,
        height_shift_range=0.1,
        zoom_range=0.1,
    )
    test_datagen = ImageDataGenerator(rescale=1. / 255)

    train_generator = train_datagen.flow_from_directory(
        train_dir,
        target_size=(input_shape[0], input_shape[1]),
        color_mode="grayscale",
        batch_size=batch_size,
        classes=classes,
        class_mode='categorical',
        shuffle=True,
        save_to_dir=augImg_dir
    )
    val_generator = test_datagen.flow_from_directory(
        val_dir,
        target_size=(input_shape[0], input_shape[1]),
        color_mode="grayscale",
        batch_size=batch_size,
        classes=classes,
        class_mode='categorical',
        shuffle=True,
    )

    history = model.fit_generator(train_generator,
                                  steps_per_epoch=train_generator.samples / batch_size,
                                  epochs=epochs,
                                  verbose=2,
                                  validation_data=val_generator,
                                  validation_steps=val_generator.samples / batch_size,
                                  callbacks=[CSVLogger(str(result_path / 'trainlog_{}.csv'.format(targetTag))),
                                             ModelCheckpointPerEpoch(period=1, save_weights_only=True,
                                                                     filepath=str(
                                                                         result_path / "weights" / "model.h5")),
                                             # ModelCheckpoint(monitor="val_loss", save_best_only=True, save_weights_only=True,
                                             #                 filepath=str(result_path / "weights" / "model.h5"))
                                             ]
                                  )
    plot_history_result(history=history, result_path=result_path, targetTag=targetTag)


def test_model(targetTag, model, result_path, batch_size,
               confusion_mat=False, tsne_features=False, cam_samples=None):

    input_shape = model.input_shape[1:]
    classes = ["others", targetTag]

    test_dir = str(result_path / "image" / "test")
    test_datagen = ImageDataGenerator(rescale=1. / 255)
    test_generator = test_datagen.flow_from_directory(
        test_dir,
        target_size=(input_shape[0], input_shape[1]),
        color_mode="grayscale",
        batch_size=batch_size,
        classes=classes,
        class_mode='categorical',
        shuffle=False,
    )

    true_classes = test_generator.classes
    test_img_num = test_generator.samples
    predict = model.predict_generator(test_generator, steps=test_img_num/batch_size)
    predict_classes = np.argmax(predict[:], 1)
    score = model.evaluate_generator(test_generator, steps=test_img_num/batch_size)

    if confusion_mat:
        cm = confusion_matrix(true_classes, predict_classes)
        cm = cm/test_img_num
        plt.figure(figsize=(2.8, 2.1))
        sns.heatmap(cm, cmap="coolwarm", annot=True, xticklabels=classes, yticklabels=classes)
        plt.xlabel("Predicted")
        plt.ylabel("True")
        plt.title("Tag={}\nSamples:{}, Loss:{:.2f}".format(targetTag, test_img_num, score[0]), fontsize=10)
        plt.savefig(str(result_path / 'cm_{}'.format(targetTag)), bbox_inches="tight")
        plt.close()

    if tsne_features or (cam_samples is not None):

        # TNなどを抽出
        TP, FP, TN, FN = cm_sample_index(y_true=true_classes, y_pred=predict_classes)

        # 特徴空間の可視化
        if tsne_features:

            # 入力テストデータ
            x_test = np.array([Kimage.img_to_array(load_img(os.path.join(test_dir, filename),
                                                                   grayscale=True,
                                                                   target_size=input_shape))
                               for filename in test_generator.filenames])

            # TNなどの平均画像を保存
            Kimage.array_to_img(np.mean(x_test[TP], axis=0)).save(str(result_path/"aveTP_{}.png".format(targetTag)))
            Kimage.array_to_img(np.mean(x_test[TN], axis=0)).save(str(result_path/"aveTN_{}.png".format(targetTag)))
            Kimage.array_to_img(np.mean(x_test[FP], axis=0)).save(str(result_path/"aveFP_{}.png".format(targetTag)))
            Kimage.array_to_img(np.mean(x_test[FN], axis=0)).save(str(result_path/"aveFN_{}.png".format(targetTag)))

            # 特徴抽出
            extract = Model(model.inputs, model.layers[-3].output)
            features = extract.predict(x_test[:, ], batch_size=batch_size)

            # TSNE
            label_tsneFeatures = np.array(["hoge" for x in range(len(true_classes))])
            label_tsneFeatures[TP] = "TP"
            label_tsneFeatures[TN] = "TN"
            label_tsneFeatures[FP] = "FP"
            label_tsneFeatures[FN] = "FN"

            points = tsne_2d(x=features,
                             y=label_tsneFeatures,
                             figname=str(result_path / "tsneClass_teatFeature_{}".format(targetTag)),
                             resize=False,
                             alpha=0.5)

            # 入力画像版のTSNE
            imscatter(x=points[:, 0],
                      y=points[:, 1],
                      ylabel_list=label_tsneFeatures,
                      image_list=[ImageOps.invert(Kimage.array_to_img(x, scale=True)) for x in x_test],
                      zoom=0.08,
                      figname=str(result_path / "tsneImg_teatFeature_{}".format(targetTag)))
        else:
            pass

        # gradcam
        if cam_samples is not None:
            cam_path = result_path / "image" / "gradcam"
            cleanup_directory(str(cam_path))
            x_test = np.array([filename for filename in test_generator.filenames])

            def cam_save(img_list, type):
                for i, img_path in enumerate(img_list):
                    x = Kimage.img_to_array(load_img(os.path.join(test_dir, img_path),
                                                            grayscale=True, target_size=input_shape))
                    picture = Grad_Cam(model=model, layer_name="conv_pw_9", x=x)
                    img = Kimage.array_to_img(picture)
                    img.save(str(cam_path/"test_{}{}_{}.png".format(type, i, img_path.replace("/", "_"))))
                    # y_cam = gradcam_plus_plus(model=model, layer_name="conv_pw_13", img_array=x)
                    # y_cam_emphasized = superimpose(os.path.join(test_dir, img_path), y_cam, emphasize=False)
                    # y_cam_emphasized = Image.fromarray(np.uint8(y_cam_emphasized))
                    # y_cam_emphasized.save(str(cam_path/"test_{}_{}{}_{}.png"
                    #                           .format(targetTag, type, i, img_path.replace("/", "_"))), 'PNG',
                    #                       quality=100, optimize=False)

            cam_save(random.sample(list(x_test[TN]), cam_samples), "TN")
            cam_save(random.sample(list(x_test[TP]), cam_samples), "TP")
            cam_save(random.sample(list(x_test[FN]), cam_samples), "FN")
            cam_save(random.sample(list(x_test[FP]), cam_samples), "FP")
        else:
            pass

    else:
        pass


if __name__ == "__main__":

    # ハイパーパラメータ
    targetTags = ["serif", "formal"]  # One-vs-OthersのOneのタグ
    img_len = 224  # 入力画像のサイズ(縦横)
    epochs = 5
    batch_size = 64
    alphClass = alphUpper[:1]  # 使用する画像のアルファベットのクラス
    lr = 0.0001
    base_result_path = Path("data").resolve() / "hoge2"  # 結果の保存先

    # model
    ssc = MobileNet(input_shape=(img_len, img_len, 1), weights=None, include_top=False)
    top_model = ssc.output
    top_model = GlobalAveragePooling2D()(top_model)
    top_model = Dense(2)(top_model)
    top_model = Activation(activation='softmax')(top_model)
    model = Model(inputs=ssc.input, outputs=top_model)
    model.compile(metrics=['accuracy'],
                  loss="categorical_crossentropy",
                  optimizer=Adam(lr=lr)
                  )

    for targetTag in targetTags:

        # ターゲット単語ごとの結果の保存先
        result_path = base_result_path / "{}".format(targetTag)

        # # train/val/testデータの準備
        prepare_data(df=pd.read_csv("fontName_tagWord.csv"), targetTag=targetTag, alphClass=alphClass,
                     result_path=result_path)

        # 学習
        train_model(targetTag=targetTag, result_path=result_path, model=model, batch_size=batch_size, epochs=epochs,
                    save_to_dir=False  # 水増しデータを保存する場合，save_to_dir=Trueに
                    )

        # パラメータ記録
        df_log = pd.read_csv(result_path / "trainlog_{}.csv".format(targetTag))
        best_epoch = df_log.sort_values(by="val_loss").reset_index(drop=True)["epoch"].values[0]
        with open(str(result_path / "params_{}.txt".format(targetTag)), mode='w') as f:
            s = "epochs: {}\nbatch_size: {}\nalphClass: {}\nlr: {}\nbest_epoch: {}" \
                .format(epochs, batch_size, ",".join(alphClass), lr, best_epoch)
            f.write(s)

        # テスト
        model.load_weights(str(result_path / "weights" / "model_epoch{}.h5".format(best_epoch)))
        test_model(targetTag=targetTag, result_path=result_path, batch_size=batch_size,
                   model=model,
                   confusion_mat=True,  # confusion_matrixを図で出力
                   tsne_features=True,  # 中間層の特徴分布を図で出力
                   cam_samples=2  # gradcamの例を出力
                   )
