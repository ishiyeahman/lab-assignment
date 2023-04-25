import cv2
import numpy as np 

IMG_DIR = '/home/rish/projects/beginner/image/'

def main():
    ext = '.pdf'

    # read image
    img_normal = cv2.imread(IMG_DIR + 'Background_Subtraction_Tutorial_frame.png' , 1)
    cv2.imwrite(IMG_DIR + 'out/ope_normal' + ext , img_normal)

    img_fliped = cv2.flip(img_normal, 0)
    cv2.imwrite(IMG_DIR + 'out/ope_flip' + ext, img_fliped)

    # resize zoom down
    img_zoom_down = cv2.resize(img_normal, (int(img_normal.shape[1] / 2), int(img_normal.shape[0] / 2)))
    cv2.imwrite(IMG_DIR + 'out/ope_zoom_down' + ext, img_zoom_down)

    # resize zoom up
    img_zoom_up = cv2.resize(img_normal, (int(img_normal.shape[1] * 2), int(img_normal.shape[0] * 2)))
    cv2.imwrite(IMG_DIR + 'out/ope_zoom_up' + ext, img_zoom_up)

    img_normal_merged = np.vstack((img_normal, img_fliped))
    img_zoom_down_merged = np.vstack((img_zoom_down, img_zoom_down, img_zoom_down, img_zoom_down))
    merged = np.hstack((img_zoom_down_merged, img_normal_merged, img_zoom_up))
    cv2.imwrite(IMG_DIR + 'out/ope_merged' + ext, merged)


    # binarization
    img_bin = cv2.imread(IMG_DIR + 'Background_Subtraction_Tutorial_frame.png' , 0)
    ret2, img_bin = cv2.threshold(img_bin, 0, 255, cv2.THRESH_OTSU)
    cv2.imwrite(IMG_DIR + 'out/ope_binarization' + ext, img_bin)


if __name__ == '__main__':
    main()
