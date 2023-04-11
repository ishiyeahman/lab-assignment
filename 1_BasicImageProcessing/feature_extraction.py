import cv2
IMG_DIR = '/home/rish/projects/beginner/image/'


def get_sift_keypoints(img, gray):
    sift = cv2.SIFT_create()
    kp = sift.detect(img, None)
    img = cv2.drawKeypoints(gray, kp, img)
    return img

def get_akaze_keypoints(img, gray):
    akaze = cv2.AKAZE_create()
    kp = akaze.detect(img, None)
    img = cv2.drawKeypoints(gray, kp, img)
    return img

def main():
    # read image
    label = 'Person2'
    file_name= label + '.png'
    img = cv2.imread(IMG_DIR + file_name + '' , 1)
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    img_shift = get_sift_keypoints(img, gray)
    cv2.imwrite(IMG_DIR + f'out/keypoints_shift_{label}.jpg', img_shift)


    img_akaze = get_akaze_keypoints(img, gray)
    cv2.imwrite(IMG_DIR + f'out/keypoints_akaze_{label}.jpg', img_akaze)

    

if __name__ == '__main__':
    main()