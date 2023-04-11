import cv2
IMG_DIR = '/home/rish/projects/beginner/image/'

def main():
     # read image
    img_frame_0 = cv2.imread(IMG_DIR + 'Background_Subtraction_Tutorial_frame.png' , 1)
    img_frame_1 = cv2.imread(IMG_DIR + 'Background_Subtraction_Tutorial_frame_1.png' , 1)
    
    # subtraction
    subtracted = cv2.subtract(img_frame_0, img_frame_1)
    # get subtraction image
    cv2.imshow('sub', subtracted)
    # save subtraction image
    cv2.imwrite(IMG_DIR + 'out/subtracted.jpg', subtracted)


    subtracted_merge = cv2.add(subtracted, img_frame_0)
    # get subtraction image
    cv2.imshow('sub', subtracted_merge)
    # save subtraction image
    cv2.imwrite(IMG_DIR + 'out/subtracted_merge.jpg', subtracted_merge)




if __name__ == '__main__':
    main()
