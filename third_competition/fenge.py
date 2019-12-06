from PIL import Image
import os

captcha_word = {
        '0': 0, '1': 1, '2': 2, '3': 3, '4': 4, '5': 5, '6': 6, '7': 7, '8': 8, '9': 9,
        'q': 10, 'w': 11, 'e': 12, 'r': 13, 't': 14, 'y': 15, 'u': 16, 'i': 17, 'o': 18, 'p': 19, 'a': 20, 's': 21,
        'd': 22, 'f': 23, 'g': 24, 'h': 25, 'j': 26, 'k': 27, 'l': 28, 'z': 29, 'x': 30, 'c': 31, 'v': 32, 'b': 33,
        'n': 34, 'm': 35,
        'Q': 36, 'W': 37, 'E': 38, 'R': 39, 'T': 40, 'Y': 41, 'U': 42, 'I': 43, 'O': 44, 'P': 45, 'A': 46, 'S': 47,
        'D': 48, 'F': 49, 'G': 50, 'H': 51, 'J': 52, 'K': 53, 'L': 54, 'Z': 55, 'X': 56, 'C': 57, 'V': 58, 'B': 59,
        'N': 60, 'M': 61
    }

def get_crop_imgs(img):
    """
    按照图片的特点,进行切割,这个要根据具体的验证码来进行工作. # 见原理图
    :param img:
    :return:
    """
    child_img_list = []
    for i in range(5):
        x =i *30 # 见原理图
        y = 0
        child_img = img.crop((x, y, x + 30, y + 30))
        #plt.show(child_img)
        child_img_list.append(child_img)
    return child_img_list

def save(child_img_list, target,count):
    for i in range(5):
        path = './picture/' + str(captcha_word[target[i]])
        print(path)
        if not os.path.exists(path):
            os.mkdir(path)
        
        child_img_list[i].save(path  +'/'+ count+str(i)+"&"+ str(captcha_word[target[i]]) + '.jpg', 'JPEG')
        
def tset_save(child_img_list, target,count):
    for i in range(5):
        path = './t_picture/' + target
        print(path)
        if not os.path.exists(path):
            os.mkdir(path)
        
        child_img_list[i].save(path  +'/'+ target+'_'+str(i) + '.jpg', 'JPEG')

def read_image(root):
    images = os.listdir(root)
    images = [os.path.join(root, image) for image in images if image.endswith('.jpg')]
    for i in range(len(images)):
        target = images[i].split(os.sep)[-1].split('.jpg')[0]
        print(target)
        print(images[i])
        img = Image.open(images[i])
        child_img_list = get_crop_imgs(img)
        #save(child_img_list, target,target+str(i))
        tset_save(child_img_list, target,target+str(i))


#read_image('D:\\大三上\\机器学习框架\\third_competition\\train\\train')
read_image('D:\\大三上\\机器学习框架\\third_competition\\test\\test')