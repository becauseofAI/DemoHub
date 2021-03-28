from imagenet_classifier import get_prediction

if __name__ == '__main__':
    image_path = 'test4.jpg'
    result = get_prediction(image_path)
    print(result)
