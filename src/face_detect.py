import cv2

CASCADE_FILEPATH = '/Users/wonjunetai/src/github.com/vendlytics/models/cascade/haarcascade_frontalface_default.xml'


def face_crop(image):
    """Returns a cropped face

    Arguments:
        image {numpy.array} -- image where face may exist

    Returns:
        numpy.array, None -- image with face, else None
    """

    face = face_detect(image)
    if face is None:
        return None
    x, y, w, h = face
    return image[y:y + h, x:x + w]


def face_detect(image):
    """Returns face positions (min_x, min_y, max_x, max_y)

    Arguments:
        image {numpy.array} -- image where face may exist

    Returns:
        numpy.array, None -- position of face, if it exists
    """

    if not hasattr(face_detect, 'cascade'):
        cascade = cv2.CascadeClassifier(CASCADE_FILEPATH)
    gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
    faces = cascade.detectMultiScale(
        gray,
        scaleFactor=1.1,
        minNeighbors=5,
        minSize=(30, 30))
    
    num_faces = len(faces)
    if num_faces == 0:
        return None

    x, y, w, h = get_largest_face(faces)
    return [x, y, x + w, y + h]


def get_largest_face(faces):
    def area(_, __, w, h):
        return w * h
    return max(faces, key=lambda rect: area(*rect))


if __name__ == '__main__':
    from src.utils import read_bag
    import sys
    import scipy.misc
    import numpy as np
    image = np.zeros((200, 100, 3), dtype=np.uint8)
    crop = face_crop(image)
    assert crop is None
    image, _ = next(read_bag(sys.argv[1]))
    crop = face_crop(image)
    assert crop is not None
    scipy.misc.imshow(crop)
