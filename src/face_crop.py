import cv2

CASCADE_FILEPATH = '../models/cascade/haarcascade_frontalface_default.xml'


# returns crop np.array
def face_crop(image):
    face = face_detect(image)
    if face is None:
        return None
    x, y, w, h = face
    return image[y:y+h, x:x+w]

# returns face (x, y, w, h)
def face_detect(image):
    if not hasattr(face_detect, 'cascade'):
        cascade = cv2.CascadeClassifier(CASCADE_FILEPATH)
    gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
    faces = cascade.detectMultiScale(
        gray,
        scaleFactor=1.1,
        minNeighbors=5,
        minSize=(30, 30))
    if len(faces) == 0:
        return None
    return get_largest_face(faces)

def get_largest_face(faces):
    def area(_, __, w, h):
        return w * h
    return max(faces, key=lambda rect: area(*rect))

if __name__ == '__main__':
    from read_bag import read_bag
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

