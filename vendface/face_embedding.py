import tensorflow as tf
import numpy as np
from PIL import Image
from scipy.misc import imresize


class Facenet:
    def __init__(self):
        self.sess = Facenet._load_model()
        self.image_size = 160

    def image_to_embedding(self, image):
        # image: width x height x 3
        return self._run(image_batch=[self._preprocess(image)])

    def batch_image_to_embedding(self, images):
        # images: list of width x height x 3
        return self._run(image_batch=map(self._preprocess, images))

    def _run(self, image_batch):
        return self.sess.run(
            "import/embeddings:0",
            feed_dict={
                "import/image_batch:0": image_batch,
                "import/phase_train:0": False})

    def _preprocess(self, image):
        image = imresize(image, (self.image_size, self.image_size))
        return (image - 127.5) / 128.0

    @staticmethod
    def _load_model():
        # model from https://github.com/davidsandberg/facenet
        model_pb_path = "20180402-114759.pb"
        with tf.gfile.FastGFile(model_pb_path, "rb") as f:
            graph_def = tf.GraphDef()
            graph_def.ParseFromString(f.read())
            g_in = tf.import_graph_def(graph_def)
        return tf.Session(graph=g_in)


if __name__ == "__main__":
    fn = Facenet()
    image = np.empty((160, 160, 3))
    fn.image_to_embedding(image)
