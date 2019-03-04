from flask import Flask, render_template, Response
from camera_node.main import CameraNode

app = Flask(__name__)

@app.route('/')
def index():
    return render_template('index.html')

def color_gen(camera_node):
    while True:
        color_frame = camera_node.get_color_frame()
        yield (b'--frame\r\n'
               b'Content-Type: image/jpeg\r\n\r\n' + color_frame + b'\r\n\r\n')

@app.route('/color_feed')
def color_feed():
    return Response(color_gen(CameraNode()),
                    mimetype='multipart/x-mixed-replace; boundary=frame')


if __name__ == '__main__':
    app.run(host='0.0.0.0', debug=True)