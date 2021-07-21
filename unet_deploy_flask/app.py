from flask import Flask, render_template, request , send_file
from model import *
import os
from math import floor
import albumentations as A
from albumentations.pytorch import ToTensorV2
from PIL import Image
import numpy as np
import hashlib

app = Flask(__name__)

app.config['UPLOAD_FOLDER'] = 'uploads'
savefolder = './static/img/'

def load(checkpoint, model):
    print("==> laoding model")
    model.load_state_dict(checkpoint["state_dict"])

model = UNET(in_channels=3, out_channels=1)
load(torch.load("./checkpoint.pth.tar",map_location=torch.device('cpu')), model)

@app.route('/')
def index():
    return render_template('index.html')


@app.route('/about')
def about():
    return render_template('about.html')


@app.route('/infer', methods=['POST'])
def success():
    global savefolder
    if request.method == 'POST':
        f = request.files['file']
        saveLocation = f.filename
        saveLocation = savefolder + saveLocation
        f.save(saveLocation)
        img = Image.open(saveLocation)
        image = np.array(img.convert("RGB"))
        augmentations = transform(image=image)
        image = augmentations["image"]
        image = image.unsqueeze(0)
        pred = model(image)
        pred = (pred >0.5)
        pred = pred.squeeze(0).squeeze(0)
        image_nparray = pred.numpy()
        output_hash = hashlib.sha1(saveLocation.encode("UTF-8")).hexdigest()[:20]
        output_image = savefolder+output_hash+".jpeg"
        result_image = Image.fromarray(image_nparray)
        result_image.resize(img.size)
        result_image.save(output_image)
        return render_template('inference.html' , saveLocation=saveLocation , output_image=output_image)

@app.route('/download')
def downloadFile ():
    #For windows you need to use drive name [ex: F:/Example.pdf]
    path = "./assets/paper.pdf"
    return send_file(path, as_attachment=True)

transform = A.Compose(
        [
            A.Resize(height=512, width=512),
            A.Normalize(
                mean=[0.0, 0.0, 0.0],
                std=[1.0, 1.0, 1.0],
                max_pixel_value=255.0,
            ),
            ToTensorV2(),
        ],
    )



if __name__ == '__main__':
    app.debug = True
    port = int(os.environ.get("PORT", 8001))
    app.run(host='0.0.0.0', port=port, debug=True)
