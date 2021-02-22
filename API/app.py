from flask import Flask, jsonify,request,url_for,redirect
import sys
sys.path.append('..')
from networks.models import build_model
from matting import pred
import requests
import numpy as np
import cv2
import base64
from PIL import Image
import io 

app = Flask(__name__)

def get_array(arg):
    return np.array(arg).astype('uint8')

class Matting_Args:
    def __init__(self):
        self.encoder = 'resnet50_GN_WS'
        self.decoder = 'fba_decoder'
        self.weights = '../models/FBA.pth'
        
args = Matting_Args()

matting_model = build_model(args)
matting_model.eval();

def get_response(new_bg,data):
    image = get_array(data.get('image'))
    response = requests.post('http://127.0.0.1:3000/',json = data)
    if response.status_code == 406:
        return jsonify({'output':image.tolist()})
    h,w,_ = image.shape
    trimap = get_array(response.json()['trimap'])
    fg, bg, alpha = pred(image/255.0,trimap,matting_model)

    alpha = Image.fromarray((alpha*255).astype(np.uint8))
    combined = Image.fromarray((image).astype(np.uint8))
    combined.putalpha(alpha)

    #combined = bg.astype('uint8')
    #combined = (alpha[...,None]*image).astype('uint8')

    #rgba = np.dstack((image, np.zeros(image.shape[:-1])))
    #rgba *= alpha[...,None]
    #combined = rgba.astype('uint8')

    #rgba = np.dstack((image, np.zeros(image.shape[:-1])))

    #out_img = np.zeros((h,w,4),dtype=np.uint8)
    #out_img = alpha[...,None]*image

    #out_img[:,:,:] = alpha[...,None]*image

    #combined = (alpha * rgba).astype('uint8')

    #combined = ((alpha[...,None]*image)).astype('uint8') #+ ((1-alpha)[...,None]*cv2.resize(new_bg,(w,h))).astype('uint8')

    #combined = np.concatenate((alpha[...,None]*image, 1 - alpha[...,None]), axis=2).astype('uint8')

    #b_channel, g_channel, r_channel = cv2.split(image)
    #img_BGRA = cv2.merge((b_channel, g_channel, r_channel, alpha[...,None]))
    #combined = img_BGRA.astype('uint8')

    #return jsonify({'output':combined.tolist()})
    return jsonify({'output':np.asarray(combined, dtype='uint8').tolist()})

@app.route('/with_bg',methods=['POST'])
def extraction():
    data = request.get_json()
    new_bg = get_array(data.get('bg'))
    return get_response(new_bg,data)

@app.route('/',methods=["POST"])
def extraction_without_bg():
    data = request.get_json()
    new_bg = cv2.imread('1.png')#[:,:,::-1]
    return get_response(new_bg,data)
        
    
if __name__ == '__main__':
    app.run(debug=True,threaded=True)
    
