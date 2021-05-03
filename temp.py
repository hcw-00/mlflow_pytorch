# curl -X POST -H "Content-Type:application/json; format=pandas-split" 
# --data '{"columns":["alcohol", "chlorides", ...],"data":[[12.8, 0.029, 0.48, 0.98, 6.2, 29, 3.33, 1.2, 0.39, 75, 0.66]]}' http://127.0.0.1:1234/invocations
# curl -X POST -H "Content-Type:application/json-numpy-split" -F "image=@D:\Dataset\REVIEW_BOE_HKC_WHTM\BOE_HKC_WHTM_210219\val\boe_open\ZHANGDAOLEIA7BAJ03_20191028113545_1353_101277_-445585_48_crop_512.jpg" http://127.0.0.1:1234/invocations
# curl -X POST -H "Content-Type:application/json-numpy-split" -d "@data.json" http://127.0.0.1:1234/invocations

import pandas as pd
import cv2
import base64
import requests
import torchvision.transforms as transforms
from PIL import Image
from flask import Flask, jsonify, request


def transform_image(image_bytes):
    my_transforms = transforms.Compose([transforms.Resize(255),
                                        transforms.CenterCrop(224),
                                        transforms.ToTensor(),
                                        transforms.Normalize(
                                            [0.485, 0.456, 0.406],
                                            [0.229, 0.224, 0.225])])
    image = Image.open(io.BytesIO(image_bytes))
    return my_transforms(image).unsqueeze(0)


img_path = r'D:\Dataset\REVIEW_BOE_HKC_WHTM\BOE_HKC_WHTM_210219\val\boe_open\ZHANGDAOLEIA7BAJ03_20191028113545_1353_101277_-445585_48_crop_512.jpg'
test_img = cv2.imread(img_path)
test_img = cv2.resize(test_img, (224,224))
files = {"index":[0], "data": [ base64.b64encode(cv2.imencode('.jpg', test_img)[1]).decode() ]}
df = pd.DataFrame.from_dict(files)

# send a random row from the test set to score
input_data = df.to_json()

headers = {'Content-Type': 'application/json'}


# #using the webservice scoring URI
# print(webservice.scoring_uri)
# http://4d08721e-3682-44e8-b5dc-26580989ef1e.westeurope.azurecontainer.io/score

scoring_uri = 'http://127.0.0.1:1234/invocations'


# def img2json(img_path):
#     file = request.files[img_path]
#     img_bytes = file.read()


# resp = requests.post(scoring_uri, input_data, headers=headers)

resp = requests.post(scoring_uri, headers = headers, json={"data": input_data})
print(resp.text)