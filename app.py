import base64
import io

import cv2
import numpy as np
# from PIL.Image import Image
from PIL import Image
from flask import Flask, request, jsonify
from WCT.stylize import main
import boto3
from botocore.exceptions import NoCredentialsError
# from flask.ext.uuid import FlaskUUID
import uuid
plagCount=0
ACCESS_KEY = 'AKIAXVFMVVXL22DN3O76'
SECRET_KEY = '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'
PLAGUE="FALSE"
app = Flask(__name__)
app.config['MAX_CONTENT_LENGTH'] = 160000 * 102400 * 102430


@app.route('/')
def hello_world():
    # main()
    return 'Hello World!'


@app.route('/gendesign', methods=['POST'])
def gen_design():
    imageBase64_1=request.json['picture1']
    imageBase64_2 = request.json['picture2']
    # ['picture1', 'picture2', 'range1', 'range2', 'range3', 'range4', 'range5']
    img1 = toRGB(stringToImage(imageBase64_1.split('base64,')[1]))
    img2 = toRGB(stringToImage(imageBase64_2.split('base64,')[1]))
    cv2.imwrite('mask1.jpg',img1)
    cv2.imwrite('0resize.jpg',img2)
    main()

    stylized_image=cv2.imread("WCT/outputs/mask1_0resize.jpg")
    retval, buffer = cv2.imencode('.jpg', stylized_image)
    jpg_as_text = (base64.b64encode(buffer)).decode()

    print(type(jpg_as_text))
    print(request.json.keys())
    return {
        'msg': 'returnig design!',
        "status": True,
        'data':{
            'image': jpg_as_text
        }
    }

def plagueChecker(PLAGUE):
    if (PLAGUE == "FALSE"):
        # plagCount=plagCount+1
        PLAGUE = "TRUE"
        return {
            'msg': 'No plague found',
            'status': True
        }
    else:
        return {
            'msg': 'No plague found',
            'status': True
        }

@app.route('/checkplag', methods=['POST'])
def check_plague():
    img1 = request.json['picture1']
    img2 = request.json['picture2']
    # print(request.json.key)

    return {
        'msg': 'No plague found',
        'status': True
    }


def upload_to_aws(local_file, bucket, s3_file):
    s3 = boto3.client('s3', aws_access_key_id=ACCESS_KEY,
                      aws_secret_access_key=SECRET_KEY)

    try:
        s3.upload_file(local_file, bucket, local_file)
        print("Upload Successful")
        return True
    except FileNotFoundError:
        print("The file was not found")
        return False
    except NoCredentialsError:
        print("Credentials not available")
        return False


def stringToImage(base64_string):
    imgdata = base64.b64decode(base64_string)
    return Image.open(io.BytesIO(imgdata))


def toRGB(image):
    return cv2.cvtColor(np.array(image), cv2.COLOR_BGR2RGB)


@app.route('/uploadfile', methods=['POST'])
def upload_to_s3():
    print("uploading ")
    filename = str(uuid.uuid1())
    imgBase64 = request.json['picture1']
    print(imgBase64)
    imgBase64=imgBase64.split('base64,')[1]
    img = toRGB(stringToImage(imgBase64))

    cv2.imwrite('AWSTEMP/' + filename + '.png', img)
    upload_to_aws("AWSTEMP/" + filename + '.png', 'emotif', filename)
    return {"msg": "https://emotif.s3.amazonaws.com/" +"AWSTEMP/" +filename + '.png',
            "status": True
            }



@app.route('/abdullahapi', methods=['POST'])
def prcessImg():
    img=request.json['img']
    action=request.json['action']
    print(action)
    img = toRGB(stringToImage(img))
    cv2.imwrite("AWSTEMP/"+str(action)+".jpg",img)

    return 'cheeta'
if __name__ == '__main__':
    # main()
    app.run()
