# with help from https://github.com/jrosebr1/simple-keras-rest-api/blob/master/simple_request.py
# import the necessary packages
import requests
import argparse

def arg_parser():
    '''
    Description: Sets arguments for running this python file

    Input:  none

    Ouput:  args - the argument for an image filepath, which also indicates how to point to an image file
    '''
    # create a parser object
    parser = argparse.ArgumentParser(description='predict.py')
    # add an argument to the parser for supplying the predict.py app with an image filepath (an image of a Brittany is supplied by default)
    parser.add_argument('--image', help='Point to image file, for example: python predict.py --image images\Brittany_02625.jpg', default='./images/Brittany_02625.jpg')
    # assign the parser arguments to an args object
    args = parser.parse_args()

    return args

def main():
    '''
    Description: Gets the provided image path (or default path), posts a request to server by sending it the image path, and then prints the prediction result made by run_server.py

    Input:  none

    Ouput:  args - the argument for an image filepath, which also indicates how to point to an image file
    '''
    # initialize the parser arguments
    args = arg_parser()

    # initialize the URL along with the input image path
    URL = "http://localhost:4000/predict"
    IMAGE_PATH = args.image

    # load the input image path and construct the payload for the request
    payload = {"image": IMAGE_PATH}

    # submit the request
    print('\nMaking prediction...\n')
    r = requests.post(URL, json=payload).json()

    # make sure the request was sucessful
    if r["success"]:
        # display prediction
        print(r["predictions"][0])
        print(r["predictions"][1])

    # if the request is unsuccessful, display error
    else:
        print("Request failed")

if __name__ == '__main__':
    main()
