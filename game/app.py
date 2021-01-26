#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Run the policy of a specific level
"""
from flask import Flask, jsonify, request
from flask_cors import CORS, cross_origin
from eval_policy import get_action_from_policy
import utils
import os

# app init
app = Flask(__name__, instance_relative_config=True)
CORS(app)

@app.route('/', methods=['POST'])
@cross_origin()
def index():
    result = {
        "status": "success",
    }
    try:
        # parse data from requests
        # grid is a 2D list json object
        r = request
        grid = request.json.get('grid', "")
        level = int(request.json.get('level', 1))
        action = get_action_from_policy(level, grid)
        result['action'] = action
    except Exception as e:
        result["status"] = "failure"
        result["err_msg"] = str(e)

    return jsonify(result)


@app.route('/', methods=['GET'])
def home():
    return "harmony solution"


if __name__ == '__main__':
    utils.init()
    utils.load_trained_policies()
    # heroku will assign a random port to the environment variable PORT
    port = os.environ.get('PORT', 5000)
    app.run(host='0.0.0.0', port=port)
