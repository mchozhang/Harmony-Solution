#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Run the policy of a specific level
"""
import os
from flask import Flask, jsonify, request
from flask_cors import CORS, cross_origin
from game.eval_policy import get_action_from_policy

# app init
app = Flask(__name__, instance_relative_config=True)
CORS(app)


@app.route('/action', methods=['GET'])
@cross_origin()
def get_action():
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
        print(str(e))

    return jsonify(result)


@app.route('/', methods=['GET'])
def index():
    return "harmony solution"


if __name__ == '__main__':
    # app.run(port=5000)

    # heroku will assign a random port to the environment variable PORT
    port = os.environ.get('PORT', 5000)
    app.run(host='0.0.0.0', port=port)
