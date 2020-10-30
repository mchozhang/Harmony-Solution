#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Run the policy of a specific level
"""
import os
import sys
from flask import Flask, jsonify, request
from game.game_env import GameEnv
from game.eval_policy import get_action_from_policy
from game import utils

# app init
app = Flask(__name__, instance_relative_config=True)


@app.route('/', methods=['POST'])
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


if __name__ == '__main__':
    utils.init()
    utils.load_trained_policies()
    app.run()
