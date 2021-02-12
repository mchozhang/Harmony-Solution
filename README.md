# Harmony Solution
Deep Q-learning agent to solve the game puzzle [harmony](https://master.dg9tray1uvpm6.amplifyapp.com/),
using `tensorflow-agents`.

## Run the model
### Run as RESTful Web API
```
# start flask app at port 5000
python app.py
```
The API can be accessed [online](https://harmony-solution.heroluapp.com/) or locally at `localhost:5000`, post the level and grid data and receive the action,
see the [sample request data](#sample-api-data). 

### Train a specify game level
```
# train level 30
python game/train.py 30

# train level 1 to 10
python game/train.py 1 10
```

### Evaluate the a policy of a level
```
# print the sequence of actions of level 11
python game eval_policy 11
```

### Development Environment
```                            
# install tensorflow-agents, numpy and so forth
pip install -r requirements.txt

# or build and run in docker
docker build -t harmony . && docker run -p 5000:5000 harmony
```

## Implementation
* Double Deep Q-learning agent
* Deep Q Network
* Uniform replay buffer
* Mask for legal actions

## Deployment
Deployed at heroku container
```
# push image
heroku login
heroku container:login
heroku container:push web -a harmony-solution

# deploy
heroku container:release web -a harmony-solution
```

## Sample API Data <a name="sample-api-data"></a>
online host: [https://harmony-solution.heroluapp.com/](https://harmony-solution.heroluapp.com/)

using curl to test
```bash
curl --location --request GET 'https://harmony-solution.herokuapp.com/action' \
--header 'Content-Type: application/json' \
--data-raw '{
    "level": 25,
    "grid":
    [
                [
                    {
                        "col": 0,
                        "row": 0,
                        "steps": 4,
                        "targetRow": 3
                    },
                    {
                        "col": 1,
                        "row": 0,
                        "steps": 1,
                        "targetRow": 0
                    },
                    {
                        "col": 2,
                        "row": 0,
                        "steps": 3,
                        "targetRow": 1
                    },
                    {
                        "col": 3,
                        "row": 0,
                        "steps": 1,
                        "targetRow": 0
                    },
                    {
                        "col": 4,
                        "row": 0,
                        "steps": 2,
                        "targetRow": 4
                    }
                ],
                [
                    {
                        "col": 0,
                        "row": 1,
                        "steps": 3,
                        "targetRow": 4
                    },
                    {
                        "col": 1,
                        "row": 1,
                        "steps": 1,
                        "targetRow": 1
                    },
                    {
                        "col": 2,
                        "row": 1,
                        "steps": 1,
                        "targetRow": 1
                    },
                    {
                        "col": 3,
                        "row": 1,
                        "steps": 3,
                        "targetRow": 2
                    },
                    {
                        "col": 4,
                        "row": 1,
                        "steps": 2,
                        "targetRow": 0
                    }
                ],
                [
                    {
                        "col": 0,
                        "row": 2,
                        "steps": 2,
                        "targetRow": 2
                    },
                    {
                        "col": 1,
                        "row": 2,
                        "steps": 2,
                        "targetRow": 3
                    },
                    {
                        "col": 2,
                        "row": 2,
                        "steps": 2,
                        "targetRow": 0
                    },
                    {
                        "col": 3,
                        "row": 2,
                        "steps": 3,
                        "targetRow": 2
                    },
                    {
                        "col": 4,
                        "row": 2,
                        "steps": 3,
                        "targetRow": 2
                    }
                ],
                [
                    {
                        "col": 0,
                        "row": 3,
                        "steps": 4,
                        "targetRow": 3
                    },
                    {
                        "col": 1,
                        "row": 3,
                        "steps": 2,
                        "targetRow": 3
                    },
                    {
                        "col": 2,
                        "row": 3,
                        "steps": 3,
                        "targetRow": 4
                    },
                    {
                        "col": 3,
                        "row": 3,
                        "steps": 3,
                        "targetRow": 4
                    },
                    {
                        "col": 4,
                        "row": 3,
                        "steps": 2,
                        "targetRow": 2
                    }
                ],
                [
                    {
                        "col": 0,
                        "row": 4,
                        "steps": 3,
                        "targetRow": 3
                    },
                    {
                        "col": 1,
                        "row": 4,
                        "steps": 2,
                        "targetRow": 0
                    },
                    {
                        "col": 2,
                        "row": 4,
                        "steps": 3,
                        "targetRow": 4
                    },
                    {
                        "col": 3,
                        "row": 4,
                        "steps": 2,
                        "targetRow": 1
                    },
                    {
                        "col": 4,
                        "row": 4,
                        "steps": 1,
                        "targetRow": 1
                    }
                ]
            ]
}'
```

response:
```
{
    "action": [
        4,
        0,
        4,
        1
    ],
    "status": "success"
}
```