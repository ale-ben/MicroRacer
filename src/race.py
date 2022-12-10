import tracks
from SAC_model import SAC
from PPO_model import PPO

cars = [SAC(), PPO()]

cars_models = [model.get_actor_model() for model in cars]

tracks.newrun(cars_models)