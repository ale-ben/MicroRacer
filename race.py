import tracks
from SACModel import SAC

sac_base = SAC()
sac_trained = SAC()

cars = [SAC(), SAC(model_name="custom_sac")]

cars_models = [model.get_actor_model() for model in cars]

tracks.newrun(cars_models)