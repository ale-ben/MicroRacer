import tracks
from SAC_model import SAC

cars = [SAC(model_name="baseline_weights/sac"), SAC()]

cars_models = [model.get_actor_model() for model in cars]

tracks.newrun(cars_models)