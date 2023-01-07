import tracks
from SAC_model import SAC
from PPO_model import PPO
from TD3_model import TD3

cars = [SAC(), TD3(), PPO()]

cars_models = [model.get_actor_model() for model in cars]

scoreboard = tracks.newrun(cars_models)

completion_codes = [
    "", "finished", "off track", "wrong direction", "under speed limit"
]

for i, car in enumerate(scoreboard):
    if car["completion"] == 1:
        print(
            f"car {i} ({cars[i].get_name()}) finished in {car['place']} position."
        )
    else:
        print(
            f"car {i} ({cars[i].get_name()}) did not finish with completion code {car['completion']} ({completion_codes[car['completion']]})."
        )
