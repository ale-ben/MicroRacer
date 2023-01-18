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

for car in scoreboard:
    if car["completion"] == 1:
        print(
            f"car {car['car']} ({cars[car['car']-1].get_name()}) finished in {car['place']} position."
        )
    else:
        print(
            f"car {car['car']} ({cars[car['car']-1].get_name()}) did not finish with completion code {car['completion']} ({completion_codes[car['completion']]})."
        )
