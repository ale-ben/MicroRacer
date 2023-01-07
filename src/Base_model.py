class Base_model:
    def __init__(self, model_name):
        self.model_name = model_name

    def get_actor_model(self):
        return self.actor_model

    def get_name(self):
        return self.model_name