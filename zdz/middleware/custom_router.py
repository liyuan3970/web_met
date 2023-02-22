from rest_framework.routers import SimpleRouter


class StandardRouter(SimpleRouter):
    def __init__(self, trailing_slash="/?"):
        super(StandardRouter, self).__init__()
        self.trailing_slash = trailing_slash
