from dermclass_api.app import create_app
from dermclass_api.config import ProductionConfig, DevelopmentConfig

# TODO: CHANGE CONFIG!!!
application = create_app(
    config_object=DevelopmentConfig)

if __name__ == '__main__':
    application.run()
