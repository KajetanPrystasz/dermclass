from dermclass_api.app import create_app
from dermclass_api.config import ProductionConfig

application = create_app(
    config_object=ProductionConfig)

if __name__ == '__main__':
    application.run()
