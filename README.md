<h1>Dermclass</h1>

Dermclass is a Master thesis project, aimed to solve problems of skin disease classification, using various data types.
To accomplish this goal, it uses several Machine Learning techniques and technologies to create REST API.
As an input, Dermclass can take up to three types of data with symptoms of disease: structured (tabular), text or image. 

To accomplish it's goals, project is splat into 2 main packages: dermclass_models and dermclass_api.

<h2>Dermclass_models</h2>
Models - main component, consisting of multiple models to be served via REST API. It's splat into 3 parts, each
 corresponding to type of input data used in it:

* Structured - First submodule created for modeling tabular data, with use of sklearn. To perform it the model is chosen
by using Bayesian Hyperparameter Tuning with Optuna and chosen between the most popular model types.

* Image - Component which uses deep learning network called EfficientNet and transfer learning to classify photos of
disease symptoms, with Tensorflow backend.

* Text - Hybrid submodule which can use either sklearn with TF-IDF or Huggingface transformers DistilBERT to perform the
classification. This element of ML structure, also uses fine tuning to let model learn specific medical text descriptions.
  
<h2>Dermclass_API</h2>  

REST API - second main component used to efficiently deploy the project as a web application in production environment.
To accomplish it's goals it uses several technologies like:


* Flask, flask-restful - main framework for creating the web application code.

* Gunicorn AKA Green Unicorn' - Python WSGI HTTP Server for UNIX

* SQLAlchemy + SQLlite - the Python SQL toolkit and Object Relational Mapper that provides power and flexibility of SQL in Python.
Combined with Marshmallow it also serves validation purposes. 

* Docker - platform that uses isolated and self contained  Docker containers for serving OS-level virtualization. 

* CircleCI - continuous integration and delivery tool.

* Heroku - platform as a service (PaaS) that enables building, running, and operaing of the applications in the cloud.

