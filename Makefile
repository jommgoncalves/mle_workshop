docker_mlflow_version = v1.0.0
docker_train_version = v1.0.0
docker_api_version = v1.0.0

docker-login:
	docker login -u jommgoncalves

docker-mlflow-build:
	docker build --tag jommgoncalves/mlflow:$(docker_mlflow_version) --rm --progress=auto -f Dockerfile_mlflow .
	#docker tag jommgoncalves/mlflow:$(docker_mlflow_version) jommgoncalves/$(docker_mlflow_version)
	docker push jommgoncalves/mlflow:$(docker_mlflow_version)

docker-train-build:
	docker build -t jommgoncalves/model_train:$(docker_train_version) --rm --progress=auto -f Dockerfile_train .
	#docker tag jommgoncalves/model_train:$(docker_train_version) jommgoncalves/model_train:$(docker_train_version)
	docker push jommgoncalves/model_train:$(docker_train_version)

docker-api-build:
	docker build -t jommgoncalves/api:$(docker_api_version) --rm --progress=auto -f Dockerfile_api .
	#docker tag jommgoncalves/api:$(docker_api_version) jommgoncalves/api:$(docker_api_version)
	docker push jommgoncalves/api:$(docker_api_version)

docker-images: docker-mlflow-build docker-train-build docker-api-build

minikube-install:
	curl -LO https://storage.googleapis.com/minikube/releases/latest/minikube-linux-amd64
	sudo install minikube-linux-amd64 /usr/local/bin/minikube

minikube-rebuild: minikube-delete minikube-setup docker-images minikube-create-all minikube-dashboard

minikube-reset: minikube-delete-all docker-images minikube-create-all

minikube-delete:
	minikube delete

minikube-setup:
	minikube config set driver docker
	minikube start
	kubectl get po -A
	alias kubectl="minikube kubectl --"

minikube-dashboard:
	minikube dashboard --url

minikube-tunnel:
	minikube tunnel

minikube-mlflow-delete:
	kubectl delete deployment mlflow-deployment
	kubectl delete service mlflow-service

minikube-mlflow-create:
	kubectl create -f k8s/mlflow.yaml

minikube-train-delete:
	kubectl delete deployment train-deployment

minikube-train-create:
	kubectl apply -f k8s/train.yaml

minikube-api-delete:
	kubectl delete deployment api-deployment
	kubectl delete service api-service

minikube-api-create:
	kubectl create -f k8s/api.yaml

minikube-minio-create:
	kubectl create -f k8s/minio.yaml

minikube-minio-delete:
	#kubectl delete pvc minio-pvc
	kubectl delete deployment minio-deployment
	kubectl delete service minio-service

minikube-delete-all: minikube-minio-delete minikube-mlflow-delete minikube-train-delete minikube-api-delete

minikube-create-all: minikube-minio-create minikube-mlflow-create minikube-train-create minikube-api-create

minikube-overview:
	kubectl get all
