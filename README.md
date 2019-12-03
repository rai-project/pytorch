# MLModelScope PyTorch Agent

[![Build Status](https://travis-ci.org/rai-project/pytorch.svg?branch=master)](https://travis-ci.org/rai-project/pytorch)
[![Build Status](https://dev.azure.com/dakkak/rai/_apis/build/status/pytorch)](https://dev.azure.com/dakkak/rai/_build/latest?definitionId=14)
[![Go Report Card](https://goreportcard.com/badge/github.com/rai-project/pytorch)](https://goreportcard.com/report/github.com/rai-project/pytorch)
[![License](https://img.shields.io/badge/License-Apache%202.0-blue.svg)](https://opensource.org/licenses/Apache-2.0)

[![](https://images.microbadger.com/badges/version/carml/pytorch:ppc64le-gpu-latest.svg)](https://microbadger.com/images/carml/pytorch:ppc64le-gpu-latest> 'Get your own version badge on microbadger.com') [![](https://images.microbadger.com/badges/version/carml/pytorch:ppc64le-cpu-latest.svg)](https://microbadger.com/images/carml/pytorch:ppc64le-cpu-latest 'Get your own version badge on microbadger.com') [![](https://images.microbadger.com/badges/version/carml/pytorch:amd64-cpu-latest.svg)](https://microbadger.com/images/carml/pytorch:amd64-cpu-latest 'Get your own version badge on microbadger.com') [![](https://images.microbadger.com/badges/version/carml/pytorch:amd64-gpu-latest.svg)](https://microbadger.com/images/carml/pytorch:amd64-gpu-latest 'Get your own version badge on microbadger.com')

This is the Pytorch agent for [MLModelScope](mlmodelscope.org), an open-source framework and hardware agnostic, extensible and customizable platform for evaluating and profiling ML models across datasets / frameworks / systems, and within AI application pipelines.

Currently it has most of the models from Pytorch Model Zoo built in, plus many models acquired from public repositories. Although the agent supports different modalities including Object Detection and Image Enhancement, most of the built-in models are for Image Classification. More built-in models are coming.
One can evaluate the **~50** models on any system of interest with either local Pytorch installation or Pytorch docker images.

Check out [MLModelScope](mlmodelscope.org) and welcome to contribute.

# Bare Minimum Installation


## Prerequsite System Library Installation
We first discuss a bare minimum pytorch-agent installation without the tracing and profiling capabilities. To make this work, you will need to have the following system libraries preinstalled in your system.

- The CUDA library (required)
- The CUPTI library (required)
- The Pytorch C++ (libtorch) library (required)
- The libjpeg-turbo library (optional, but preferred)

### The CUDA Library

Please refer to Nvidia CUDA library installation on this. Find the localation of your local CUDA installation, which is typically at `/usr/local/cuda/`, and setup the path to the `libcublas.so` library. Place the following in either your `~/.bashrc` or `~/.zshrc` file:

```
export LIBRARY_PATH=$LIBRARY_PATH:/usr/local/cuda/lib64
export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:/usr/local/cuda/lib64
```

### The CUPTI Library

Please refer to Nvidia CUPTI library installation on this. Find the localation of your local CUPTI installation, which is typically at `/usr/local/cuda/extras/CUPTI`, and setup the path to the `libcupti.so` library. Place the following in either your `~/.bashrc` or `~/.zshrc` file:

```
export LIBRARY_PATH=$LIBRARY_PATH:/usr/local/cuda/extras/CUPTI/lib64
export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:/usr/local/cuda/extras/CUPTI/lib64
```

### The Pytorch C++ (libtorch) Library

The Pytorch C++ library is required for our Pytorch Go package. If you want to use Pytorch Docker Images (e.g. NVIDIA GPU CLOUD (NGC)) instead, skip this step for now and refer to our later section on this.

You can download pre-built Pytorch C++ (libtorch) library from [Pytorch](https://pytorch.org). Choose `Pytorch Build = Stable (1.3)`, `Your OS = <fill>`, `Package = LibTorch`, `Language = C++` and `CUDA = <fill>`. Download `Pre-cxx11 ABI` or `cxx11 ABI` version based on local gcc/g++ version. 

Extract the downloaded archive to `/opt/libtorch/`.

```
tar -C /opt/libtorch -xzf (downloaded file)
```

Configure the linker environmental variables since the Pytorch C++ library is extracted to a non-system directory. Place the following in either your `~/.bashrc` or `~/.zshrc` file

Linux

```
export LIBRARY_PATH=$LIBRARY_PATH:/opt/libtorch/lib
export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:/opt/libtorch/lib
```

macOS

```
export LIBRARY_PATH=$LIBRARY_PATH:/opt/libtorch/lib
export DYLD_LIBRARY_PATH=$DYLD_LIBRARY_PATH:/opt/libtorch/lib
```

You can test the installed Pytorch C++ library using an [example C++ program](https://pytorch.org/tutorials/advanced/cpp_frontend.html), although we suggest running an example in `github.com/rai-project/go-pytorch` as per its documentation to confirm library installation.

To build the Pytorch C++ library from source, refer to [github.com/rai-project/go-pytorch/scripts](https://github.com/rai-project/go-pytorch).

### Use libjpeg-turbo for Image Preprocessing

[libjpeg-turbo](https://github.com/libjpeg-turbo/libjpeg-turbo) is a JPEG image codec that uses SIMD instructions (MMX, SSE2, AVX2, NEON, AltiVec) to accelerate baseline JPEG compression and decompression. It outperforms libjpeg by a significant amount.

You need libjpeg installed.
```
sudo apt-get install libjpeg-dev
```
The default is to use libjpeg-turbo, to opt-out, use build tag `nolibjpeg`.

To install libjpeg-turbo, refer to [libjpeg-turbo](https://github.com/libjpeg-turbo/libjpeg-turbo/releases).

Linux

```
  export TURBO_VER=2.0.2
  cd /tmp
  wget https://cfhcable.dl.sourceforge.net/project/libjpeg-turbo/${TURBO_VER}/libjpeg-turbo-official_${TURBO_VER}_amd64.deb
  sudo dpkg -i libjpeg-turbo-official_${TURBO_VER}_amd64.deb
```

macOS

```
brew install jpeg-turbo
```


## Installation of GO for Compilation

Since we use `go` for MLModelScope development, it's required to have `go` installed in your system before proceeding.

Please follow [Installing Go Compiler](https://github.com/rai-project/rai/blob/master/docs/developer_guide.md) to have `go` installed.


## Bare Minimum Pytorch-agent Installation

Download and install the MLModelScope Pytorch Agent by running the following command in any location, assuming you have installed `go` following the above instruction.

```
go get -v github.com/rai-project/pytorch

```

You can then install the dependency packages through `go get`.

```
cd $GOPATH/src/github.com/rai-project/pytorch
go get -u -v ./...
```

An alternative to install the dependency packages is to use [Dep](https://github.com/golang/dep).

```
dep ensure -v
```

This installs the dependency in `vendor/`.

The CGO interface passes go pointers to the C API. There is an error in the CGO runtime. We can disable the error by placing

```
export GODEBUG=cgocheck=0
```

in your `~/.bashrc` or `~/.zshrc` file and then run either `source ~/.bashrc` or `source ~/.zshrc`


Build the Pytorch agent with GPU enabled
```
cd $GOPATH/src/github.com/rai-project/pytorch/pytorch-agent
go build
```

Build the Pytorch agent without GPU or libjpeg-turbo
```
cd $GOPATH/src/github.com/rai-project/pytorch/pytorch-agent
go build -tags="nogpu nolibjpeg"
```

If everything is successful, you should have an executable `pytorch-agent` binary in the current directory.


### Configuration Setup

To run the agent, you need to setup the correct configuration file for the agent. Some of the information may not make perfect sense for all testing scenarios, but they are required and will be needed for later stage testing. Some of the port numbers as specified below can be changed depending on your later setup for those service.

So let's just set them up as is, and worry about the detailed configuration parameter values later.

You must have a `carml` config file called `.carml_config.yml` under your home directory. An example config file `carml_config.yml.example` is in [github.com/rai-project/MLModelScope](https://github.com/rai-project/MLModelScope) . You can move it to `~/.carml_config.yml`.

The following configuration file can be placed in `$HOME/.carml_config.yml` or can be specified via the `--config="path"` option.

```yaml
app:
  name: carml
  debug: true
  verbose: true
  tempdir: ~/data/carml
registry:
  provider: consul
  endpoints:
    - localhost:8500
  timeout: 20s
  serializer: jsonpb
database:
  provider: mongodb
  endpoints:
    - localhost
tracer:
  enabled: true
  provider: jaeger
  endpoints:
    - localhost:9411
  level: FULL_TRACE
logger:
  hooks:
    - syslog
```

## Test Installation

With the configuration and the above bare minimumn installation, you should be ready to test the installation and see how things works.

Here are a few examples. First, make sure we are in the right location
```
cd $GOPATH/src/github.com/rai-project/pytorch/pytorch-agent
```

To see a list of help
```
./pytorch-agent -h
```

To see a list of models that we can run with this agent
```
./pytorch-agent info models
```

To run an inference using the default DNN model `alexnet` with a default input image.

```
./pytorch-agent predict urls --model_name TorchVision_Alexnet --profile=false --publish=false
```

The above `--profile=false --publish=false` command parameters tell the agent that we do not want to use profiling capability and publish the results, as we haven't installed the MongoDB database to store profiling data and the tracer service to accept tracing information.

# External Service Installation to Enable Tracing and Profiling

We now discuss how to install a few external services that make the agent fully useful in terms of collecting tracing and profiling data.

## External Srvices

MLModelScope relies on a few external services. These services provide tracing, registry, and database servers.

These services can be installed and enabled in different ways. We discuss how we use `docker` below to show how this can be done. You can also not use `docker` but install those services from either binaries or source codes directly.

### Installing Docker

Refer to [Install Docker](https://docs.docker.com/install/).

On Ubuntu, an easy way is using

```
curl -fsSL get.docker.com -o get-docker.sh | sudo sh
sudo usermod -aG docker $USER
```

On macOS, [intsall Docker Destop](https://docs.docker.com/docker-for-mac/install/)


### Starting Trace Server

This service is required.

- On x86 (e.g. intel) machines, start [jaeger](http://jaeger.readthedocs.io/en/latest/getting_started/) by

```
docker run -d -e COLLECTOR_ZIPKIN_HTTP_PORT=9411 -p5775:5775/udp -p6831:6831/udp -p6832:6832/udp \
  -p5778:5778 -p16686:16686 -p14268:14268 -p9411:9411 jaegertracing/all-in-one:latest
```

- On ppc64le (e.g. minsky) machines, start [jaeger](http://jaeger.readthedocs.io/en/latest/getting_started/) machine by

```
docker run -d -e COLLECTOR_ZIPKIN_HTTP_PORT=9411 -p5775:5775/udp -p6831:6831/udp -p6832:6832/udp \
  -p5778:5778 -p16686:16686 -p14268:14268 -p9411:9411 carml/jaeger:ppc64le-latest
```

The trace server runs on http://localhost:16686

### Starting Registry Server

This service is not required if using TensorFlow-agent for local evaluation.

- On x86 (e.g. intel) machines, start [consul](https://hub.docker.com/_/consul/) by

```
docker run -p 8500:8500 -p 8600:8600 -d consul
```

- On ppc64le (e.g. minsky) machines, start [consul](https://hub.docker.com/_/consul/) by

```
docker run -p 8500:8500 -p 8600:8600 -d carml/consul:ppc64le-latest
```

The registry server runs on http://localhost:8500

### Starting Database Server

This service is not required if not using database to publish evaluation results.

- On x86 (e.g. intel) machines, start [mongodb](https://hub.docker.com/_/mongo/) by

```
docker run -p 27017:27017 --restart always -d mongo:3.0
```

You can also mount the database volume to a local directory using

```
docker run -p 27017:27017 --restart always -d  -v $HOME/data/carml/mongo:/data/db mongo:3.0
```

### Configuration

You must have a `carml` config file called `.carml_config.yml` under your home directory. An example config file `~/.carml_config.yml` is already discussed above. Please update the port numbers for the above external services accordingly if you decide to choose a different ports above.


### Testing

The testing steps are very similar to those testing we discussed above, except that you can now safely use both the profiling and publishing services.

# Alternative ways to Install Pytorch-agent using a Pytorch Docker Image

Instead of using a local Pytorch library to install the MLModelScope `pytorch-agent`, we can also use a pytorch docker image to start this process.

## Local Setup

You need to follow the above similar procedures to setup `go` and get all the related `rai-project` projects in your local go development environment.

```
go get -v github.com/rai-project/pytorch
cd $GOPATH/src/github.com/rai-project/pytorch
go get -u -v ./...
```

You also need to have the `.carml_config.yml` configuraiton file as discussed above to be placed under $HOME as `.carml_config.yml`

You can also setup all the external services as discussed above in your local host machine where you plan to use the Tensorflow Docker container.

## Setup within Pytorch Docker Image (TO BE UPDATED)

Continue if you have

* installed all the dependencies
* downloaded carml_config_example.yml to $HOME as .carml_config.yml
* launched docker external services on the host machine of the docker container you are going to use

, otherwise read above

Assuming you want to use the NGC Pytorch docker image. Here is an example on how to do this:

```
nvidia-docker run --shm-size=1g --ulimit memlock=-1 --ulimit stack=67108864 -it --privileged=true --network host \
-v $GOPATH:/workspace/go1.12/global \
-v $GOROOT:/workspace/go1.12_root \
-v ~/.carml_config.yml:/root/.carml_config.yml \
-v ~/data/carml:/root/data/carml \
nvcr.io/nvidia/tensorflow:19.06-py2
```

NOTE: The SHMEM allocation limit is set to the default of 64MB.  This may be
   insufficient for TensorFlow.  NVIDIA recommends the use of the following flags:
   ```nvidia-docker run --shm-size=1g --ulimit memlock=-1 --ulimit stack=67108864 ...```

Within the container, set up the environment so that the agent can find the Pytorch C++ library.

```
export GOPATH=/workspace/go1.12/global
export GOROOT=/workspace/go1.12_root
export PATH=$GOROOT/bin:$PATH

ln -s /usr/local/lib/tensorflow/libtensorflow_cc.so /usr/local/lib/tensorflow/libtensorflow.so
export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:/usr/local/lib/tensorflow
export CGO_LDFLAGS="${CGO_LDFLAGS} -L /usr/local/lib/tensorflow/"

export PATH=$PATH:$(go env GOPATH)/bin
export GODEBUG=cgocheck=0
```

Build the Pytorch agent with GPU enabled
```
cd $GOPATH/src/github.com/rai-project/pytorch/pytorch-agent
go build
```

Build the Pytorch agent without GPU or libjpeg-turbo
```
cd $GOPATH/src/github.com/rai-project/pytorch/pytorch-agent
go build -tags="nogpu nolibjpeg"
```


# Use the Agent with the [MLModelScope Web UI](https://github.com/rai-project/mlmodelscope)

```
./pytorch-agent serve -l -d -v
```

Refer to [TODO] to run the web UI to interact with the agent.

# Use the Agent through Command Line

Run ```./pytorch-agent -h``` to list the available commands.

Run ```./pytorch-agent info models``` to list the available models.

Run ```./pytorch-agent predict``` to evaluate a model. This runs the default evaluation.
```./pytorch-agent predict -h``` shows the available flags you can set.

An example run is

```
./pytorch-agent predict urls --trace_level=FRAMEWORK_TRACE --model_name=TorchVision_AlexNet
```

Refer to [TODO] to run the web UI to interact with the agent.

# Notes on installing Pytorch C++ from source (TO BE UPDATED)
