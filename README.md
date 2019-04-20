# MLModelScope PyTorch Agent

[![Build Status](https://travis-ci.org/rai-project/pytorch.svg?branch=master)](https://travis-ci.org/rai-project/pytorch)
[![Build Status](https://dev.azure.com/dakkak/rai/_apis/build/status/pytorch)](https://dev.azure.com/dakkak/rai/_build/latest?definitionId=14)
[![Go Report Card](https://goreportcard.com/badge/github.com/rai-project/pytorch)](https://goreportcard.com/report/github.com/rai-project/pytorch)
[![License](https://img.shields.io/badge/License-Apache%202.0-blue.svg)](https://opensource.org/licenses/Apache-2.0)

[![](https://images.microbadger.com/badges/version/carml/pytorch:ppc64le-gpu-latest.svg)](https://microbadger.com/images/carml/pytorch:ppc64le-gpu-latest> 'Get your own version badge on microbadger.com') [![](https://images.microbadger.com/badges/version/carml/pytorch:ppc64le-cpu-latest.svg)](https://microbadger.com/images/carml/pytorch:ppc64le-cpu-latest 'Get your own version badge on microbadger.com') [![](https://images.microbadger.com/badges/version/carml/pytorch:amd64-cpu-latest.svg)](https://microbadger.com/images/carml/pytorch:amd64-cpu-latest 'Get your own version badge on microbadger.com') [![](https://images.microbadger.com/badges/version/carml/pytorch:amd64-gpu-latest.svg)](https://microbadger.com/images/carml/pytorch:amd64-gpu-latest 'Get your own version badge on microbadger.com')

## Installation

Download and install the MLModelScope PyTorch Agent:

```
go get -v github.com/rai-project/pytorch

```

The agent requires PyTorch C++ library (Libtorch).


## Libtorch Installation

### Pre-built binaries

Download the relevant `Libtorch` pre-built binary available on [Pytorch website](https://pytorch.org). Note that we provide the option of profiling through pytorch's in-built autograd profiler. Incidentally, Pytorch C++ frontend does not have access to the autograd profiler as per release `1.0.1`. Kindly download nightly build post March 24th 2019 to enable the profiling. Without profiling, our codebase should be compatible with prior versions.

### Build from source

Refer to `$GOPATH/src/github.com/rai-project/go-pytorch/dockerfiles` to know how to build `Libtorch` from source. Note that one can also use `build_libtorch.py` script provided as part of the Pytorch repository to do the same.

Place the extracted/built library to `/opt/libtorch/`.

Configure the linker environmental variables since Libtorch library has been extracted to a non-system directory. Place the following in either your `~/.bashrc` or `~/.zshrc` file

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

### Build from source using python pip

```
pip3 install torch torchvision
```

or

```
conda install pytorch-nightly -c pytorch
```

then build using

```
go build -tags=nogpu -tags=python
```

#### Go packages

You can install the dependency through `go get`.

```
cd $GOPATH/src/github.com/rai-project/pytorch
go get -u -v ./...
```

Or use [Dep](https://github.com/golang/dep).

```
dep ensure -v
```

This installs the dependency in `vendor/`.

#### libjpeg-turbo

[libjpeg-turbo](https://github.com/libjpeg-turbo/libjpeg-turbo) is a JPEG image codec that uses SIMD instructions (MMX, SSE2, AVX2, NEON, AltiVec) to accelerate baseline JPEG compression and decompression. It outperforms libjpeg by a significant amount.

The default is to use libjpeg-turb, to opt-out, use build tag `nolibjpeg`.

To install libjpeg-turbo, refer to [libjpeg-turbo](https://github.com/libjpeg-turbo/libjpeg-turbo/releases).

Linux

```
  cd /tmp
  wget http://downloads.sourceforge.net/project/libjpeg-turbo/$1/libjpeg-turbo-official_$1_amd64.deb
  dpkg -x libjpeg-turbo-official_$1_amd64.deb /tmp/libjpeg-turbo-official
  mv /tmp/libjpeg-turbo-official/opt/libjpeg-turbo /tmp/libjpeg-turbo
```

macOS

```
brew install jpeg-turbo
```

## External services

MLModelScope relies on a few external services.
These services provide tracing, registry, and database servers.

#### Installing Docker

[Install Docker](https://docs.docker.com/engine/installation/). An easy way is using

```
curl -fsSL get.docker.com -o get-docker.sh | sudo sh
sudo usermod -aG docker $USER
```

#### Configuration

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

#### Starting Trace Server

-   On x86 (e.g. intel) machines, start [jaeger](http://jaeger.readthedocs.io/en/latest/getting_started/) by

```
docker run -d -e COLLECTOR_ZIPKIN_HTTP_PORT=9411 -p5775:5775/udp -p6831:6831/udp -p6832:6832/udp \
  -p5778:5778 -p16686:16686 -p14268:14268 -p9411:9411 jaegertracing/all-in-one:latest
```

-   On ppc64le (e.g. minsky) machines, start [jaeger](http://jaeger.readthedocs.io/en/latest/getting_started/) machine by

```
docker run -d -e COLLECTOR_ZIPKIN_HTTP_PORT=9411 -p5775:5775/udp -p6831:6831/udp -p6832:6832/udp \
  -p5778:5778 -p16686:16686 -p14268:14268 -p9411:9411 MLModelScope/jaeger:ppc64le-latest
```

The trace server runs on http://localhost:16686

#### Starting Registry Server

-   On x86 (e.g. intel) machines, start [consul](https://hub.docker.com/_/consul/) by

```
docker run -p 8500:8500 -p 8600:8600 -d consul
```

-   On ppc64le (e.g. minsky) machines, start [consul](https://hub.docker.com/_/consul/) by

```
docker run -p 8500:8500 -p 8600:8600 -d MLModelScope/consul:ppc64le-latest
```

The registry server runs on http://localhost:8500

#### Starting Database Server

-   On x86 (e.g. intel) machines, start [mongodb](https://hub.docker.com/_/mongo/) by

```
docker run -p 27017:27017 --restart always -d mongo:3.0
```

You can also mount the database volume to a local directory using

```
docker run -p 27017:27017 --restart always -d  -v $HOME/data/MLModelScope/mongo:/data/db mongo:3.0
```

## Usage

Run the agent with GPU enabled

```
cd $GOPATH/src/github.com/rai-project/pytorch
go run pytorch-agent/main.go -l -d -v
```

Run the agent ithout GPU or libjpeg-turbo

```
cd $GOPATH/src/github.com/rai-project/pytorch
go run -tags="nogpu nolibjpeg" pytorch-agent/main.go -l -d -v
```
