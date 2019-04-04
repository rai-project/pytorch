package pytorch

import (
	"os"

	assetfs "github.com/elazarl/go-bindata-assetfs"
	"github.com/rai-project/dlframework"
	"github.com/rai-project/dlframework/framework"
)

// FrameworkManifest ...
var FrameworkManifest = dlframework.FrameworkManifest{
	Name:    "PyTorch",
	Version: "1.0",
	Container: map[string]*dlframework.ContainerHardware{
		"amd64": {
			Cpu: "raiproject/carml-pytorch:amd64-cpu",
			Gpu: "raiproject/carml-pytorch:amd64-gpu",
		},
		"ppc64le": {
			Cpu: "raiproject/carml-pytorch:ppc64le-gpu",
			Gpu: "raiproject/carml-pytorch:ppc64le-gpu",
		},
	},
}

func assetFS() *assetfs.AssetFS {
	assetInfo := func(path string) (os.FileInfo, error) {
		return os.Stat(path)
	}
	for k := range _bintree.Children {
		return &assetfs.AssetFS{Asset: Asset, AssetDir: AssetDir, AssetInfo: assetInfo, Prefix: k}
	}
	panic("unreachable")
}

func Register() {
	err := framework.Register(FrameworkManifest, assetFS())
	if err != nil {
		log.WithError(err).Error("Failed to register server")
    }
}
