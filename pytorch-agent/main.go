package main

import (
	"fmt"
	"os"

	cmd "github.com/rai-project/dlframework/framework/cmd/server"
	"github.com/rai-project/pytorch"
	_ "github.com/rai-project/pytorch/predictor"
	"github.com/rai-project/tracer"
)

func main() {
	rootCmd, err := cmd.NewRootCommand(pytorch.Register, pytorch.FrameworkManifest)
	if err != nil {
		fmt.Println(err)
		os.Exit(-1)
	}

	defer tracer.Close()
	if err := rootCmd.Execute(); err != nil {
		fmt.Println(err)
		os.Exit(-1)
	}
}
