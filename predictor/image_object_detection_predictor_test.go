package predictor

import (
	"context"
	"os"
	"path/filepath"
	"testing"

	"github.com/k0kubun/pp"
	"github.com/rai-project/dlframework/framework/options"
	raiimage "github.com/rai-project/image"
	"github.com/rai-project/image/types"
	nvidiasmi "github.com/rai-project/nvidia-smi"
	py "github.com/rai-project/pytorch"
	"github.com/stretchr/testify/assert"
	gotensor "gorgonia.org/tensor"
)

func TestObjectDetection(t *testing.T) {
	py.Register()
	model, err := py.FrameworkManifest.FindModel("mobilenet_ssd_v1.0:1.0")
	assert.NoError(t, err)
	assert.NotEmpty(t, model)

	device := options.CPU_DEVICE
	if nvidiasmi.HasGPU {
		device = options.CUDA_DEVICE
	}

	batchSize := 1
	ctx := context.Background()
	opts := options.New(options.Context(ctx),
		options.Device(device, 0),
		options.BatchSize(batchSize))

	predictor, err := NewObjectDetectionPredictor(*model, options.WithOptions(opts))
	assert.NoError(t, err)
	assert.NotEmpty(t, predictor)
	defer predictor.Close()

	imgDir, _ := filepath.Abs("./_fixtures")
	imgPath := filepath.Join(imgDir, "lane_control.jpg")
	r, err := os.Open(imgPath)
	if err != nil {
		panic(err)
	}

	preprocessOpts, err := predictor.GetPreprocessOptions()
	assert.NoError(t, err)
	channels := preprocessOpts.Dims[0]
	height := preprocessOpts.Dims[1]
	width := preprocessOpts.Dims[2]
	mode := preprocessOpts.ColorMode

	var imgOpts []raiimage.Option
	if mode == types.RGBMode {
		imgOpts = append(imgOpts, raiimage.Mode(types.RGBMode))
	} else {
		imgOpts = append(imgOpts, raiimage.Mode(types.BGRMode))
	}

	img, err := raiimage.Read(r, imgOpts...)
	if err != nil {
		panic(err)
	}

	imgOpts = append(imgOpts, raiimage.Resized(height, width))
	imgOpts = append(imgOpts, raiimage.ResizeAlgorithm(types.ResizeAlgorithmLinear))
	resized, err := raiimage.Resize(img, imgOpts...)

	input := make([]*gotensor.Dense, batchSize)
	imgFloats, err := normalizeImageHWC(resized, preprocessOpts.MeanImage, preprocessOpts.Scale)
	if err != nil {
		panic(err)
	}

	for ii := 0; ii < batchSize; ii++ {
		input[ii] = gotensor.New(
			gotensor.WithShape(channels, height, width),
			gotensor.WithBacking(imgFloats),
		)
	}

	err = predictor.Predict(ctx, input)
	assert.NoError(t, err)
	if err != nil {
		return
	}

	pred, err := predictor.ReadPredictedFeatures(ctx)
	assert.NoError(t, err)
	if err != nil {
		return
	}

	pp.Println("Prediction: ", pred[0][0].GetProbability())

}
