package predictor

import (
	"context"
	"image"
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

func normalizeImageHWC(in0 image.Image, mean []float32, scale float32) ([]float32, error) {
	height := in0.Bounds().Dy()
	width := in0.Bounds().Dx()
	out := make([]float32, 3*height*width)
	switch in := in0.(type) {
	case *types.RGBImage:
		for y := 0; y < height; y++ {
			for x := 0; x < width; x++ {
				offset := y*in.Stride + x*3
				rgb := in.Pix[offset : offset+3]
				r, g, b := rgb[0], rgb[1], rgb[2]
				out[offset+0] = ((float32(r>>8) / 255.0) - (mean[0] / 255.0)) / (scale / 255.0)
				out[offset+1] = ((float32(g>>8) / 255.0) - (mean[1] / 255.0)) / (scale / 255.0)
				out[offset+2] = ((float32(b>>8) / 255.0) - (mean[2] / 255.0)) / (scale / 255.0)
			}
		}
	case *types.BGRImage:
		for y := 0; y < height; y++ {
			for x := 0; x < width; x++ {
				offset := y*in.Stride + x*3
				bgr := in.Pix[offset : offset+3]
				b, g, r := bgr[0], bgr[1], bgr[2]
				out[offset+0] = ((float32(b>>8) / 255.0) - (mean[0] / 255.0)) / (scale / 255.0)
				out[offset+0] = ((float32(g>>8) / 255.0) - (mean[1] / 255.0)) / (scale / 255.0)
				out[offset+0] = ((float32(r>>8) / 255.0) - (mean[2] / 255.0)) / (scale / 255.0)

			}
		}
	default:
		panic("unreachable")
	}

	return out, nil
}

func normalizeImageCHW(in *types.RGBImage, mean []float32, stddev []float32) ([]float32, error) {
	height := in.Bounds().Dy()
	width := in.Bounds().Dx()
	out := make([]float32, 3*height*width)
	for y := 0; y < height; y++ {
		for x := 0; x < width; x++ {
			offset := y*in.Stride + x*3
			rgb := in.Pix[offset : offset+3]
			r, g, b := rgb[0], rgb[1], rgb[2]
			out[y*width+x] = ((float32(r>>8) / 255.0) - mean[0]) / stddev[0]
			out[width*height+y*width+x] = ((float32(g>>8) / 255.0) - mean[1]) / stddev[1]
			out[2*width*height+y*width+x] = ((float32(b>>8) / 255.0) - mean[2]) / stddev[2]
		}
	}
	return out, nil
}

func TestPredictorNew(t *testing.T) {
	py.Register()
	model, err := py.FrameworkManifest.FindModel("torchvision_alexnet:1.0")
	assert.NoError(t, err)
	assert.NotEmpty(t, model)

	predictor, err := NewImageClassificationPredictor(*model)
	assert.NoError(t, err)
	assert.NotEmpty(t, predictor)

	defer predictor.Close()

	_, ok := predictor.(*ImageClassificationPredictor)
	assert.True(t, ok)
}

func TestImageClassification(t *testing.T) {
	py.Register()
	model, err := py.FrameworkManifest.FindModel("torchvision_alexnet:1.0")
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

	predictor, err := NewImageClassificationPredictor(*model, options.WithOptions(opts))
	assert.NoError(t, err)
	assert.NotEmpty(t, predictor)
	defer predictor.Close()

	imgDir, _ := filepath.Abs("./_fixtures")
	imgPath := filepath.Join(imgDir, "platypus.jpg")
	r, err := os.Open(imgPath)
	if err != nil {
		panic(err)
	}

	preprocessOpts, err := predictor.GetPreprocessOptions(ctx)
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

	pp.Println("Prediction: ", pred[0][0].GetClassification().GetIndex())
	pp.Println("Probability: ", pred[0][0].GetProbability())

}
