package predictor

// import (
// 	"context"
// 	"os"
// 	"path/filepath"
// 	"testing"

// 	"github.com/rai-project/dlframework/framework/options"
// 	"github.com/rai-project/image"
// 	"github.com/rai-project/image/types"
// 	nvidiasmi "github.com/rai-project/nvidia-smi"
// 	py "github.com/rai-project/pytorch"
// 	"github.com/stretchr/testify/assert"
// 	gotensor "gorgonia.org/tensor"
// )

// func TestInstanceSegmentation(t *testing.T) {
// 	py.Register()
// 	model, err := py.FrameworkManifest.FindModel("mask_rcnn_inception_v2_coco:1.0")
// 	assert.NoError(t, err)
// 	assert.NotEmpty(t, model)

// 	device := options.CPU_DEVICE
// 	if nvidiasmi.HasGPU {
// 		device = options.CUDA_DEVICE
// 	}

// 	batchSize := 1
// 	ctx := context.Background()
// 	opts := options.New(options.Context(ctx),
// 		options.Device(device, 0),
// 		options.BatchSize(batchSize))

// 	predictor, err := NewInstanceSegmentationPredictor(*model, options.WithOptions(opts))
// 	assert.NoError(t, err)
// 	assert.NotEmpty(t, predictor)
// 	defer predictor.Close()

// 	imgDir, _ := filepath.Abs("./_fixtures")
// 	imgPath := filepath.Join(imgDir, "lane_control.jpg")
// 	r, err := os.Open(imgPath)
// 	if err != nil {
// 		panic(err)
// 	}
// 	img, err := image.Read(r)
// 	if err != nil {
// 		panic(err)
// 	}

// 	height := img.Bounds().Dy()
// 	width := img.Bounds().Dx()
// 	channels := 3
// 	input := make([]*gotensor.Dense, batchSize)
// 	imgBytes := img.(*types.RGBImage).Pix

// 	for ii := 0; ii < batchSize; ii++ {
// 		input[ii] = gotensor.New(
// 			gotensor.WithShape(height, width, channels),
// 			gotensor.WithBacking(imgBytes),
// 		)
// 	}

// 	err = predictor.Predict(ctx, input)
// 	assert.NoError(t, err)
// 	if err != nil {
// 		return
// 	}

// 	pred, err := predictor.ReadPredictedFeatures(ctx)
// 	assert.NoError(t, err)
// 	if err != nil {
// 		return
// 	}

// 	assert.InDelta(t, float32(0.998607), pred[0][0].GetProbability(), 0.001)
// }
