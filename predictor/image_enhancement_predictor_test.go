package predictor

import (
	"context"
	"image"
	"image/color"
	"image/jpeg"
	"os"
	"path/filepath"
	"testing"

	"github.com/rai-project/dlframework"

	"github.com/rai-project/dlframework/framework/options"
	raiimage "github.com/rai-project/image"
	"github.com/rai-project/image/types"
	nvidiasmi "github.com/rai-project/nvidia-smi"
	py "github.com/rai-project/pytorch"
	"github.com/stretchr/testify/assert"
	gotensor "gorgonia.org/tensor"
)

func TestImageEnhancement(t *testing.T) {
	py.Register()
	model, err := py.FrameworkManifest.FindModel("srgan_v1.0:1.0")
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

	predictor, err := NewImageEnhancementPredictor(*model, options.WithOptions(opts))
	assert.NoError(t, err)
	assert.NotEmpty(t, predictor)
	defer predictor.Close()

	imgDir, _ := filepath.Abs("./_fixtures")
	imgPath := filepath.Join(imgDir, "penguin.png")
	r, err := os.Open(imgPath)
	if err != nil {
		panic(err)
	}

	preprocessOpts, err := predictor.GetPreprocessOptions()
	assert.NoError(t, err)
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

	channels := 3
	height := img.Bounds().Dy()
	width := img.Bounds().Dx()

	input := make([]*gotensor.Dense, batchSize)
	imgFloats, err := normalizeImageCHW(img, preprocessOpts.MeanImage, preprocessOpts.Scale)
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
		panic(err)
	}

	f, ok := pred[0][0].Feature.(*dlframework.Feature_RawImage)
	if !ok {
		panic("expecting an image feature")
	}

	fl := f.RawImage.GetFloatList()
	outWidth := f.RawImage.GetWidth()
	outHeight := f.RawImage.GetHeight()
	offset := 0
	outImg := types.NewRGBImage(image.Rect(0, 0, int(outWidth), int(outHeight)))
	for h := 0; h < int(outHeight); h++ {
		for w := 0; w < int(outWidth); w++ {
			R := uint8(fl[offset+0])
			G := uint8(fl[offset+1])
			B := uint8(fl[offset+2])
			outImg.Set(w, h, color.RGBA{R, G, B, 255})
			offset += 3
		}
	}

	if false {
		output, err := os.Create("output.jpg")
		if err != nil {
			panic(err)
		}
		defer output.Close()
		err = jpeg.Encode(output, outImg, nil)
		if err != nil {
			panic(err)
		}
	}

	assert.Equal(t, int32(1356), outHeight)
	assert.Equal(t, int32(2040), outWidth)
	assert.Equal(t, types.RGB{
		R: 0xc2,
		G: 0xc2,
		B: 0xc6,
	}, outImg.At(0, 0))
}
